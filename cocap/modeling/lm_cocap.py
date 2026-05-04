# -*- coding: utf-8 -*-

import copy
import logging
import math
import os
from collections import defaultdict
from typing import Any

import pytorch_lightning as pl
import torch
import torch.distributed as dist
import torch.nn as nn
from hydra_zen import builds
from pytorch_lightning.utilities.types import OptimizerLRScheduler
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from transformers import GPT2Tokenizer

from cocap.modules.compressed_video.motion_encoder import MotionTransformer
from cocap.modules.focalcap_phase2 import FocalCapPhase2
from cocap.modeling.loss import Phase2CaptionLoss, phase2_caption_loss_cfg
from cocap.modeling.eval_captioning import evaluate
from cocap.utils.json import save_json
from cocap.utils.train_utils import gather_object_multiple_gpu, get_timestamp

logger = logging.getLogger(__name__)


class CoCapLM(pl.LightningModule):
    """Strict Phase-2 one-shot caption training (Action Encoder + AGDTR + GPT-2 LoRA)."""

    def __init__(
        self,
        cocap_model: nn.Module,
        loss: Phase2CaptionLoss,
        use_preextracted_features: bool = True,
        init_motion_ckpt: str = "",
        lr_scratch: float = 1e-4,
        lr_lora: float = 2e-4,
        weight_decay: float = 0.05,
        warmup_ratio: float = 0.05,
        min_lr_ratio: float = 0.10,
        max_caption_len_train: int = 50,
        max_new_tokens_eval: int = 60,
    ):
        super().__init__()
        self.model = cocap_model
        self.loss = loss
        self.use_preextracted_features = bool(use_preextracted_features)
        self.init_motion_ckpt = init_motion_ckpt

        self.lr_scratch = float(lr_scratch)
        self.lr_lora = float(lr_lora)
        self.weight_decay = float(weight_decay)
        self.warmup_ratio = float(warmup_ratio)
        self.min_lr_ratio = float(min_lr_ratio)

        self.max_caption_len_train = int(max_caption_len_train)
        self.max_new_tokens_eval = int(max_new_tokens_eval)

        base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
        gpt2_path = os.path.join(base_dir, "model_zoo", "gpt2_model")
        self.tokenizer = GPT2Tokenizer.from_pretrained(gpt2_path, local_files_only=True)
        self.tokenizer.pad_token = self.tokenizer.eos_token

        self.batch_res = None
        self._best_val_cider = None

    def on_fit_start(self) -> None:
        if self.init_motion_ckpt and os.path.isfile(self.init_motion_ckpt):
            self._load_motion_ckpt(self.init_motion_ckpt)
            logger.info("Initialized motion encoder from %s", self.init_motion_ckpt)
            return

        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
        candidates = [
            os.path.join(project_root, "logs", "vatex_pretrain", "motion_encoder_best.pt"),
            os.path.join(project_root, "logs", "vatex_captioning", "phase1", "checkpoints", "modules", "motion_encoder_best.pt"),
            os.path.join(
                os.path.dirname(project_root),
                "Distilled-Motion-MAE",
                "logs",
                "vatex_captioning",
                "phase1",
                "checkpoints",
                "modules",
                "motion_encoder_best.pt",
            ),
        ]
        for default_p in candidates:
            if os.path.isfile(default_p):
                self._load_motion_ckpt(default_p)
                logger.info("Auto-resolved init_motion_ckpt=%s", default_p)
                return

    def _load_motion_ckpt(self, ckpt_path: str) -> None:
        raw = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        state = raw.get("state_dict", raw)

        normalized = {}
        for k, v in state.items():
            nk = k
            for p in ("model.motion_encoder.", "module.model.motion_encoder.", "motion_encoder."):
                if nk.startswith(p):
                    nk = nk[len(p):]
                    break
            normalized[nk] = v

        student_state = {k[len("student."):]: v for k, v in normalized.items() if k.startswith("student.")}
        residual_state = {k[len("residual_student."):]: v for k, v in normalized.items() if k.startswith("residual_student.")}

        if not student_state:
            raise RuntimeError(
                f"No 'student.*' weights found in {ckpt_path}. "
                "Expected Phase-1 motion_encoder_best.pt with student/residual_student prefixes."
            )

        m_miss, m_unexp = self.model.motion_encoder.student.load_state_dict(student_state, strict=True)
        logger.info("Loaded motion student from %s (missing=%d unexpected=%d)", ckpt_path, len(m_miss), len(m_unexp))

        if getattr(self.model.motion_encoder, "residual_student", None) is not None:
            if not residual_state:
                raise RuntimeError(
                    f"No 'residual_student.*' weights found in {ckpt_path}. "
                    "Your Phase-2 spec requires freezing both motion and residual encoders."
                )
            r_miss, r_unexp = self.model.motion_encoder.residual_student.load_state_dict(residual_state, strict=True)
            logger.info("Loaded residual student from %s (missing=%d unexpected=%d)", ckpt_path, len(r_miss), len(r_unexp))

        # Enforce freeze/eval after load.
        for p in self.model.motion_encoder.student.parameters():
            p.requires_grad = False
        self.model.motion_encoder.student.eval()
        if getattr(self.model.motion_encoder, "residual_student", None) is not None:
            for p in self.model.motion_encoder.residual_student.parameters():
                p.requires_grad = False
            self.model.motion_encoder.residual_student.eval()

    @property
    def total_steps(self) -> int:
        return int(getattr(self.trainer, "estimated_stepping_batches", 0))

    def configure_optimizers(self) -> OptimizerLRScheduler:
        scratch_decay, scratch_no_decay = [], []
        lora_params = []

        for name, p in self.model.named_parameters():
            if not p.requires_grad:
                continue
            if "lora_" in name:
                lora_params.append(p)
                continue

            is_no_decay = (
                p.ndim == 1
                or name.endswith(".bias")
                or "LayerNorm" in name
                or "layernorm" in name.lower()
                or "ln_" in name
                or "norm" in name.lower()
            )
            if is_no_decay:
                scratch_no_decay.append(p)
            else:
                scratch_decay.append(p)

        param_groups = []
        if scratch_decay:
            param_groups.append({"params": scratch_decay, "lr": self.lr_scratch, "weight_decay": self.weight_decay})
        if scratch_no_decay:
            param_groups.append({"params": scratch_no_decay, "lr": self.lr_scratch, "weight_decay": 0.0})
        if lora_params:
            param_groups.append({"params": lora_params, "lr": self.lr_lora, "weight_decay": 0.0})

        optimizer = AdamW(param_groups, eps=1e-8, betas=(0.9, 0.999))

        total_steps = max(1, self.total_steps)
        warmup_steps = max(1, int(total_steps * self.warmup_ratio))
        min_ratio = max(0.0, min(1.0, self.min_lr_ratio))

        def lr_lambda(step: int) -> float:
            if step < warmup_steps:
                return float(step + 1) / float(warmup_steps)
            progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
            progress = max(0.0, min(1.0, progress))
            cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
            return min_ratio + (1.0 - min_ratio) * cosine

        scheduler = LambdaLR(optimizer, lr_lambda=lr_lambda)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
                "name": "warmup_cosine",
            },
        }

    def _run_model(self, batch: dict) -> dict:
        video = batch.get("video", batch)

        motion_vectors = video.get("motion_vectors", video.get("motion_vector", None))
        if motion_vectors is None:
            raise RuntimeError("Missing motion vectors in batch.")

        input_mask_mv = video.get("input_mask_mv", None)
        if input_mask_mv is None:
            bsz, gops, num_mv = motion_vectors.shape[:3]
            input_mask_mv = torch.zeros((bsz, gops, num_mv), dtype=torch.long, device=motion_vectors.device)

        clip_i_cls = video.get("clip_i_cls", None)
        clip_i_spatial = video.get("clip_i_spatial", None)

        if clip_i_cls is None or clip_i_spatial is None:
            if self.use_preextracted_features:
                raise RuntimeError("use_preextracted_features=True but clip_i_cls/clip_i_spatial missing.")

        return self.model(
            clip_i_cls=clip_i_cls,
            clip_i_spatial=clip_i_spatial,
            motion_vectors=motion_vectors,
            input_ids=batch["input_ids"],
            attention_mask=batch["input_mask"],
            labels=batch["input_labels"],
            input_mask_mv=input_mask_mv,
            input_mask_gop=video.get("input_mask_gop", None),
            residual=video.get("residual", None),
            input_mask_res=video.get("input_mask_res", None),
            iframe=video.get("iframe", None),
        )

    def training_step(self, batch, batch_idx):
        del batch_idx
        output = self._run_model(batch)
        loss = self.loss(target=batch, output=output)

        bsz = batch["input_labels"].size(0)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True, batch_size=bsz, sync_dist=True)

        # Surface a few AGDTR + Action diagnostics on the tqdm bar so a glance
        # tells you whether routing is collapsing, gates are dead, or the action
        # stream is producing garbage. The rest go to TensorBoard / CSV only.
        bar_keys = {"phase2/gate_mean", "phase2/budget_std", "phase2/patch_diversity"}
        comp = getattr(self.loss, "latest_loss_components", {})
        if isinstance(comp, dict):
            for k, v in comp.items():
                if torch.is_tensor(v):
                    self.log(
                        k,
                        v,
                        on_step=True,
                        on_epoch=True,
                        prog_bar=(k in bar_keys),
                        logger=True,
                        batch_size=bsz,
                        sync_dist=True,
                    )

        return loss

    def validation_step(self, batch, batch_idx):
        del batch_idx
        output = self._run_model(batch)
        val_loss = self.loss(target=batch, output=output)
        bsz = batch["input_labels"].size(0)
        self.log("val_loss", val_loss, on_step=False, on_epoch=True, prog_bar=True, logger=True, batch_size=bsz, sync_dist=True)

        video = batch.get("video", batch)
        motion_vectors = video.get("motion_vectors", video.get("motion_vector", None)).float()
        input_mask_mv = video.get("input_mask_mv", None)
        if input_mask_mv is None:
            bsz_, gops_, num_mv_ = motion_vectors.shape[:3]
            input_mask_mv = torch.zeros((bsz_, gops_, num_mv_), dtype=torch.long, device=motion_vectors.device)

        gen_ids = self.model.generate(
            clip_i_cls=video.get("clip_i_cls", None),
            clip_i_spatial=video.get("clip_i_spatial", None),
            motion_vectors=motion_vectors,
            input_mask_mv=input_mask_mv,
            input_mask_gop=video.get("input_mask_gop", None),
            residual=video.get("residual", None),
            input_mask_res=video.get("input_mask_res", None),
            iframe=video.get("iframe", None),
            max_new_tokens=self.max_new_tokens_eval,
            min_new_tokens=5,
            num_beams=4,
            no_repeat_ngram_size=3,
            length_penalty=1.0,
        )
        gen_texts = self.tokenizer.batch_decode(gen_ids, skip_special_tokens=True)

        # Generation health: did beams actually emit EOS, or did they all hit the
        # max_new_tokens cap? And how long are the captions on average?
        eos_id = int(self.tokenizer.eos_token_id)
        eos_hits = (gen_ids == eos_id).any(dim=1).float().mean()
        gen_lengths = (gen_ids != eos_id).sum(dim=1).float().mean()
        self.log("gen_eos_rate", eos_hits, on_step=False, on_epoch=True, prog_bar=True, logger=True, batch_size=bsz, sync_dist=True)
        self.log("gen_mean_len", gen_lengths, on_step=False, on_epoch=True, prog_bar=True, logger=True, batch_size=bsz, sync_dist=True)

        if self.batch_res is None:
            self.batch_res = {
                "version": "VERSION 1.0",
                "results": defaultdict(list),
                "external_data": {"used": "true", "details": "focalcap_phase2"},
            }

        video_ids = batch["metadata"][0]
        gt_texts = batch["metadata"][1]
        for i, pred in enumerate(gen_texts):
            vid = video_ids[i]
            self.batch_res["results"][vid].append({"sentence": pred.strip(), "gt_sentence": gt_texts[i]})

    def on_validation_epoch_start(self) -> None:
        self.batch_res = {
            "version": "VERSION 1.0",
            "results": defaultdict(list),
            "external_data": {"used": "true", "details": "focalcap_phase2"},
        }

    def on_validation_epoch_end(self) -> None:
        if self.batch_res is None:
            return

        json_res = copy.deepcopy(self.batch_res)
        if dist.is_initialized():
            all_results = gather_object_multiple_gpu(list(json_res["results"].items()))
            json_res["results"] = {k: v for k, v in all_results}

        if (not dist.is_initialized()) or dist.get_rank() == 0:
            res_filepath = os.path.join(self.trainer.default_root_dir, f"caption_pred_val_{get_timestamp()}.json")
            os.makedirs(os.path.dirname(res_filepath), exist_ok=True)
            save_json(json_res, res_filepath, save_pretty=True)

            val_dl = self.trainer.val_dataloaders
            if isinstance(val_dl, list):
                val_dl = val_dl[0]
            json_ref = getattr(val_dl.dataset, "json_ref", None)

            if json_ref and json_res.get("results"):
                metrics = evaluate(json_res, json_ref)
                self.log_dict(metrics, on_step=False, on_epoch=True, logger=True, sync_dist=False)
                pretty = {k: round(float(v), 4) for k, v in metrics.items()}
                self.print(
                    f"\n================ Validation Metrics ================\n"
                    f"{pretty}\n"
                    f"=================================================="
                )
                cider = metrics.get("CIDEr", metrics.get("cider", None))
                if cider is not None:
                    self.log("val_CIDEr", float(cider), on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=False)
                    if self._best_val_cider is None or float(cider) > self._best_val_cider:
                        self._best_val_cider = float(cider)
                        logger.info("New best val_CIDEr: %.4f", self._best_val_cider)
                        mod_dir = os.path.join(self.trainer.default_root_dir, "checkpoints", "modules")
                        os.makedirs(mod_dir, exist_ok=True)
                        payload = {
                            "phase": 2,
                            "epoch": int(self.current_epoch),
                            "step": int(self.global_step),
                            "best_val_cider": float(self._best_val_cider),
                            "state_dict": self.model.motion_encoder.state_dict(),
                        }
                        torch.save(payload, os.path.join(mod_dir, "motion_encoder_best.pt"))
                        if hasattr(self.model, "patch_router"):
                            torch.save(
                                {
                                    "phase": 2,
                                    "epoch": int(self.current_epoch),
                                    "step": int(self.global_step),
                                    "best_val_cider": float(self._best_val_cider),
                                    "state_dict": self.model.patch_router.state_dict(),
                                },
                                os.path.join(mod_dir, "router_best.pt"),
                            )

        if dist.is_initialized():
            dist.barrier()

    def on_train_epoch_end(self) -> None:
        if dist.is_initialized() and dist.get_rank() != 0:
            return
        mod_dir = os.path.join(self.trainer.default_root_dir, "checkpoints", "modules")
        phase_dir = os.path.join(mod_dir, "phase2")
        os.makedirs(mod_dir, exist_ok=True)
        os.makedirs(phase_dir, exist_ok=True)
        ep = int(self.current_epoch)
        st = int(self.global_step)

        motion_payload = {
            "phase": 2,
            "epoch": ep,
            "step": st,
            "state_dict": self.model.motion_encoder.state_dict(),
        }
        torch.save(motion_payload, os.path.join(mod_dir, "motion_encoder_last.pt"))
        torch.save(motion_payload, os.path.join(phase_dir, f"motion_encoder-epoch{ep:03d}-step{st}.pt"))

        if hasattr(self.model, "patch_router"):
            router_payload = {
                "phase": 2,
                "epoch": ep,
                "step": st,
                "state_dict": self.model.patch_router.state_dict(),
            }
            torch.save(router_payload, os.path.join(mod_dir, "router_last.pt"))
            torch.save(router_payload, os.path.join(phase_dir, f"router-epoch{ep:03d}-step{st}.pt"))


_BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

focalcap_phase2_cfg = builds(
    FocalCapPhase2,
    gpt2_model_path=os.path.join(_BASE_DIR, "model_zoo", "gpt2_model"),
    clip_state_dict=os.path.join(_BASE_DIR, "model_zoo", "clip_model", "ViT-B-16.pt"),
    motion_encoder=builds(MotionTransformer, populate_full_signature=True),
    populate_full_signature=True,
)

cocap_lm_cfg = builds(
    CoCapLM,
    cocap_model=focalcap_phase2_cfg,
    loss=phase2_caption_loss_cfg,
    populate_full_signature=True,
)
