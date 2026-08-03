# -*- coding: utf-8 -*-

import copy
import io
import logging
import math
import os
import time
from contextlib import redirect_stderr, redirect_stdout
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
        lr_motion_scale: float = 0.10,
        train_motion_encoder: bool = True,
        weight_decay: float = 0.05,
        warmup_ratio: float = 0.05,
        min_lr_ratio: float = 0.10,
        max_caption_len_train: int = 50,
        max_new_tokens_eval: int = 60,
        gpu_cooldown_train_step_sec: float = 0.0,
        gpu_cooldown_val_step_sec: float = 0.0,
    ):
        super().__init__()
        self.model = cocap_model
        self.loss = loss
        self.use_preextracted_features = bool(use_preextracted_features)
        self.init_motion_ckpt = init_motion_ckpt

        # With pre-extracted features the visual tokens (CLS + spatial) come from
        # the .pt files, so the frozen CLIP/SigLIP2 tower is never called even in
        # the full AGDTR pipeline. Drop it here — before Lightning moves the
        # module to the GPU — to free VRAM for the motion/residual/action stack.
        if self.use_preextracted_features:
            dropped = []
            if getattr(self.model, "clip", None) is not None:
                self.model.clip = None
                dropped.append("clip")
            if getattr(self.model, "siglip", None) is not None:
                self.model.siglip = None
                dropped.append("siglip")
            if dropped:
                logger.info("use_preextracted_features=True -> dropped unused vision tower(s): %s", ", ".join(dropped))

        self.lr_scratch = float(lr_scratch)
        self.lr_lora = float(lr_lora)
        self.lr_motion_scale = float(lr_motion_scale)
        self.train_motion_encoder = bool(train_motion_encoder)
        self.weight_decay = float(weight_decay)
        self.warmup_ratio = float(warmup_ratio)
        self.min_lr_ratio = float(min_lr_ratio)

        self.max_caption_len_train = int(max_caption_len_train)
        self.max_new_tokens_eval = int(max_new_tokens_eval)
        self.gpu_cooldown_train_step_sec = max(0.0, float(gpu_cooldown_train_step_sec))
        self.gpu_cooldown_val_step_sec = max(0.0, float(gpu_cooldown_val_step_sec))

        base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
        gpt2_path = os.path.join(base_dir, "model_zoo", "gpt2_model")
        self.tokenizer = GPT2Tokenizer.from_pretrained(gpt2_path, local_files_only=True)
        self.tokenizer.pad_token = self.tokenizer.eos_token

        self.batch_res = None
        self._best_val_cider = None
        self._probe_train_sum = defaultdict(float)
        self._probe_train_cnt = defaultdict(int)
        self._probe_val_sum = defaultdict(float)
        self._probe_val_cnt = defaultdict(int)

    @staticmethod
    def _probe_add(sum_store: dict, cnt_store: dict, key: str, value: Any) -> None:
        if value is None:
            return
        if torch.is_tensor(value):
            if value.numel() == 0:
                return
            value = float(value.detach().float().mean().cpu())
        else:
            value = float(value)
        if math.isfinite(value):
            sum_store[key] += value
            cnt_store[key] += 1

    @staticmethod
    def _probe_avg(sum_store: dict, cnt_store: dict, key: str) -> float:
        n = int(cnt_store.get(key, 0))
        if n <= 0:
            return float("nan")
        return float(sum_store.get(key, 0.0)) / float(n)

    def _probe_accumulate(self, phase: str, output: dict) -> None:
        if phase == "train":
            sum_store, cnt_store = self._probe_train_sum, self._probe_train_cnt
        else:
            sum_store, cnt_store = self._probe_val_sum, self._probe_val_cnt

        keys = (
            "attn_cls_mass", "attn_act_mass", "attn_patch_mass",
            "attn_cls_per_tok", "attn_act_per_tok", "attn_patch_per_tok",
            "attn_act_r0", "attn_act_r1", "attn_act_r2", "attn_act_r3",
            "cls_norm_post", "act_norm_post", "patch_norm_post",
            "gate_mean", "budget_std", "patch_diversity",
            "motion_within_pairc", "residual_within_pairc",
            "raw_action_within_pairc", "action_within_pairc", "action_inter_pairc", "cls_within_pairc", "patch_within_pairc",
            "ae_region_pairc", "ae_region_max_pairc",
            "ae_region_cos_01", "ae_region_cos_02", "ae_region_cos_03",
            "ae_region_cos_12", "ae_region_cos_13", "ae_region_cos_23",
            "ae_dyn_h0_attn", "ae_dyn_h1_attn", "ae_dyn_balance", "ae_dyn_entropy",
            "ae_dyn_out_norm", "ae_scene_out_norm",
            "ae_half_cos", "ae_delta_norm",
            "ae_scene_top5_r0", "ae_scene_top5_r1", "ae_scene_top5_r2", "ae_scene_top5_r3",
            "ae_scene_focus", "ae_scene_gate",
            "action_dyn_cos",
            "action_common_norm", "action_detail_norm", "action_detail_ratio",
            "action_cls_cos", "post_action_cls_cos",
            "action_own_patch_cos", "action_other_patch_cos", "action_patch_copy_gap",
            "fuser_gate", "fuser_delta_norm", "fuser_entropy",
            "fuser_attn_r0", "fuser_attn_r1", "fuser_attn_r2", "fuser_attn_r3",
            "router_cls_act_agree", "router_patch_top10",
            "patch_cls_cos", "patch_action_cos",
        )
        for k in keys:
            self._probe_add(sum_store, cnt_store, k, output.get(k, None))

        cls_m = output.get("attn_cls_mass", None)
        act_m = output.get("attn_act_mass", None)
        pat_m = output.get("attn_patch_mass", None)
        if torch.is_tensor(cls_m):
            cls_m = float(cls_m.detach().float().cpu())
        if torch.is_tensor(act_m):
            act_m = float(act_m.detach().float().cpu())
        if torch.is_tensor(pat_m):
            pat_m = float(pat_m.detach().float().cpu())
        if cls_m is not None and act_m is not None and pat_m is not None:
            total = float(cls_m) + float(act_m) + float(pat_m)
            if math.isfinite(total) and total > 1e-8:
                self._probe_add(sum_store, cnt_store, "attn_share_cls", float(cls_m) / total)
                self._probe_add(sum_store, cnt_store, "attn_share_act", float(act_m) / total)
                self._probe_add(sum_store, cnt_store, "attn_share_patch", float(pat_m) / total)

    def _print_probe_report(self, phase: str) -> None:
        def f(v: float) -> str:
            return "n/a   " if not math.isfinite(v) else f"{v:+.3f} "
        def f5(v: float) -> str:
            return "n/a     " if not math.isfinite(v) else f"{v:+.5f} "

        g = lambda k: self._probe_avg(
            self._probe_train_sum if phase == "train" else self._probe_val_sum,
            self._probe_train_cnt if phase == "train" else self._probe_val_cnt,
            k,
        )
        cls_only = bool(getattr(self.model, "cls_only", False))
        attn_n = int((self._probe_train_cnt if phase == "train" else self._probe_val_cnt).get("attn_share_cls", 0))
        ep = int(self.current_epoch)

        use_agdtr = bool(getattr(self.model, "use_agdtr", True))
        act_tokens_per_gop = int(getattr(self.model, "action_prefix_tokens_per_gop", 1))
        action_prefix_mode = str(getattr(self.model, "action_prefix_mode", "region4"))
        if cls_only:
            probe_mode = "cls_only"
        elif use_agdtr:
            probe_mode = "full"
        elif action_prefix_mode == "fused_cls":
            probe_mode = "action-fused-cls"
        elif action_prefix_mode == "interleaved_region4":
            probe_mode = "cls+act-interleaved"
        elif action_prefix_mode == "interleaved_action1":
            probe_mode = "cls+act1-interleaved"
        else:
            probe_mode = "cls+act"
        act_total = 0 if cls_only else 8 * act_tokens_per_gop
        cls_pt = g("attn_cls_per_tok")
        act_pt = g("attn_act_per_tok")
        act_cls_pt_ratio = act_pt / max(cls_pt, 1e-8) if math.isfinite(cls_pt) and math.isfinite(act_pt) else float("nan")
        lines = [
            "",
            f"─── PROBE ({phase.upper()}) Ep{ep:02d}  {probe_mode} ───────",
            "",
            f"  [GPT-2] caption→visual attention share  (n={attn_n})",
            f"    CLS {f(g('attn_share_cls'))} ACT {f(g('attn_share_act'))} PATCH {f(g('attn_share_patch'))}",
            f"    per-token mass: CLS {f5(cls_pt)} ACT {f5(act_pt)} PATCH {f5(g('attn_patch_per_tok'))}  ACT/CLS {f(act_cls_pt_ratio)}",
            f"    ACT by region: TL {f5(g('attn_act_r0'))} TR {f5(g('attn_act_r1'))} BL {f5(g('attn_act_r2'))} BR {f5(g('attn_act_r3'))}",
            f"    token counts: CLS=8(1/GOP)  ACT={act_total}({0 if cls_only else act_tokens_per_gop}/GOP)  PATCH={64 if use_agdtr else 0}",
            "",
            f"  [ACTION-TOKEN DIVERSITY] pairwise cosine ↓ lower = more distinct",
            f"    REGION offdiag mean {f(g('ae_region_pairc'))} max {f(g('ae_region_max_pairc'))}",
            f"    pairs TL-TR {f(g('ae_region_cos_01'))} TL-BL {f(g('ae_region_cos_02'))} TL-BR {f(g('ae_region_cos_03'))}",
            f"          TR-BL {f(g('ae_region_cos_12'))} TR-BR {f(g('ae_region_cos_13'))} BL-BR {f(g('ae_region_cos_23'))}",
            f"    INTER-GOP prefix action {f(g('action_inter_pairc'))}   ← action changes across GOPs?",
            f"    raw region INTRA {f(g('raw_action_within_pairc'))}  prefix/action INTRA {f(g('action_within_pairc'))}  cls/GOP {f(g('cls_within_pairc'))}",
        ]
        if not cls_only:
            lines += [
                "",
                f"  [ACTION ENCODER]  dynamic-seeded region tokens + local scene grounding",
                f"    temporal pool: h0 {f(g('ae_dyn_h0_attn'))} h1 {f(g('ae_dyn_h1_attn'))} balance {f(g('ae_dyn_balance'))} entropy {f(g('ae_dyn_entropy'))}",
                f"    temporal shape: h0-h1 cos {f(g('ae_half_cos'))} delta_norm {f(g('ae_delta_norm'))}",
                f"    branch norms: dyn {f(g('ae_dyn_out_norm'))} scene {f(g('ae_scene_out_norm'))} scene_gate {f(g('ae_scene_gate'))}",
                f"    region spread: common_norm {f(g('action_common_norm'))} detail_norm {f(g('action_detail_norm'))} detail/common {f(g('action_detail_ratio'))}",
                f"    scene top5/49: TL {f(g('ae_scene_top5_r0'))} TR {f(g('ae_scene_top5_r1'))} BL {f(g('ae_scene_top5_r2'))} BR {f(g('ae_scene_top5_r3'))} avg {f(g('ae_scene_focus'))}",
                f"    source similarity: act↔dyn {f(g('action_dyn_cos'))} act↔CLS(pre) {f(g('action_cls_cos'))} act↔CLS(post) {f(g('post_action_cls_cos'))}",
                f"    patch copy: own_quad {f(g('action_own_patch_cos'))} other_quad {f(g('action_other_patch_cos'))} own-other {f(g('action_patch_copy_gap'))}",
                f"    CLS fuser: gate {f(g('fuser_gate'))} delta_norm {f(g('fuser_delta_norm'))} entropy {f(g('fuser_entropy'))}",
                f"      fuser attn: TL {f(g('fuser_attn_r0'))} TR {f(g('fuser_attn_r1'))} BL {f(g('fuser_attn_r2'))} BR {f(g('fuser_attn_r3'))}",
            ]
        if use_agdtr and not cls_only:
            lines += [
                "",
                f"  [AGDTR ROUTER]",
                f"    patch↔cls {f(g('patch_cls_cos'))} patch↔act {f(g('patch_action_cos'))}",
                f"    cls↔act_agree {f(g('router_cls_act_agree'))} router_top10 {f(g('router_patch_top10'))}",
            ]
        lines += [
            "",
            f"  [HEALTH]",
            f"    norms cls {f(g('cls_norm_post'))} act {f(g('act_norm_post'))} patch {f(g('patch_norm_post'))}",
            f"    router_gate {f(g('gate_mean'))} budget_std {f(g('budget_std'))} patch_div {f(g('patch_diversity'))}",
            "─────────────────────────────────────────────────────────────────────────",
        ]
        self.print("\n".join(lines))

    def on_fit_start(self) -> None:
        if self.init_motion_ckpt and os.path.isfile(self.init_motion_ckpt):
            self._load_motion_ckpt(self.init_motion_ckpt)
            logger.info("Initialized motion encoder from %s", self.init_motion_ckpt)
            return

        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
        candidates = [
            os.path.join(project_root, "checkpoints", "phase1_v2", "best-007-0.6796.ckpt"),
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
        motion_state = None
        residual_state_direct = None
        if isinstance(raw, dict) and "motion_encoder" in raw and isinstance(raw["motion_encoder"], dict):
            state = raw
            normalized = {}
            motion_state = raw["motion_encoder"]
            residual_state_direct = raw.get("residual_encoder", {})
        else:
            state = raw.get("state_dict", raw)
            normalized = {}
            residual_state_direct = None
            for k, v in state.items():
                nk = k
                for p in ("model.motion_encoder.", "module.model.motion_encoder.", "motion_encoder."):
                    if nk.startswith(p):
                        nk = nk[len(p):]
                        break
                normalized[nk] = v

        motion_module = getattr(self.model.motion_encoder, "motion_encoder", None)
        residual_module = getattr(self.model.motion_encoder, "residual_encoder", None)

        if motion_module is None:
            motion_module = getattr(self.model.motion_encoder, "student", None)
        if residual_module is None:
            residual_module = getattr(self.model.motion_encoder, "residual_student", None)

        if motion_module is None:
            raise RuntimeError("Phase-2 motion wrapper exposes neither `motion_encoder` nor `student`.")

        if motion_state is None:
            motion_state = {}
            for pref in ("motion_encoder.", "student.", "encoder.", "model.encoder.", "module.encoder."):
                motion_state = {k[len(pref):]: v for k, v in normalized.items() if k.startswith(pref)}
                if motion_state:
                    break

            if not motion_state:
                # direct module state-dict (already stripped to encoder keys)
                if "stem.net.0.weight" in normalized or "queries" in normalized:
                    motion_state = normalized

        if not motion_state:
            raise RuntimeError(
                f"No motion-encoder weights found in {ckpt_path}. "
                "Expected one of: motion_encoder.*, student.*, or encoder.*"
            )

        m_miss, m_unexp = motion_module.load_state_dict(motion_state, strict=True)
        logger.info("Loaded motion encoder from %s (missing=%d unexpected=%d)", ckpt_path, len(m_miss), len(m_unexp))

        if residual_module is not None:
            if isinstance(residual_state_direct, dict) and residual_state_direct:
                residual_state = residual_state_direct
            else:
                residual_state = {}
                for pref in ("residual_encoder.", "residual_student.", "model.residual_encoder.", "module.residual_encoder."):
                    residual_state = {k[len(pref):]: v for k, v in normalized.items() if k.startswith(pref)}
                    if residual_state:
                        break

            if not residual_state:
                raise RuntimeError(
                    f"No residual-encoder weights found in {ckpt_path}. "
                    "This run requires the residual tower from phase-1 as well."
                )

            r_miss, r_unexp = residual_module.load_state_dict(residual_state, strict=True)
            logger.info("Loaded residual encoder from %s (missing=%d unexpected=%d)", ckpt_path, len(r_miss), len(r_unexp))

        # Apply tuning policy after loading.
        do_tune_motion = self.train_motion_encoder and bool(getattr(self.model, "tune_motion_encoder", True))
        if do_tune_motion:
            for p in motion_module.parameters():
                p.requires_grad = True
            motion_module.train()
            if residual_module is not None:
                for p in residual_module.parameters():
                    p.requires_grad = True
                residual_module.train()
        else:
            for p in motion_module.parameters():
                p.requires_grad = False
            motion_module.eval()
            if residual_module is not None:
                for p in residual_module.parameters():
                    p.requires_grad = False
                residual_module.eval()

    @property
    def total_steps(self) -> int:
        return int(getattr(self.trainer, "estimated_stepping_batches", 0))

    def configure_optimizers(self) -> OptimizerLRScheduler:
        scratch_decay, scratch_no_decay = [], []
        motion_decay, motion_no_decay = [], []
        lora_params = []
        lr_motion = self.lr_scratch * self.lr_motion_scale

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

            if name.startswith("motion_encoder."):
                if is_no_decay:
                    motion_no_decay.append(p)
                else:
                    motion_decay.append(p)
            else:
                if is_no_decay:
                    scratch_no_decay.append(p)
                else:
                    scratch_decay.append(p)

        param_groups = []
        if scratch_decay:
            param_groups.append({"params": scratch_decay, "lr": self.lr_scratch, "weight_decay": self.weight_decay})
        if scratch_no_decay:
            param_groups.append({"params": scratch_no_decay, "lr": self.lr_scratch, "weight_decay": 0.0})
        if motion_decay:
            param_groups.append({"params": motion_decay, "lr": lr_motion, "weight_decay": self.weight_decay})
        if motion_no_decay:
            param_groups.append({"params": motion_no_decay, "lr": lr_motion, "weight_decay": 0.0})
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
        self._probe_accumulate("train", output)
        loss = self.loss(target=batch, output=output)

        # Diversity penalty removed: with 64 action tokens the CE gradient per
        # token is 8× weaker than CLS. The 0.05 diversity loss dominated and
        # produced orthogonal-to-everything noise tokens. Let CE alone handle
        # differentiation — it will, given enough epochs and signal.

        bsz = batch["input_labels"].size(0)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True, batch_size=bsz, sync_dist=True)

        # Surface a few AGDTR + Action diagnostics on the tqdm bar so a glance
        # tells you whether routing is collapsing, gates are dead, or the action
        # stream is producing garbage. The rest go to TensorBoard / CSV only.
        bar_keys = {
            "phase2/gate_mean",
            "phase2/budget_std",
            "phase2/patch_diversity",
            "phase2/attn_patch_mass",
        }
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

        if self.gpu_cooldown_train_step_sec > 0.0:
            time.sleep(self.gpu_cooldown_train_step_sec)
        return loss

    def validation_step(self, batch, batch_idx):
        if hasattr(self.model, "force_probe"):
            self.model.force_probe = (int(batch_idx) == 0)
        output = self._run_model(batch)
        self._probe_accumulate("val", output)
        val_loss = self.loss(target=batch, output=output)
        bsz = batch["input_labels"].size(0)
        self.log("val_loss", val_loss, on_step=False, on_epoch=True, prog_bar=True, logger=True, batch_size=bsz, sync_dist=True)

        video = batch.get("video", batch)
        motion_vectors = video.get("motion_vectors", video.get("motion_vector", None))
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
        eos_hits = (gen_ids == eos_id).any(dim=1).to(dtype=torch.bfloat16).mean()
        gen_lengths = (gen_ids != eos_id).sum(dim=1).to(dtype=torch.bfloat16).mean()
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

        if self.gpu_cooldown_val_step_sec > 0.0:
            time.sleep(self.gpu_cooldown_val_step_sec)

    def on_validation_epoch_start(self) -> None:
        self.batch_res = {
            "version": "VERSION 1.0",
            "results": defaultdict(list),
            "external_data": {"used": "true", "details": "focalcap_phase2"},
        }
        self._probe_val_sum = defaultdict(float)
        self._probe_val_cnt = defaultdict(int)

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
                # Silence verbose PTBTokenizer prints from pycocoevalcap.
                with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
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

                self._print_probe_report("val")

        if dist.is_initialized():
            dist.barrier()

    def on_train_epoch_end(self) -> None:
        if dist.is_initialized() and dist.get_rank() != 0:
            return
        self._print_probe_report("train")
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

    def on_train_epoch_start(self) -> None:
        self._probe_train_sum = defaultdict(float)
        self._probe_train_cnt = defaultdict(int)


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
