# -*- coding: utf-8 -*-
"""Quantify the iframe_cls "leak" in the Phase-1 student summary head.

Runs validation twice on a trained checkpoint:
  Pass A: normal forward (iframe_cls flows into both summary heads).
  Pass B: iframe_cls forced to None (motion+residual must do all work alone).

Prints L_cos / L_recon / L_res_cos for both passes and the GAP =
L_cos_no_cls − L_cos_with_cls. Small gap (<0.05) means motion+residual
genuinely learned the delta direction; large gap (>0.15) means CLS context
was carrying most of the signal.

USAGE (from repo root, same env as training):
    python tools/eval_with_without_cls.py \\
        ckpt=/path/to/checkpoints/latest/latest-epoch00X-stepNNN.ckpt \\
        --config-name exp/train/vatex_captioning

It re-uses the same Hydra config tree as train_net.py, so the model and
val_dataloader match training exactly.
"""

import io
import logging
import os
import sys
from pathlib import Path
from typing import Any, Optional

if sys.stdout.encoding and sys.stdout.encoding.lower() != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
if sys.stderr.encoding and sys.stderr.encoding.lower() != 'utf-8':
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

import pytorch_lightning as pl
import torch
from hydra_zen import builds, store, zen
from omegaconf import MISSING
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

from cocap.modeling.lm_cocap import cocap_lm_cfg

logger = logging.getLogger(__name__)
torch.set_float32_matmul_precision('high')


def _accumulate_metrics(
    model,
    dataloader,
    device,
    max_batches: Optional[int] = None,
    pass_name: str = "eval",
    show_progress: bool = True,
    progress_every: int = 25,
) -> dict:
    """Run val pass; return per-sample-weighted means of the loss components."""
    model.eval()
    sums = {'L_cos': 0.0, 'L_recon': 0.0, 'L_res_cos': 0.0, 'L_infonce': 0.0, 'L_mse': 0.0}
    n_samples = 0
    total_batches = None
    try:
        total_batches = len(dataloader)
    except Exception:
        total_batches = None
    if (max_batches is not None) and (total_batches is not None):
        total_batches = min(total_batches, int(max_batches))

    iterator = enumerate(dataloader)
    if show_progress:
        iterator = tqdm(
            iterator,
            total=total_batches,
            desc=f"[CLS-LEAK] {pass_name}",
            dynamic_ncols=True,
            leave=True,
        )

    with torch.no_grad():
        for batch_idx, batch in iterator:
            if max_batches is not None and batch_idx >= max_batches:
                break

            video = batch.get('video', batch)

            def _to_dev(x):
                return x.to(device) if torch.is_tensor(x) else x

            kwargs = dict(
                clip_i_cls=_to_dev(video.get('clip_i_cls', None)),
                clip_i_spatial=_to_dev(video.get('clip_i_spatial', None)),
                clip_p_spatial=_to_dev(video.get('clip_p_spatial', None)),
                motion_vectors=_to_dev(video.get('motion_vectors', video.get('motion_vector', None))),
                input_ids=_to_dev(batch['input_ids']),
                attention_mask=_to_dev(batch['input_mask']),
                labels=_to_dev(batch['input_labels']),
                input_mask_mv=_to_dev(video.get('input_mask_mv', None)),
                input_mask_gop=_to_dev(video.get('input_mask_gop', None)),
                target_delta=None,
                iframe=_to_dev(video.get('iframe', None)),
                last_p_frame=_to_dev(video.get('last_p_frame', None)),
                residual=_to_dev(video.get('residual', None)),
                input_mask_res=_to_dev(video.get('input_mask_res', None)),
            )

            outputs = model.model(**kwargs)
            _ = model.loss(target=batch, output=outputs)

            comp = getattr(model.loss, 'latest_loss_components', {}) or {}
            bs = batch['input_labels'].size(0)
            sums['L_cos'] += float(comp.get('delta_cosine_loss', torch.tensor(0.0)).item()) * bs
            sums['L_recon'] += float(comp.get('reconstruct_loss', torch.tensor(0.0)).item()) * bs
            sums['L_res_cos'] += float(comp.get('res_cos_loss', torch.tensor(0.0)).item()) * bs
            sums['L_infonce'] += float(comp.get('infonce_loss', torch.tensor(0.0)).item()) * bs
            sums['L_mse'] += float(comp.get('delta_mse_loss', torch.tensor(0.0)).item()) * bs
            n_samples += bs
            if show_progress and n_samples > 0 and ((batch_idx + 1) % max(1, int(progress_every)) == 0):
                iterator.set_postfix({
                    'L_cos': f"{(sums['L_cos'] / n_samples):.4f}",
                    'L_recon': f"{(sums['L_recon'] / n_samples):.4f}",
                    'L_res_cos': f"{(sums['L_res_cos'] / n_samples):.4f}",
                })

    return {k: (v / max(1, n_samples)) for k, v in sums.items()}


def _patch_to_disable_iframe_cls(model):
    """Monkey-patch HavCoCap.forward so iframe_cls is set to None before reaching
    the motion encoder. Returns an undo function."""
    cocap = model.model
    original_forward = cocap.forward

    def patched_forward(*args, **kwargs):
        kwargs['clip_i_cls'] = None
        return original_forward(*args, **kwargs)

    cocap.forward = patched_forward
    return lambda: setattr(cocap, 'forward', original_forward)


def evaluate(
    model: pl.LightningModule,
    train_dataloader: DataLoader,  # accepted for config compatibility, unused
    val_dataloader: Any,
    trainer: pl.Trainer,           # accepted for config compatibility, unused
    ckpt: str = "",
    max_batches: int = 0,
    show_progress: bool = True,
    progress_every: int = 25,
):
    del train_dataloader, trainer  # unused — we run our own eval loop

    if not ckpt:
        raise SystemExit("Pass ckpt=/path/to/checkpoint.ckpt as a Hydra override.")
    if not os.path.isfile(ckpt):
        raise SystemExit(f"Checkpoint not found: {ckpt}")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print(f"[CLS-LEAK EVAL] Loading checkpoint: {ckpt}")
    state = torch.load(ckpt, map_location='cpu', weights_only=False)
    sd = state.get('state_dict', state)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    print(f"[CLS-LEAK EVAL] state_dict loaded (missing={len(missing)}, unexpected={len(unexpected)})")

    model = model.to(device)
    max_batches_arg = None if max_batches <= 0 else int(max_batches)

    print("\n[CLS-LEAK EVAL] Pass A: WITH iframe_cls (normal)")
    metrics_with = _accumulate_metrics(
        model,
        val_dataloader,
        device,
        max_batches_arg,
        pass_name="with_cls",
        show_progress=show_progress,
        progress_every=progress_every,
    )
    for k, v in metrics_with.items():
        print(f"  {k} = {v:.4f}")

    print("\n[CLS-LEAK EVAL] Pass B: WITHOUT iframe_cls (motion+residual only)")
    undo = _patch_to_disable_iframe_cls(model)
    try:
        metrics_without = _accumulate_metrics(
            model,
            val_dataloader,
            device,
            max_batches_arg,
            pass_name="without_cls",
            show_progress=show_progress,
            progress_every=progress_every,
        )
    finally:
        undo()
    for k, v in metrics_without.items():
        print(f"  {k} = {v:.4f}")

    gap_cos = metrics_without['L_cos'] - metrics_with['L_cos']
    gap_recon = metrics_without['L_recon'] - metrics_with['L_recon']
    gap_res_cos = metrics_without['L_res_cos'] - metrics_with['L_res_cos']

    print("\n[CLS-LEAK EVAL] GAPS  (no_cls − with_cls; positive = CLS was helping)")
    print(f"  L_cos    : +{gap_cos:.4f}    (interpretation below)")
    print(f"  L_recon  : +{gap_recon:.4f}    (should be ~0; recon path doesn't use iframe_cls)")
    print(f"  L_res_cos: +{gap_res_cos:.4f}    (residual summary head also uses iframe_cls)")

    print("\n[CLS-LEAK EVAL] Verdict on L_cos gap:")
    if gap_cos < 0.05:
        print("  GOOD: motion+residual carry ~95%+ of the direction signal.")
    elif gap_cos < 0.15:
        print("  ACCEPTABLE: CLS provided a meaningful but bounded boost.")
    else:
        print("  WARNING: CLS context was carrying a large fraction of the cos signal.")
    print("\n  L_recon is the honest 'did motion+residual learn P-frame replacement' metric.")
    print(f"  L_recon (with cls)    = {metrics_with['L_recon']:.4f}")
    print(f"  L_recon (without cls) = {metrics_without['L_recon']:.4f}")


if __name__ == '__main__':
    print("\n" + "!" * 60)
    print("CLS-LEAK ABLATION EVAL (with vs without iframe_cls)")
    print("!" * 60 + "\n")

    store(
        evaluate,
        model=cocap_lm_cfg,
        train_dataloader=builds(DataLoader, dataset=MISSING, populate_full_signature=True),
        val_dataloader=builds(DataLoader, dataset=MISSING, populate_full_signature=True),
        trainer=builds(pl.Trainer, gradient_clip_val=1.0, populate_full_signature=True),
        ckpt="",
        max_batches=0,
        show_progress=True,
        progress_every=25,
        populate_full_signature=True,
        name='evaluate',
    )
    # Compatibility alias: exp/train/vatex_captioning.yaml defaults include `/train`.
    # Registering the same eval schema under `train` lets this script reuse that
    # existing experiment config without edits.
    store(
        evaluate,
        model=cocap_lm_cfg,
        train_dataloader=builds(DataLoader, dataset=MISSING, populate_full_signature=True),
        val_dataloader=builds(DataLoader, dataset=MISSING, populate_full_signature=True),
        trainer=builds(pl.Trainer, gradient_clip_val=1.0, populate_full_signature=True),
        ckpt="",
        max_batches=0,
        show_progress=True,
        progress_every=25,
        populate_full_signature=True,
        name='train',
    )
    store.add_to_hydra_store()

    zen(evaluate).hydra_main(
        config_path=(Path(__file__).parent.parent / 'configs').as_posix(),
        config_name='exp/train/vatex_captioning',
        version_base=None,
    )
