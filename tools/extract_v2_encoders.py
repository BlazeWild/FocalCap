#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
from pathlib import Path
from typing import Dict, Iterable, Tuple

import torch


def _extract_first_prefix(state: Dict[str, torch.Tensor], prefixes: Iterable[str]) -> Tuple[Dict[str, torch.Tensor], str]:
    for pref in prefixes:
        picked = {k[len(pref):]: v for k, v in state.items() if k.startswith(pref)}
        if picked:
            return picked, pref
    return {}, ""


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract v2 motion/residual encoder tensors from a phase-1 checkpoint.")
    parser.add_argument("--in-ckpt", required=True, help="Path to input .ckpt/.pt")
    parser.add_argument("--out-pt", required=True, help="Path to output encoder-only .pt")
    args = parser.parse_args()

    in_ckpt = Path(args.in_ckpt).expanduser().resolve()
    out_pt = Path(args.out_pt).expanduser().resolve()
    out_pt.parent.mkdir(parents=True, exist_ok=True)

    raw = torch.load(str(in_ckpt), map_location="cpu", weights_only=False)
    state = raw.get("state_dict", raw) if isinstance(raw, dict) else raw
    if not isinstance(state, dict):
        raise RuntimeError(f"Unsupported checkpoint format in {in_ckpt}")

    motion_state, motion_pref = _extract_first_prefix(
        state,
        (
            "encoder.",
            "model.encoder.",
            "module.encoder.",
            "motion_encoder.",
            "model.motion_encoder.motion_encoder.",
            "module.model.motion_encoder.motion_encoder.",
        ),
    )
    residual_state, residual_pref = _extract_first_prefix(
        state,
        (
            "residual_encoder.",
            "model.residual_encoder.",
            "module.residual_encoder.",
            "residual_student.",
            "model.motion_encoder.residual_encoder.",
            "module.model.motion_encoder.residual_encoder.",
        ),
    )

    if not motion_state:
        raise RuntimeError(
            "Could not find v2 motion encoder tensors. "
            "Expected keys starting with encoder.* / motion_encoder.*"
        )

    payload = {
        "format": "focalcap_v2_encoders",
        "source_checkpoint": str(in_ckpt),
        "motion_prefix": motion_pref,
        "residual_prefix": residual_pref,
        "motion_encoder": motion_state,
        "residual_encoder": residual_state,
    }
    torch.save(payload, str(out_pt))

    motion_params = sum(v.numel() for v in motion_state.values())
    residual_params = sum(v.numel() for v in residual_state.values())
    print(f"saved: {out_pt}")
    print(f"motion prefix:   {motion_pref or 'N/A'}")
    print(f"residual prefix: {residual_pref or 'N/A'}")
    print(f"motion params:   {motion_params:,}")
    print(f"residual params: {residual_params:,}")


if __name__ == "__main__":
    main()
