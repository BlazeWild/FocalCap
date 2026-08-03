# -*- coding: utf-8 -*-

import logging
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pytorch_lightning as pl
import torch
from hydra_zen import builds, store, zen
from omegaconf import MISSING
from torch.utils.data import DataLoader

from cocap.modeling.lm_cocap import cocap_lm_cfg

logger = logging.getLogger(__name__)
torch.set_float32_matmul_precision("high")


def train(
    model: pl.LightningModule,
    train_dataloader: DataLoader,
    val_dataloader: DataLoader,
    trainer: pl.Trainer,
    ckpt_path: str = None,
):
    if ckpt_path is None:
        ckpt_path = os.environ.get("CKPT_PATH")

    try:
        from torchinfo import summary
        model_summary = summary(model, depth=4, verbose=0, col_names=("num_params", "trainable"))
        print("\n" + "=" * 80)
        print("MODEL SUMMARY")
        print("=" * 80)
        print(model_summary)
        print("=" * 80 + "\n")
    except Exception as exc:
        logger.warning("Could not print model summary: %s", exc)

    trainer.fit(
        model=model,
        train_dataloaders=train_dataloader,
        val_dataloaders=val_dataloader,
        ckpt_path=ckpt_path,
    )


if __name__ == "__main__":
    store(
        train,
        model=cocap_lm_cfg,
        train_dataloader=builds(DataLoader, dataset=MISSING, populate_full_signature=True),
        val_dataloader=builds(DataLoader, dataset=MISSING, populate_full_signature=True),
        trainer=builds(pl.Trainer, gradient_clip_val=1.0, populate_full_signature=True),
        populate_full_signature=True,
        name="train",
    )
    store.add_to_hydra_store()

    zen(train).hydra_main(
        config_path=(Path(__file__).parent.parent / "configs").as_posix(),
        config_name="exp/train/vatex_captioning",
        version_base=None,
    )
