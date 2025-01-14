#!/usr/bin/env python3
# Scene Text Recognition Model Hub
# Copyright 2022 Darwin Bautista
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import math
import os
# import time
from pathlib import Path

import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, open_dict

import torch

from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, # StochasticWeightAveraging # Removed SWA callback to avoid conflicts with OneCycleLR
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.strategies import DDPStrategy
from pytorch_lightning.utilities.model_summary import summarize

from strhub.data.module import SceneTextDataModule
from strhub.models.base import BaseSystem
from strhub.models.utils import get_pretrained_weights


# Copied from OneCycleLR
def _annealing_cos(start, end, pct):
    'Cosine anneal from `start` to `end` as pct goes from 0.0 to 1.0.'
    cos_out = math.cos(math.pi * pct) + 1
    return end + (start - end) / 2.0 * cos_out


def get_swa_lr_factor(warmup_pct, swa_epoch_start, div_factor=25, final_div_factor=1e4) -> float:
    """Get the SWA LR factor for the given `swa_epoch_start`. Assumes OneCycleLR Scheduler."""
    total_steps = 1000  # Can be anything. We use 1000 for convenience.
    start_step = int(total_steps * warmup_pct) - 1
    end_step = total_steps - 1
    step_num = int(total_steps * swa_epoch_start) - 1
    pct = (step_num - start_step) / (end_step - start_step)
    return _annealing_cos(1, 1 / (div_factor * final_div_factor), pct)


@hydra.main(config_path='configs', config_name='main', version_base='1.2')
def main(config: DictConfig):
    trainer_strategy = 'auto'
    with open_dict(config):
        # Resolve absolute path to data.root_dir
        config.data.root_dir = hydra.utils.to_absolute_path(config.data.root_dir)
        # Special handling for GPU-affected config
        gpu = config.trainer.get('accelerator') == 'gpu'
        devices = config.trainer.get('devices', 0)
        if gpu:
            # Use mixed-precision training
            config.trainer.precision = 'bf16-mixed' if torch.get_autocast_dtype('cuda') is torch.bfloat16 else '16-mixed'
            # Set float32 matrix multiplication precision
            torch.set_float32_matmul_precision('medium')
        if gpu and devices > 1:
            # Use DDP with optimizations
            trainer_strategy = DDPStrategy(find_unused_parameters=False, gradient_as_bucket_view=True)
            # Scale steps-based config
            config.trainer.val_check_interval //= devices
            if config.trainer.get('max_steps', -1) > 0:
                config.trainer.max_steps //= devices

    # Special handling for PARseq
    if config.model.get('perm_mirrored', False):
        assert config.model.perm_num % 2 == 0, 'perm_num should be even if perm_mirrored = True'

    # Instantiate the model
    model: BaseSystem = hydra.utils.instantiate(config.model)
    # If specified, use pretrained weights to initialize the model
    if config.pretrained is not None:
        m = model.model if config.model._target_.endswith('PARseq') else model
        m.load_state_dict(get_pretrained_weights(config.pretrained))
    print(summarize(model, max_depth=2))

    # Instantiate the data module
    datamodule: SceneTextDataModule = hydra.utils.instantiate(config.data)

    # Configure checkpointing
    checkpoint = ModelCheckpoint(
        monitor='val_accuracy',
        mode='max',
        save_top_k=3,
        save_last=True,
        filename='{epoch}-{step}-{val_accuracy:.4f}-{val_NED:.4f}',
    )
    
    # Removed SWA callback to avoid conflicts with OneCycleLR
    # swa_epoch_start = 0.75
    # swa_lr = config.model.lr * get_swa_lr_factor(config.model.warmup_pct, swa_epoch_start)
    # swa = StochasticWeightAveraging(swa_lr, swa_epoch_start)
    

    # Determine the output directory
    cwd = (
        HydraConfig.get().runtime.output_dir
        if config.ckpt_path is None
        else str(Path(config.ckpt_path).parents[1].absolute())
    )

    # Configure the Trainer
    trainer: Trainer = hydra.utils.instantiate(
        config.trainer,
        logger=TensorBoardLogger(cwd, '', '.'),
        strategy=trainer_strategy,
        enable_model_summary=False,
        # callbacks=[checkpoint, swa], # Removed SWA callback to avoid conflicts with OneCycleLR
        callbacks=[checkpoint],  
    )

    # Train the model
    trainer.fit(model, datamodule=datamodule, ckpt_path=config.ckpt_path)

    # Save the final checkpoint with a unique name
    ckpt_path = config.ckpt_path if config.ckpt_path is not None else cwd

    # Ensure ckpt_path is a directory, not a file
    if os.path.isfile(ckpt_path):
        ckpt_path = os.path.dirname(ckpt_path)

    # Create a unique filename for the final checkpoint
    # final_ckpt_path = os.path.join(ckpt_path, f'final_{int(time.time())}.ckpt')
    final_ckpt_path = os.path.join(ckpt_path, f'final.ckpt')

    # Save the final checkpoint
    trainer.save_checkpoint(final_ckpt_path)
    print(f"Final checkpoint saved to {final_ckpt_path}")

    # Extract the model name from config.model._target_
    model_name = config.model._target_.split('.')[-1].lower()  # e.g., "parseq" -> "parsec", "trba", "trbc"
    final_model_path = os.path.join(ckpt_path, f'{model_name}_model.pt')

    # Save the model state_dict along with hyperparameters
    torch.save({
        'state_dict': model.state_dict(),
        'hparams': model.hparams,  # Ensure hyperparameters are saved
    }, final_model_path)
    print(f"Model saved to {final_model_path}")


if __name__ == '__main__':
    main()