import argparse
from typing import Optional
import os
from pathlib import Path

import pytorch_lightning as pl
import torch
from datasets import Dataset
from torch.utils.data import DataLoader
from pytorch_lightning.callbacks import ModelCheckpoint

from model import BartSummarizer
from dtu_mlops_group32_project import _PATH_DATA, _PROJECT_ROOT
import wandb

DEFAULT_TRAINING = _PATH_DATA + "/processed/train"
DEFAULT_VAL = _PATH_DATA + "/processed/validation"
DEFAULT_CHECK = _PROJECT_ROOT + "/models/checkpoints"

def train(config: str, wandbkey: Optional[str] = None, debug_mode: bool = False):
    if not (wandbkey is None):
        wandb.login(key=wandbkey)  # input API key for wandb for docker
        project = "dtu_mlops_group32_project"
        entity = "arzuburcuguven"
        anonymous = None
        mode = "online"
    else:
        project = None
        entity = None
        anonymous = "must"
        mode = "disabled"

    wandb.init(
        project=project,
        entity=entity,
        anonymous=anonymous,
        config=config,
        mode=mode,
    )

    lr = wandb.config.lr
    epochs = wandb.config.epochs
    batch_size = wandb.config.batch_size
    seed = wandb.config.batch_size

    if seed is not None:
        torch.manual_seed(seed)

    model = BartSummarizer(learning_rate=lr, batch_size=batch_size)

    if not (wandbkey is None):
        wandb.watch(model, log_freq=100)
        logger = pl.loggers.WandbLogger(
            project="dtu_mlops_group32_project", entity="arzuburcuguven"
        )
    else:
        logger = True

    """
    #TODO correct these
    Train the BART summarization model
    
    Parameters
    ----------
    train_data_path : str
        Path to training data
    val_data_path : str, optional
        Path to validation data
    batch_size : int
        Batch size for training
    max_epochs : int
        Number of epochs to train
    learning_rate : float
        Learning rate for training
    debug_mode : bool
        If True, train on a small subset of data
    checkpoint_dir : str
        Directory to save model checkpoints
    """
    # Load datasets
    train_dataset = Dataset.load_from_disk(DEFAULT_TRAINING)
    val_dataset = Dataset.load_from_disk(DEFAULT_VAL)

    
    if debug_mode:
        train_dataset = train_dataset.select(range(min(100, len(train_dataset))))
        val_dataset = val_dataset.select(range(min(20, len(val_dataset))))
    
    print(f"Training on {len(train_dataset)} examples")
    print(f"Validating on {len(val_dataset)} examples")
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        persistent_workers=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        persistent_workers=True
    )
        
    # Setup callbacks
    checkpoint_callback = ModelCheckpoint(
        dirpath=DEFAULT_CHECK,
        filename='bart-summarizer-{epoch:02d}-{val_loss:.2f}',
        save_top_k=3,
        monitor='val_loss',
        mode='min',
        save_last=True
    )
    
    # Set up trainer
    trainer = pl.Trainer(
        max_epochs=epochs,
        accelerator='auto',
        devices=1,
        enable_progress_bar=True,
        callbacks=[checkpoint_callback],
        limit_train_batches=1.0 if not debug_mode else 0.1,
        limit_val_batches=None if debug_mode else 0.1,
        val_check_interval=0.25,
        log_every_n_steps=10
    )
    
    # Train
    trainer.fit(
        model=model, 
        train_dataloaders=train_loader,
        val_dataloaders=val_loader
    )
    
    # Save final model
    torch.save(model.state_dict(), f"{DEFAULT_CHECK}/final_model.pt")
    print(f"Final model saved to {DEFAULT_CHECK}/final_model.pt")
    print(f"Best model path: {checkpoint_callback.best_model_path}")
    print(f"Best validation loss: {checkpoint_callback.best_model_score:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default=_PROJECT_ROOT + "/src/dtu_mlops_group32_project/config/default_params.yaml",
        type=str,
        help="configuration file with hyperparameters",
    )
    parser.add_argument("--wandbkey", default=None, type=str, help="W&B API key")
    parser.add_argument(
        "--debug_mode", action="store_true", help="Run only 10 percent of data"
    )

    args = parser.parse_args()

    train(args.config, args.wandbkey, args.debug_mode)
