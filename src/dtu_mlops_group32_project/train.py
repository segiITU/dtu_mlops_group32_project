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

## TODO: Correct checkpoint dir

DEFAULT_TRAINING = _PATH_DATA + "/processed/train"
DEFAULT_VAL = _PATH_DATA + "/processed/validation"


def train(
    train_data_path: str,
    val_data_path: Optional[str] = None,
    batch_size: int = 4,
    max_epochs: int = 1,
    learning_rate: float = 2e-5,
    debug_mode: bool = False,
    checkpoint_dir: str = _PROJECT_ROOT + "models/checkpoints"
):
    """
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
    train_dataset = Dataset.load_from_disk(train_data_path)
    if val_data_path:
        val_dataset = Dataset.load_from_disk(val_data_path)
    else:
        datasets = train_dataset.train_test_split(test_size=0.1)
        train_dataset = datasets['train']
        val_dataset = datasets['test']
    
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
    
    # Initialize model
    model = BartSummarizer(
        learning_rate=learning_rate,
        batch_size=batch_size
    )
    
    # Setup callbacks
    checkpoint_callback = ModelCheckpoint(
        dirpath=checkpoint_dir,
        filename='bart-summarizer-{epoch:02d}-{val_loss:.2f}',
        save_top_k=3,
        monitor='val_loss',
        mode='min',
        save_last=True
    )
    
    # Set up trainer
    trainer = pl.Trainer(
        max_epochs=max_epochs,
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
    torch.save(model.state_dict(), f"{checkpoint_dir}/final_model.pt")
    print(f"Final model saved to {checkpoint_dir}/final_model.pt")
    print(f"Best model path: {checkpoint_callback.best_model_path}")
    print(f"Best validation loss: {checkpoint_callback.best_model_score:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train BART Summarizer')
    parser.add_argument(
        '--train_data_path',
        type=str,
        default=DEFAULT_TRAINING,
        help='Path to training data'
    )
    parser.add_argument(
        '--val_data_path',
        type=str,
        default=DEFAULT_VAL,
        help='Path to validation data (optional)'
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=4,
        help='Batch size for training'
    )
    parser.add_argument(
        '--max_epochs',
        type=int,
        default=1,
        help='Number of epochs to train'
    )
    parser.add_argument(
        '--learning_rate',
        type=float,
        default=2e-5,
        help='Learning rate for training'
    )
    parser.add_argument(
        '--debug_mode',
        action='store_true',
        help='Run in debug mode with limited data'
    )
    parser.add_argument(
        '--checkpoint_dir',
        type=str,
        default=_PROJECT_ROOT + "/models/checkpoints",
        help='Directory to save model checkpoints'
    )
    
    args = parser.parse_args()
    
    train(
        train_data_path=args.train_data_path,
        val_data_path=args.val_data_path,
        batch_size=args.batch_size,
        max_epochs=args.max_epochs,
        learning_rate=args.learning_rate,
        debug_mode=args.debug_mode,
        checkpoint_dir=args.checkpoint_dir
    )
