import argparse
import os
import pytorch_lightning as pl
import torch
from datasets import Dataset
from torch.utils.data import DataLoader
from model import BartSummarizer
from dtu_mlops_group32_project import _PROJECT_ROOT

print(_PROJECT_ROOT)
def evaluate(
    checkpoint_dir: str = _PROJECT_ROOT + "/models/checkpoints",
    test_data_path: str = _PROJECT_ROOT + "/data/test",
    output_dir: str = _PROJECT_ROOT + "/models/evaluation",
    batch_size: int = 4,
    debug_mode: bool = False
):
    os.makedirs(output_dir, exist_ok=True)

    checkpoint_files = [f for f in os.listdir(checkpoint_dir) if f.endswith(".ckpt")]
    if not checkpoint_files:
        raise FileNotFoundError(f"No checkpoint files found in {checkpoint_dir}")
    checkpoint_path = os.path.join(checkpoint_dir, checkpoint_files[0])
    print(f"Using checkpoint: {checkpoint_path}")

    model = BartSummarizer.load_from_checkpoint(
    checkpoint_path=_PROJECT_ROOT + "/models/checkpoints/last.ckpt",
    map_location="cpu"
    )    
    model.test_step = model.validation_step
    
    test_dataset = Dataset.load_from_disk(test_data_path)
    if debug_mode:
        test_dataset = test_dataset.select(range(min(20, len(test_dataset))))
    print(f"Evaluating on {len(test_dataset)} examples")
    
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    trainer = pl.Trainer(
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        limit_test_batches=None if debug_mode else 1.0  
    )
    results = trainer.test(model=model, dataloaders=test_loader)

    results_path = os.path.join(output_dir, "evaluation_results.json")
    with open(results_path, "w") as f:
        import json
        json.dump(results, f, indent=4)
    print(f"Results saved to {results_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate BART Summarizer")
    parser.add_argument("--checkpoint_dir", type=str, default=_PROJECT_ROOT + "/models/checkpoints")
    parser.add_argument("--test_data_path", type=str, default=_PROJECT_ROOT + "/data/processed/test")
    parser.add_argument("--output_dir", type=str, default=_PROJECT_ROOT + "models/evaluation")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--debug_mode", action="store_true")
    
    args = parser.parse_args()
    evaluate(**vars(args))