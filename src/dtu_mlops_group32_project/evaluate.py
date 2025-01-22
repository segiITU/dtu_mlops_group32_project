import pytorch_lightning as pl
import torch
from datasets import Dataset
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.utils.data import DataLoader

from model import BartSummarizer
from dtu_mlops_group32_project import _PROJECT_ROOT, _PATH_DATA


if __name__ == "__main__":

    model = BartSummarizer.load_from_checkpoint(
    checkpoint_path=_PROJECT_ROOT + "/models/checkpoints/last.ckpt",
    map_location="cpu"
)   
    testset = Dataset.load_from_disk(_PATH_DATA + "/processed/test")
    testloader = DataLoader(testset, num_workers=8)

    epochs = 1

    checkpoint_callback = ModelCheckpoint(dirpath=_PROJECT_ROOT + "/models")

    if torch.cuda.is_available():
        trainer = pl.Trainer(
            max_epochs=epochs,
            default_root_dir="",
            callbacks=[checkpoint_callback],
            accelerator="gpu",
            devices=[6],
            strategy="ddp",
        )
    else:
        trainer = pl.Trainer(
            max_epochs=epochs, default_root_dir="", callbacks=[checkpoint_callback]
        )

    results = trainer.test(model=model, dataloaders=testloader, verbose=True)

    print(results)