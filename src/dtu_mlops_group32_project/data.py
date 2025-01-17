from pathlib import Path
import typer
from typing import Optional
from datasets import Dataset, load_dataset, DatasetDict
import re
#TO DO:
#connect this to DVC

DEFAULT_CACHE_DIR = Path(__file__).parent.parent.parent / "data/processed"

class MyDataset:
    """
    Class optimized for huggingface datasets
    cleans up and saves the dataset in to ../processed

    Parameters
    ----------
    dataset_name : str, required
        Name of the dataset
    cache_dir : Path, Optional
        A path to where the data is located.

    k : integer, optional
        The amount of datapoints to include from the the dataset.
    
    """

    def __init__(self, dataset_name: str) -> None:
        self.dataset = load_dataset(dataset_name)

    def __len__(self) -> int:
        return len(self.dataset["train"])

    def __getitem__(self, index: int):
        keylist = list(self.dataset.keys())
        result = {}
        for key in keylist:
            result[key] = self.dataset[key][index]  # Fetch the index-th element for each key
        return result

    @staticmethod
    def cleaner(text):
        text = re.sub(r"\[\s*\d+(?:\s*, *\d+)*\s*\]|\[\s*\d+\s*-\s*\d+\s*\]", "", text) # Remove references e.g [ 3, 5 ]
        text = re.sub(r'\s+', ' ', text)  # Remove multiple spaces
        return text

    def preprocess(self, cache_dir: Path, k: Optional[int] = None) -> None:
        """Preprocess the raw data and save it to the output folder."""
        features = self.dataset["train"].features
        feature_list  = list(features.keys())
        self.dataset = self.dataset.map(lambda x: {key: self.cleaner(x[key]) if key in feature_list else x[key] for key in x.keys()})
        if k is not None:
        # Create new DatasetDict with sliced datasets
            sliced_dataset = DatasetDict({
                'train': self.dataset['train'].select(range(k)),
                'validation': self.dataset['validation'].select(range(min(k, len(self.dataset['validation'])))),
                'test': self.dataset['test']
            })
            self.dataset = sliced_dataset

        # Save processed dataset
        self.dataset.save_to_disk(cache_dir)


def preprocess(
    dataset_name: str = typer.Argument(),
    cache_dir: Path = typer.Argument(default=DEFAULT_CACHE_DIR),
    k: Optional[int] = typer.Option()
) -> None:
    print("Preprocessing data...")
    dataset = MyDataset(dataset_name)
    dataset = dataset.preprocess(cache_dir, k)


if __name__ == "__main__":
    typer.run(preprocess)
