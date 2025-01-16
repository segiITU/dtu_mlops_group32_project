# -*- coding: utf-8 -*-
import logging
import os
from pathlib import Path
from typing import Optional
import click
from datasets import Dataset, load_dataset
from dotenv import find_dotenv, load_dotenv

@click.command()
@click.argument('cache_dir', type=click.Path(exists=True))
@click.argument('k', type=int)
def main(cache_dir: str, k: Optional[int] = None) -> None:
    """
    Runs data processing scripts to turn raw data from (../raw) into
    cleaned data ready to be analyzed (saved in ../processed).
    
    Parameters
    ----------
    cache_dir : str, required
        A path to where the data is located.
    k : integer, optional
        The amount of datapoints to include from the dataset.
        
    Raises
    ------
    TypeError
        If the cache_dir isn't a string.
    TypeError
        If k isn't an integer.
    ValueError
        If k is a negative integer
    """
    if type(cache_dir) is not str:
        raise TypeError("cache_dir must be a string denoting the path to the data location.")
    if k is not None and type(k) is not int:
        raise TypeError("k must denote the amount (in an integer) of datapoints to include.")
    if k is not None and k <= 0:
        raise ValueError("k must be a positive amount of datapoints.")

    logger = logging.getLogger(__name__)
    logger.info("making final data set from raw data")

    dataset = load_dataset('ccdv/pubmed-summarization', cache_dir=cache_dir)
    
    if k is None:
        traindata: Dataset = Dataset.from_dict(dataset["train"])
        valdata: Dataset = Dataset.from_dict(dataset["validation"])
        testdata: Dataset = Dataset.from_dict(dataset["test"])
    else:
        traindata: Dataset = Dataset.from_dict(dataset["train"][:k])
        valdata: Dataset = Dataset.from_dict(dataset["validation"][:k])
        testdata: Dataset = Dataset.from_dict(dataset["test"][:k])

    # Save to data/raw directory
    raw_dir = os.path.join(cache_dir, "raw")
    os.makedirs(raw_dir, exist_ok=True)
    
    traindata.save_to_disk(os.path.join(raw_dir, "train"))
    valdata.save_to_disk(os.path.join(raw_dir, "validation"))
    testdata.save_to_disk(os.path.join(raw_dir, "test"))

if __name__ == "__main__":
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    
    project_dir = Path(__file__).resolve().parents[2]
    load_dotenv(find_dotenv())
    main()