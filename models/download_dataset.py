from datasets import load_dataset

# Download and save locally
dataset = load_dataset('ccdv/pubmed-summarization', cache_dir='./my_cache')

dataset.save_to_disk('C:/Users/sebas/dtu_mlops_group32_project/data')
