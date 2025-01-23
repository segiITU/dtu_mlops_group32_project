## How to download the data

To download the processed PubMed data used for this project, run the src\dtu_mlops_group32_project\data.py file.

The data will be saved in a .arrow format in a processed/ folder with nested /train, /test and /validation folders, respectively. 

## About the data
The dataset consists of full-length articles and their corresponding abstracts. The dataset was created specifically for the task of improving abstractive summarization on longer documents.

The dataset has been created by Cohan et al. for the paper 'A Discourse-Aware Attention Model for Abstractive Summarization of Long Documents'. The article can be found [here](https://arxiv.org/abs/1804.05685) and the dataset can also be viewed [from here](https://huggingface.co/datasets/ccdv/pubmed-summarization). 