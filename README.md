Exam project for group 32 as part of MLOps course Winter 2025.

#### Overall goal of the project
To fine-tune an existing model to accurately summarize biomedical research papers. 

#### What framework are you going to use, and you do you intend to include the framework into your project?
We will implement the Transformers framework from 🤗 Hugging Face as it includes both the pre-trained T5 model and the dataset for fine-tuning needed for this project. 

#### What data are you going to run on (initially, may change)
We have considered used the [PubMed summarization dataset](https://huggingface.co/datasets/ccdv/pubmed-summarization/viewer) consisting of biomedical articles and their respective abstracts.

#### What models do you expect to use
[BART](https://huggingface.co/docs/transformers/model_doc/bart) is a encoder-decoder transformer model that has achieved particularly good results in the abstractive summarization task, which is why we have chosen to fine-tune this model. 





