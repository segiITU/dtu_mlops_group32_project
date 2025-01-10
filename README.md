Exam project for group 32 as part of MLOps course Winter 2025.

#### Overall goal of the project
To fine-tune an existing model to accurately summarize biomedical research papers. 

#### What framework are you going to use, and you do you intend to include the framework into your project?
We will implement the transformers framework from Hugging Face as it includes both the pre-trained models and the datasets. 

#### What data are you going to run on (initially, may change)
We have considered using one of the following datasets:
  - [PubMed summarization dataset](https://huggingface.co/datasets/ccdv/pubmed-summarization/viewer) consisting of biomedical articles and their respective abstracts.
  - [MLSR 2022 dataset](https://huggingface.co/datasets/allenai/mslr2022) consisting of abstract reviews.

#### What models do you expect to use
[BART](https://huggingface.co/docs/transformers/model_doc/bart) or [T5](https://huggingface.co/docs/transformers/model_doc/t5). 




## Project structure

The directory structure of the project looks like this:
```txt
├── .github/                  # Github actions and dependabot
│   ├── dependabot.yaml
│   └── workflows/
│       └── tests.yaml
├── configs/                  # Configuration files
├── data/                     # Data directory
│   ├── processed
│   └── raw
├── dockerfiles/              # Dockerfiles
│   ├── api.Dockerfile
│   └── train.Dockerfile
├── docs/                     # Documentation
│   ├── mkdocs.yml
│   └── source/
│       └── index.md
├── models/                   # Trained models
├── notebooks/                # Jupyter notebooks
├── reports/                  # Reports
│   └── figures/
├── src/                      # Source code
│   ├── project_name/
│   │   ├── __init__.py
│   │   ├── api.py
│   │   ├── data.py
│   │   ├── evaluate.py
│   │   ├── models.py
│   │   ├── train.py
│   │   └── visualize.py
└── tests/                    # Tests
│   ├── __init__.py
│   ├── test_api.py
│   ├── test_data.py
│   └── test_model.py
├── .gitignore
├── .pre-commit-config.yaml
├── LICENSE
├── pyproject.toml            # Python project file
├── README.md                 # Project README
├── requirements.txt          # Project requirements
├── requirements_dev.txt      # Development requirements
└── tasks.py                  # Project tasks
```


Created using [mlops_template](https://github.com/SkafteNicki/mlops_template),
a [cookiecutter template](https://github.com/cookiecutter/cookiecutter) for getting
started with Machine Learning Operations (MLOps).

