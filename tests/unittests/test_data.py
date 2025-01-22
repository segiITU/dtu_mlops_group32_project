from torch.utils.data import Dataset
# import nltk
# nltk.download('wordnet')
from nltk.util import ngrams
from nltk.translate.bleu_score import sentence_bleu

# from ....dtu_mlops_group32_project.src.dtu_mlops_group32_project.data import MyDataset
from src.dtu_mlops_group32_project.data import MyDataset

def test_my_dataset():
    """Test the MyDataset class correctly loads the dataset and its keys."""
    dataset = MyDataset("ccdv/pubmed-summarization")

    assert list(dataset[0].keys()) == ["train", "validation", "test"], "Dataset keys are incorrect"
    assert list(dataset[0]["train"].keys()) == ["article", "abstract"], "Dataset keys are incorrect"
    assert list(dataset[0]["validation"].keys()) == ["article", "abstract"], "Dataset keys are incorrect"
    assert list(dataset[0]["test"].keys()) == ["article", "abstract"], "Dataset keys are incorrect"


def test_abstract_length():
    """Test that summaries are shorter than articles."""

    dataset = MyDataset("ccdv/pubmed-summarization")

    train_article = dataset[1]["train"]["article"]
    train_abstract = dataset[1]["train"]["abstract"]
    validation_article = dataset[1]["validation"]["article"]
    validation_abstract = dataset[1]["validation"]["abstract"]
    test_article = dataset[1]["test"]["article"]
    test_abstract = dataset[1]["test"]["abstract"]
    
    assert len(train_abstract) < len(train_article), "abstract in training data is not shorter than the article"
    assert len(validation_abstract) < len(validation_article), "abstract in validation data is not shorter than the article"
    assert len(test_abstract) < len(test_article), "abstract in test data is not shorter than the article"


def test_abstract_content_overlap():
    """Test content overlap between articles and summaries."""

    dataset = MyDataset("ccdv/pubmed-summarization")

    article = dataset[2]["train"]["article"]
    abstract = dataset[2]["train"]["abstract"]
    
    # Tokenizing into words
    article_words = article.split()
    abstract_words = abstract.split()

    # Generating n-grams for content overlap calculation
    article_unigrams = set(ngrams(article_words, 1))
    abstract_unigrams = set(ngrams(abstract_words, 1))

    # Calculating overlap ratio
    overlap = len(article_unigrams & abstract_unigrams) / len(abstract_unigrams)
    
    assert overlap > 0.1, "Content overlap between article and abstract is too low"