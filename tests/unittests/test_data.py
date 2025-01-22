from torch.utils.data import Dataset
# import nltk
# nltk.download('wordnet')
from nltk.util import ngrams
from nltk.translate.bleu_score import sentence_bleu

# from ....dtu_mlops_group32_project.src.dtu_mlops_group32_project.data import MyDataset
from src.dtu_mlops_group32_project.data import MyDataset
 


def test_my_dataset():
    """Test the MyDataset class correctly loads the dataset and its keys."""
    dataset = MyDataset("ccdv/pubmed-summarization")[0]

    assert list(dataset.keys()) == ["article", "summary"], "Dataset keys are incorrect"


def test_summary_length():
    """Test that summaries are shorter than articles."""

    dataset = MyDataset("ccdv/pubmed-summarization")[0]

    for data in dataset:
        article = data["article"]
        summary = data["summary"]
        assert len(summary) < len(article), "Summary is not shorter than the article"


def test_summary_content_overlap():
    """Test content overlap between articles and summaries."""

    dataset = MyDataset("ccdv/pubmed-summarization")[0]

    for data in dataset:
        article = data["article"]
        summary = data["summary"]
        
        # Tokenizing into words
        article_words = article.split()
        summary_words = summary.split()

        # Generating n-grams for content overlap calculation
        article_unigrams = set(ngrams(article_words, 1))
        summary_unigrams = set(ngrams(summary_words, 1))

        # Calculating overlap ratio
        overlap = len(article_unigrams & summary_unigrams) / len(summary_unigrams)
        
        assert overlap > 0.1, "Content overlap between article and summary is too low"
