import torch
from transformers import BartTokenizer
from model import BartSummarizer
from typing import Dict, Union
from google.cloud import storage
import os

def download_model(bucket_name: str, source_blob_name: str, destination_file_name: str) -> None:
    """Downloads a file from Google Cloud Storage."""
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(source_blob_name)
    blob.download_to_filename(destination_file_name)
    print(f"Downloaded {source_blob_name} to {destination_file_name}.")

def generate_summary(text: str, model_path: str = "final_model.pt") -> Dict[str, str]:
    """
    Generate summary from input text using loaded BART model.
    
    Args:
        text: Input text to summarize
        model_path: Path to the saved model weights (local or GCS)
        
    Returns:
        Dict containing original text and generated summary
    """
    # Download the model from GCS if it doesn't exist locally
    if not os.path.exists(model_path):
        print("Downloading model from GCS...")
        download_model(
            bucket_name="dtu_mlops_group32_project_bucket",
            source_blob_name="models/final_model.pt",
            destination_file_name=model_path
        )
    
    # Load the model
    model = BartSummarizer()
    model.load_state_dict(torch.load(model_path))
    model.eval()
    
    # Move to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    inputs = model.tokenizer(
        text,
        max_length=model.max_source_length,
        padding="max_length",
        truncation=True,
        return_tensors="pt"
    ).to(device)
    
    with torch.no_grad():
        summary_ids = model.model.generate(
            inputs.input_ids,
            attention_mask=inputs.attention_mask,
            max_length=model.max_target_length,
            num_beams=4,
            early_stopping=True
        )
    
    summary = model.tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    
    return {
        "input_text": text,
        "summary": summary
    }