from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict
import os
import torch
from transformers import BartTokenizer
from src.dtu_mlops_group32_project.model import BartSummarizer
from google.cloud import storage
from contextlib import asynccontextmanager


# Define request model
class SummarizationRequest(BaseModel):
    text: str


# Load model function
def load_model(model_path: str = "final_model.pt") -> BartSummarizer:
    """Load the BART model from the specified path."""
    model = BartSummarizer()
    model.load_state_dict(torch.load(model_path))
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    return model

# Download model from GCS
def download_model(bucket_name: str, source_blob_name: str, destination_file_name: str) -> None:
    """Download a file from Google Cloud Storage."""
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(source_blob_name)
    blob.download_to_filename(destination_file_name)
    print(f"Downloaded {source_blob_name} to {destination_file_name}.")

# Generate summary function
def generate_summary(text: str, model: BartSummarizer) -> Dict[str, str]:
    """
    Generate summary from input text using the provided BART model.
    
    Args:
        text: Input text to summarize
        model: Loaded BART model
        
    Returns:
        Dict containing original text and generated summary
    """
    device = next(model.parameters()).device  # Get the device of the model
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

@asynccontextmanager
async def lifespan(app: FastAPI):
    MODEL_PATH = "final_model.pt"
    if not os.path.exists(MODEL_PATH):
        print("Downloading model from GCS...")
        download_model(
            bucket_name="dtu_mlops_group32_project_bucket",
            source_blob_name="models/final_model.pt",
            destination_file_name=MODEL_PATH
        )

    global model
    model = load_model(MODEL_PATH)
    yield
    del model, MODEL_PATH

app = FastAPI(lifespan=lifespan)


# Define the API endpoint
@app.post("/summarize")
async def summarize(request: SummarizationRequest):
    try:
        result = generate_summary(request.text, model)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Root endpoint
@app.get("/")
async def root():
    return {"message": "Summarization API is running!"}
