from contextlib import asynccontextmanager
import torch
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import RedirectResponse
import pytorch_lightning as pl

from .model import BartSummarizer

app = FastAPI()

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load and clean up model on startup and shutdown."""
    global summarizer, device, gen_kwargs
    print("Loading custom BART summarization model from final_model.pt")

    summarizer = BartSummarizer(model_name="facebook/bart-base")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_path = "C:/Users/sebas/dtu_mlops_group32_project/models/checkpoints/final_model.pt"
    checkpoint = torch.load(model_path, map_location=device)  # Load the checkpoint to correct device

    summarizer.load_state_dict(checkpoint)  # Load the state_dict into the model
    summarizer.to(device)  # Move the model to the appropriate device
    summarizer.eval()  # Set the model to evaluation mode

    gen_kwargs = {"max_length": 256, "num_beams": 4, "early_stopping": True}

    yield

    print("Cleaning up")
    del summarizer, device, gen_kwargs

app = FastAPI(lifespan=lifespan)

@app.get("/")
async def root():
    """Redirect to the API documentation."""
    return RedirectResponse(url="/docs")

@app.post("/summarize/")
async def summarize(file: UploadFile = File(...)):
    """Generate a summary for a text file."""
    paper_text = await file.read()
    paper_text = paper_text.decode("utf-8")

    inputs = summarizer.tokenizer(paper_text, return_tensors="pt", truncation=True, max_length=summarizer.max_source_length).input_ids.to(device)

    summary_ids = summarizer.model.generate(inputs, **gen_kwargs)
    summary = summarizer.tokenizer.decode(summary_ids[0], skip_special_tokens=True)

    return {"summary": summary}