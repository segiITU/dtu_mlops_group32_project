from contextlib import asynccontextmanager
import torch
from fastapi import FastAPI, File, UploadFile
import pytorch_lightning as pl

# Import your BartSummarizer class (the class definition you provided)
from model import BartSummarizer

app = FastAPI()

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load and clean up model on startup and shutdown."""
    global summarizer, device, gen_kwargs
    print("Loading custom BART summarization model from final_model.pt")

    # Initialize BartSummarizer instance and load the saved state_dict from final_model.pt
    summarizer = BartSummarizer(model_name="facebook/bart-base")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load the saved model state
    model_path = "path/to/final_model.pt"
    checkpoint = torch.load(model_path, map_location=device)  # Load the checkpoint to correct device

    # Load the state_dict into the model
    summarizer.load_state_dict(checkpoint['state_dict'])  # Make sure your saved model has 'state_dict' key
    summarizer.to(device)
    summarizer.eval()  # Set the model to evaluation mode

    # Define generation arguments for beam search summarization
    gen_kwargs = {"max_length": 256, "num_beams": 4, "early_stopping": True}

    yield

    print("Cleaning up")
    del summarizer, device, gen_kwargs

app = FastAPI(lifespan=lifespan)

@app.post("/summarize/")
async def summarize(file: UploadFile = File(...)):
    """Generate a summary for a text file."""
    # Read content from the uploaded text file
    paper_text = await file.read()
    paper_text = paper_text.decode("utf-8")

    # Tokenize the input text using the summarizer's tokenizer
    inputs = summarizer.tokenizer(paper_text, return_tensors="pt", truncation=True, max_length=summarizer.max_source_length).input_ids.to(device)

    # Generate summary using the model's generate function
    summary_ids = summarizer.model.generate(inputs, **gen_kwargs)
    summary = summarizer.tokenizer.decode(summary_ids[0], skip_special_tokens=True)

    return {"summary": summary}