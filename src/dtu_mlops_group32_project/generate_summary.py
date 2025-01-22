import torch
from transformers import BartTokenizer
from model import BartSummarizer
from typing import Dict, Union

def generate_summary(text: str, model_path: str = "models/checkpoints/final_model.pt") -> Dict[str, str]:
    """
    Generate summary from input text using loaded BART model.
    
    Args:
        text: Input text to summarize
        model_path: Path to the saved model weights
        
    Returns:
        Dict containing original text and generated summary
    """
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

#result = generate_summary("{article: xxx}")
#print(result["summary"])
