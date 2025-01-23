from src.dtu_mlops_group32_project.model import BartSummarizer

def test_model():
    """Test the BartSummarizer model."""
    model = BartSummarizer()
    print(model)

    # Test model forward pass
    input_ids = model.tokenizer("This is a test", return_tensors="pt")["input_ids"]
    attention_mask = model.tokenizer("This is a test", return_tensors="pt")["attention_mask"]
    outputs = model(input_ids, attention_mask)
    # Check loss and output structure
    if hasattr(outputs, "loss"):
        print(f"Loss from the forward pass: {outputs.loss}")
    else:
        raise AssertionError("Model forward pass does not return loss")

    # Test _step method
    batch = {
        "article": ["This is a test", "Another test"],
        "abstract": ["This is a abstract", "Another abstract"]
    }
    loss = model._step(batch)
    assert loss is not None, "Model _step method does not return loss"
    assert loss.item() >= 0, "Model _step method returns negative loss"

    # Test training step
    loss = model.training_step(batch, 0)
    assert loss is not None, "Model training_step method does not return loss"
    assert loss.item() >= 0, "Model training_step method returns negative loss"

    # Test validation step
    loss = model.validation_step(batch, 0)
    assert loss is not None, "Model validation_step method does not return loss"
    assert loss.item() >= 0, "Model validation_step method returns negative loss"

test_model()
