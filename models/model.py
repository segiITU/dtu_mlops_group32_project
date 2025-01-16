import pytorch_lightning as pl
import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer
from typing import Optional, Dict, List

class PubMedSummarizer(pl.LightningModule):
    def __init__(
        self,
        model_name: str = "google/flan-t5-base",
        learning_rate: float = 2e-5,
        batch_size: int = 4,
        max_source_length: int = 1024,
        max_target_length: int = 256,
    ) -> None:
        """
        Initialize PubMed summarization model based on FLAN-T5.
        
        Parameters
        ----------
        model_name : str
            Name of the pretrained model to use
        learning_rate : float
            Learning rate for training
        batch_size : int
            Batch size for training
        max_source_length : int
            Maximum length for source texts (articles)
        max_target_length : int
            Maximum length for target texts (summaries)
        """
        super().__init__()
        self.save_hyperparameters()
        
        # Validate inputs
        if learning_rate <= 0:
            raise ValueError("Learning rate must be positive")
        if batch_size <= 0:
            raise ValueError("Batch size must be positive")
            
        # Load model and tokenizer
        self.model = T5ForConditionalGeneration.from_pretrained(model_name)
        self.tokenizer = T5Tokenizer.from_pretrained(model_name)
        
        # Save parameters
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.max_source_length = max_source_length
        self.max_target_length = max_target_length

    def forward(self, input_ids, attention_mask, labels=None):
        """Forward pass of the model"""
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )
        return outputs
    
    def _step(self, batch: Dict[str, List[str]]) -> torch.Tensor:
        """Common step for training, validation and testing"""
        # Add summarization prefix
        source_text = ["summarize: " + text for text in batch["article"]]
        
        # Tokenize inputs
        inputs = self.tokenizer(
            source_text,
            max_length=self.max_source_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        ).to(self.device)
        
        # Tokenize summaries
        labels = self.tokenizer(
            batch["abstract"],
            max_length=self.max_target_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        ).input_ids.to(self.device)
        
        # Forward pass
        outputs = self(
            input_ids=inputs.input_ids,
            attention_mask=inputs.attention_mask,
            labels=labels
        )
        
        return outputs.loss
    
    def training_step(self, batch: Dict[str, List[str]], batch_idx: int) -> torch.Tensor:
        """Training step"""
        loss = self._step(batch)
        self.log("train_loss", loss, batch_size=self.batch_size, sync_dist=True)
        return loss

    def validation_step(self, batch: Dict[str, List[str]], batch_idx: int) -> torch.Tensor:
        """Validation step"""
        loss = self._step(batch)
        self.log("val_loss", loss, batch_size=self.batch_size, sync_dist=True)
        return loss
    
    def configure_optimizers(self) -> torch.optim.Optimizer:
        """Configure optimizer"""
        return torch.optim.AdamW(self.parameters(), lr=self.learning_rate)