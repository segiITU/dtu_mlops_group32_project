# Use a CPU-only PyTorch container
FROM python:3.10-slim

# Install system dependencies
RUN apt-get update && \
    apt-get install --no-install-recommends -y \
    build-essential \
    git \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy the requirements file
COPY requirements.txt .

# Install dependencies from requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Install additional libraries if necessary (e.g., transformers)
RUN pip install --no-cache-dir \
    transformers==4.24.0 \
    datasets \
    pytorch-lightning

# Copy the application code (models and data directories)
COPY models/ models/
COPY data/ data/

# Create the checkpoint directory
RUN mkdir -p models/checkpoints

# Set the PYTHONPATH environment variable
ENV PYTHONPATH=/app

# Set entrypoint and default arguments for the training script
ENTRYPOINT ["python", "-u", "models/train_model.py"]
CMD ["--train_data_path", "data/train", \
     "--val_data_path", "data/validation", \
     "--batch_size", "8", \
     "--max_epochs", "3", \
     "--checkpoint_dir", "models/checkpoints"]
