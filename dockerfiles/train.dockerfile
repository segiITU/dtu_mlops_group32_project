# Use newer PyTorch container for better compatibility
FROM nvcr.io/nvidia/pytorch:23.12-py3

# Install system dependencies
RUN apt-get update && \
    apt-get install --no-install-recommends -y \
    build-essential \
    git \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt \
    && pip install --no-cache-dir \
    transformers \
    datasets \
    pytorch-lightning

COPY models/ models/
COPY data/ data/

RUN mkdir -p models/checkpoints

ENV PYTHONPATH=/app

ENTRYPOINT ["python", "-u", "models/train_model.py"]

CMD ["--train_data_path", "data/train", \
     "--val_data_path", "data/validation", \
     "--batch_size", "8", \
     "--max_epochs", "3", \
     "--checkpoint_dir", "models/checkpoints"]