# Use newer PyTorch container for better compatibility
FROM nvcr.io/nvidia/pytorch:23.12-py3

# Install system dependencies
RUN apt-get update && \
    apt-get install --no-install-recommends -y \
    build-essential \
    git \
    curl \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy only requirements first to leverage Docker cache
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt \
    && pip install --no-cache-dir \
    transformers \
    datasets \
    pytorch-lightning

# Copy the model and training code
COPY models/model.py models/
COPY models/train_model.py models/

# Copy the data
COPY data/ data/

# Create directory for checkpoints
RUN mkdir -p models/checkpoints

# Set environment variables
ENV PYTHONPATH=/app

# Default command (can be overridden)
ENTRYPOINT ["python", "-u", "models/train_model.py"]
# Default arguments (can be overridden)
CMD ["--batch_size", "8", "--max_epochs", "3"]