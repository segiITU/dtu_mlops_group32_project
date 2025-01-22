FROM python:3.11-slim
WORKDIR /app

# Install dependencies
COPY requirements_predict.txt .
RUN pip install -r requirements_predict.txt

# Copy application code
COPY . .

# Set the port environment variable
ENV PORT=8080

# Set the entrypoint
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8080"]