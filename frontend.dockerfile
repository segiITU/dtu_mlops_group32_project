FROM python:3.11-slim

WORKDIR /app

COPY requirements_frontend.txt .
RUN pip install -r requirements_frontend.txt

COPY app/frontend.py .

ENV PORT=8080
CMD ["gradio", "frontend.py", "--server.port", "$PORT", "--server.address=0.0.0.0"]