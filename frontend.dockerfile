FROM python:3.11-slim

WORKDIR /app

COPY requirements_frontend.txt .
RUN pip install -r requirements_frontend.txt

COPY app/ ./app/

ENV PORT=8080
CMD ["python", "app/frontend.py", "--server.port", "8080", "--server.address=0.0.0.0"]