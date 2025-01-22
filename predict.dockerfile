FROM python:3.11-slim
WORKDIR /app
COPY requirements_predict.txt .
RUN pip install -r requirements_predict.txt
COPY . .
CMD ["python", "app.py"]