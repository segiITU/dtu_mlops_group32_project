FROM python:3.11-slim

WORKDIR /app

COPY requirements_predict.txt .
RUN pip install -r requirements_predict.txt

COPY . .

ENV PORT=8080

CMD ["uvicorn", "src.dtu_mlops_group32_project.main:app", "--host", "0.0.0.0", "--port", "8080"]