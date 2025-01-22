FROM python:3.11-slim
WORKDIR /app

COPY requirements_predict.txt .
RUN pip install -r requirements_predict.txt

RUN apt-get update && apt-get install -y curl
RUN curl -sSL https://sdk.cloud.google.com | bash
ENV PATH $PATH:/root/google-cloud-sdk/bin

COPY . .

CMD ["python", "app.py"]