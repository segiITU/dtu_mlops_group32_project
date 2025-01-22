from fastapi import FastAPI
from pydantic import BaseModel
import joblib

app = FastAPI()

@app.get("/")
def read_root():
    return {"message": "Welcome to the DTU MLOps Group 32 Project API"}

@app.get("/items/{item_id}")
def read_item(item_id: int, q: str = None):
    return {"item_id": item_id, "q": q}

@app.post("/items/")
def create_item(item: dict):
    return {"item": item}

class InferenceRequest(BaseModel):
    data: list

class InferenceResponse(BaseModel):
    prediction: list

# Load your pre-trained model
model = joblib.load('/path/to/your/model.joblib')

@app.post("/predict/", response_model=InferenceResponse)
def predict(request: InferenceRequest):
    data = request.data
    prediction = model.predict(data).tolist()
    return InferenceResponse(prediction=prediction)