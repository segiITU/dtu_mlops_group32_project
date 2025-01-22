from fastapi import FastAPI, File, UploadFile
from pydantic import BaseModel
import uvicorn
import joblib

app = FastAPI()

# Load your model (replace 'your_model.pkl' with your actual model file)
model = joblib.load('your_model.pkl')

class Prediction(BaseModel):
    prediction: str

@app.post("/predict/", response_model=Prediction)
async def predict(file: UploadFile = File(...)):
    contents = await file.read()
    text = contents.decode('utf-8')
    
    # Perform inference (replace this with your actual inference code)
    prediction = model.predict([text])[0]
    
    return Prediction(prediction=prediction)


