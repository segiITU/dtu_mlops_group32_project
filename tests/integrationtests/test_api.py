from fastapi.testclient import TestClient
from dtu_mlops_group32_project.main import app
client = TestClient(app)

def test_read_root():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "Summarization API is running!"}


