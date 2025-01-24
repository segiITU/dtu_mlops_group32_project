from fastapi.testclient import TestClient
from dtu_mlops_group32_project.main import app
client = TestClient(app)

def test_read_root():
    response = client.get("/")
    print(response.text)  # Debug: Print the raw response content
    assert response.status_code == 200
