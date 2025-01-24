import gradio as gr
import requests
import os

BACKEND_URL = os.getenv("BACKEND_URL", "https://bart-summarizer-backend-962941447685.europe-west1.run.app")

def summarize_file(file):
    """Send the file to the backend for summarization."""
    with open(file.name, "r") as f:
        text = f.read()
    response = requests.post(f"{BACKEND_URL}/summarize", json={"text": text})
    if response.status_code == 200:
        return response.json().get("summary", "No summary returned.")
    return "Failed to summarize the text."

iface = gr.Interface(
    fn=summarize_file,
    inputs=gr.File(label="Upload a .txt file"),
    outputs=gr.Textbox(label="Summary"),
    title="Text Summarizer",
    description="Upload a .txt file to generate a summary."
)

if __name__ == "__main__":
    iface.launch(server_port=int(os.getenv("PORT", 8080)), server_name="0.0.0.0")