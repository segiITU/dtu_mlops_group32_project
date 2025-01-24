import os
import requests
import gradio as gr

from google.cloud import run_v2


### DEFINE PROJECT AND REGION NAME BEFORE DEPLOYING
@st.cache_resource  
def get_backend_url():
    """Get the URL of the backend service."""
    parent = "projects/dtu_mlops_group32/locations/<region>"
    client = run_v2.ServicesClient()
    services = client.list_services(parent=parent)
    for service in services:
        if service.name.split("/")[-1] == "production-model":
            return service.uri
    name = os.environ.get("BACKEND", None)
    return name


def summarize_text(text, backend):
    """Send the text to the backend for summarization."""
    summarize_url = f"{backend}/summarize"
    payload = {"text": text}
    response = requests.post(summarize_url, json=payload, timeout=10)
    if response.status_code == 200:
        return response.json()
    return None

def summarize_file(uploaded_file):
    """Main summarization function for the Gradio interface."""
    backend = get_backend_url()
    if backend is None:
        return "Backend service not found"

    # Read the uploaded text file
    text = uploaded_file.read().decode("utf-8")

    # Summarize the text
    result = summarize_text(text, backend=backend)
    if result is not None:
        summary = result.get("summary", "No summary returned.")
        return summary
    else:
        return "Failed to summarize the text"

# Define the Gradio interface
def create_gradio_interface():
    """Set up and launch the Gradio interface."""
    title = "Text Summarizer"
    
    with gr.Blocks() as demo:
        gr.Markdown(f"# {title}")

        with gr.Row():
            # File input and summary output
            text_file_input = gr.File(label="Upload a .txt file", type="file")
            summary_output = gr.Textbox(label="Summary", lines=10)

        def gradio_summarize(uploaded_file):
            summary = summarize_file(uploaded_file)
            return summary

        text_file_input.change(gradio_summarize, inputs=text_file_input, outputs=summary_output)

    return demo

if __name__ == "__main__":
    # Launch Gradio app
    app = create_gradio_interface()
    app.launch()
