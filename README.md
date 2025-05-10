# RAG Application with Mistral 7B

This is a simple RAG (Retrieval-Augmented Generation) application that uses Mistral 7B as the language model, FAISS as the vector database, and Mistral's API for text generation.

<img src="https://github.com/obiyann/simple_rag/blob/main/rag_app_workflow.png" title="RAG Application Workflow">

## Features

- Add text to the knowledge base by copying and pasting
- Upload text files to the knowledge base
- Ask questions and get answers based on the stored knowledge
- Modern and user-friendly web interface
- Instrumented with Datadog LLM Obs - Your will need a Datadog instance (sign for a free trial here: https://www.datadoghq.com/free-datadog-trial/)
- You will need a Mistral La Plateforme free account and an API Key (sign up here: https://auth.mistral.ai/ui/registration)

## Setup

1. Clone this repository
2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Create a `.env` file in the root directory and add your Mistral API key:
   ```
   MISTRAL_API_KEY=your_mistral_api_key_here
   DATADOG_API_KEY=your_datadog_api_key_here
   ```
   You can get an API key from [Mistral's platform](https://console.mistral.ai/).

## Running the Application

1. Start the FastAPI server:
   ```bash
   python app.py
   ```
2. Open your web browser and navigate to `http://localhost:8080`

## Usage

1. **Adding Text to Knowledge Base**:
   - Enter text in the text area and click "Add Text"
   - Or upload a text file using the file upload form
   - Use Curl to add text to the Knowledge Base:
      curl -X POST "http://localhost:8080/query" \
           -H "Content-Type: application/x-www-form-urlencoded" \
           -d "question=Can you explain the main features of FastAPI?"

2. **Asking Questions**:
   - Enter your question in the input field
   - Click "Ask" to get a response based on the stored knowledge
   - Use Curl to ask questions:
      curl -X POST "http://localhost:8080/query" \
           -H "Content-Type: application/x-www-form-urlencoded" \
           -d "question=What%27s the difference between FastAPI %26 Flask%3F

## Technical Details

- The application uses FAISS for efficient similarity search
- Text embeddings are generated using the SentenceTransformer model
- Mistral 7B is used through their API for generating responses
- The web interface is built with FastAPI and uses Tailwind CSS for styling 
