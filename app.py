########
#! /usr/bin/env python3
# Basic RAG: simple FastAPI app to serve a web page and query a Mistral model
# Author: Yannick Guillerm - yannick.guillerm@gmail.com
# Date: 05/05/2025
########

from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi import Request
import os
import faiss
import numpy as np
from typing import List
import json
from dotenv import load_dotenv
from mistralai import Mistral
from ddtrace.llmobs import LLMObs
from ddtrace.llmobs.decorators import llm, tool, workflow, agent, embedding

# Load .env file
load_dotenv()

# Initialize FastAPI app
app = FastAPI()

# Initialize templates
templates = Jinja2Templates(directory="templates")

# Initialize FAISS index
dimension = 1024
index = faiss.IndexFlatL2(dimension)
texts = []

# Initialize Mistral client (updated)
client = Mistral(api_key=os.getenv("MISTRAL_API_KEY"))

# Size of chuncks to split files into
CHUNK_SIZE = 2048

# Initialize Datadog LLM Obs
dd_api_key=os.getenv("DATADOG_API_KEY")
LLMObs.enable(
  ml_app="testing_rag",
  api_key=dd_api_key,
  site="datadoghq.com",
  agentless_enabled=True,
)


# Function to split text into chunks
@tool("chunk_text")
def chunk_text(text, chunk_size=CHUNK_SIZE):
    print("Chunking text ...")
    chunks = [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]
    print(f"# of chunks: {len(chunks)}")
    return chunks

# Function to get embeddings from Mistral
@embedding(model_name="mistral-embed", model_provider="mistral")
def get_text_embedding(text):

    # DD span annotation
    LLMObs.annotate(
        input_data=text,
    )

    embeddings_batch_response = client.embeddings.create(
        model="mistral-embed",
        inputs=[text]
    )

    # DD span annotation
    LLMObs.annotate(
        output_data=embeddings_batch_response.data[0].embedding,
    )

    return embeddings_batch_response.data[0].embedding

# Function to get response from Mistral
@llm(model_name="mistral-7b", name="get_mistral_model_response", model_provider="mistral")
def get_mistral_model_response(prompt, context):

    # DD span annotation
    LLMObs.annotate(
        input_data=prompt,
    )


    print(f"Context: {context}")
    try:
        user_message = f"""
        Context information is below.
        ---------------------
        {context}
        ---------------------
        Given the context information and not prior knowledge, answer the query.
        Query: {prompt}
        Answer:
        """
        messages = [
            {"role": "user", "content": user_message}
        ]
        chat_response = client.chat.complete(
            model="open-mistral-7b",
            messages=messages
        )

        # DD span annotation
        LLMObs.annotate(
            output_data=chat_response.choices[0].message.content,
        )

        return chat_response.choices[0].message.content
    except Exception as e:
        return f"Error getting response from Mistral: {str(e)}"

# Route to serve the main HTML page
@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

# Route to add text
@app.post("/add_text")
@workflow("add_text")
async def add_text(text: str = Form(...)):
    # Split text into chunks
    chunks = chunk_text(text)
    count = 0
    for chunk in chunks:
        embedding = np.array(get_text_embedding(chunk), dtype=np.float32)
        index.add(np.array([embedding]))
        texts.append(chunk)
        count += 1
    print(f"Text added successfully as {count} chunk(s)")
    return {"message": f"Text added successfully as {count} chunk(s)"}

# Route to upload a file
@app.post("/upload_file")
@workflow("upload_file")
async def upload_file(file: UploadFile = File(...)):
    content = await file.read()
    text = content.decode("utf-8")
    # Split text into chunks
    chunks = chunk_text(text)
    count = 0
    for chunk in chunks:
        embedding = np.array(get_text_embedding(chunk), dtype=np.float32)
        index.add(np.array([embedding]))
        texts.append(chunk)
        count += 1
    print(f"File uploaded and processed successfully as {count} chunk(s)")
    return {"message": f"File uploaded and processed successfully as {count} chunk(s)"}

# Route to query the model
@app.post("/query")
@agent("query")
async def query(question: str = Form(...)):

    LLMObs.annotate(
        input_data=question,
    )

    # Generate embedding for the question using Mistral
    print("Generating embeddings for the user query")
    question_embedding = np.array([get_text_embedding(question)])

    # Search similarities in Vector DB (FAISS)
    print("Search similar context in the Vector DB")
    k = 3  # Number of similar texts to retrieve
    distances, indices = index.search(question_embedding, k)

    # Get the most relevant context
    print("Getting the most relevant context")
    if len(texts) != 0:
        # There is an existing context
        context = " ".join([texts[i] for i in indices[0] if i < len(texts)])
    else:
        # No context
        context = ""

    # Get response from Mistral
    print("Getting Mistral model response")
    response = get_mistral_model_response(question, context)

    LLMObs.annotate(
        output_data=response,
    )

    print(f"Response: {response}")
    
    return {"response": response}

# Run the app
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="localhost", port=8080) 
