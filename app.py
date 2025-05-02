from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi import Request
import os
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from typing import List
import json
from dotenv import load_dotenv
from mistralai.client import MistralClient
from mistralai.models.chat_completion import ChatMessage

load_dotenv()

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

########################################################
# OLD CODE WITH SentenceTransformer
# Remove SentenceTransformer, use Mistral for embeddings
# dimension = 384  # old
# model = SentenceTransformer('all-MiniLM-L6-v2')
########################################################

dimension = 1024  # Mistral-embed output dimension

# Initialize FAISS index
index = faiss.IndexFlatL2(dimension)
texts = []

# Initialize Mistral client
mistral_client = MistralClient(api_key=os.getenv("MISTRAL_API_KEY"))

# Helper function to split text into 2048-character chunks
CHUNK_SIZE = 2048

def chunk_text(text, chunk_size=CHUNK_SIZE):
    print("Chunking text ...")
    chunks = [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]
    print(f"# of chunks: {len(chunks)}")
    return chunks

def get_text_embedding(text):
    response = mistral_client.embeddings(
        model="mistral-embed",
        input=[text]
    )
    return response.data[0].embedding

#def get_mistral_response(prompt: str, context: str) -> str:
def get_mistral_response(prompt, context):
    try:
        messages = [
            ChatMessage(role="system", content=f"You are a helpful assistant. Use the following context to answer the question: {context}"),
            ChatMessage(role="user", content=prompt)
        ]
        
        chat_response = mistral_client.chat(
            model="open-mistral-7b",
            messages=messages
        )
        
        return chat_response.choices[0].message.content
    except Exception as e:
        return f"Error getting response from Mistral: {str(e)}"

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/add_text")
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

@app.post("/upload_file")
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

@app.post("/query")
async def query(question: str = Form(...)):
    # Generate embedding for the question using Mistral
    question_embedding = np.array(get_text_embedding(question), dtype=np.float32)

    # Search in FAISS
    k = 3  # Number of similar texts to retrieve
    distances, indices = index.search(np.array([question_embedding]).reshape(1, -1), k)

    # Get the most relevant context
    context = " ".join([texts[i] for i in indices[0] if i < len(texts)])

    # Get response from Mistral
    response = get_mistral_response(question, context)
    
    return {"response": response}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="localhost", port=8080) 
