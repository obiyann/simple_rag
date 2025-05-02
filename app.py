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

# Initialize the sentence transformer model
model = SentenceTransformer('all-MiniLM-L6-v2')
dimension = 384  # Dimension of the embeddings

# Initialize FAISS index
index = faiss.IndexFlatL2(dimension)
texts = []

# Initialize Mistral client
mistral_client = MistralClient(api_key=os.getenv("MISTRAL_API_KEY"))

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
    # Generate embedding for the new text
    embedding = model.encode([text])[0]
    
    # Add to FAISS index
    index.add(np.array([embedding]).astype('float32'))
    texts.append(text)
    
    return {"message": "Text added successfully"}

@app.post("/upload_file")
async def upload_file(file: UploadFile = File(...)):
    content = await file.read()
    text = content.decode("utf-8")
    
    # Generate embedding for the text
    embedding = model.encode([text])[0]
    
    # Add to FAISS index
    index.add(np.array([embedding]).astype('float32'))
    texts.append(text)
    
    return {"message": "File uploaded and processed successfully"}

@app.post("/query")
async def query(question: str = Form(...)):
    # Generate embedding for the question
    question_embedding = model.encode([question])[0]
    
    # Search in FAISS
    k = 3  # Number of similar texts to retrieve
    distances, indices = index.search(np.array([question_embedding]).astype('float32'), k)
    
    # Get the most relevant context
    context = " ".join([texts[i] for i in indices[0] if i < len(texts)])
    
    # Get response from Mistral
    response = get_mistral_response(question, context)
    
    return {"response": response}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="localhost", port=8080) 