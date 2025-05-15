import os
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
from langchain_unstructured import UnstructuredLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_ollama.embeddings import OllamaEmbeddings
from langchain_ollama import OllamaLLM
from langchain.chains import RetrievalQA

app = FastAPI(title="RAG Pipeline API")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize global variables
embedding = OllamaEmbeddings(model="mxbai-embed-large")
llm = OllamaLLM(model="llama3:70b")
vectordb = None

class Query(BaseModel):
    question: str

def process_documents(docs):
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(docs)
    return Chroma.from_documents(chunks, embedding)

def initialize_vectordb():
    global vectordb
    docs = []
    if os.path.exists("docs"):
        for file in os.listdir("docs"):
            if file.endswith((".pdf", ".md", ".txt")):
                loader = UnstructuredLoader(os.path.join("docs", file))
                docs.extend(loader.load())
    if docs:
        vectordb = process_documents(docs)

@app.on_event("startup")
async def startup_event():
    initialize_vectordb()

@app.post("/query")
async def query_endpoint(query: Query):
    if not vectordb:
        raise HTTPException(status_code=400, detail="No documents have been indexed yet")
    
    qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=vectordb.as_retriever())
    answer = qa_chain.invoke({"query": query.question})
    return {"answer": answer}

@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    if not file.filename.endswith((".pdf", ".md", ".txt")):
        raise HTTPException(status_code=400, detail="Unsupported file type")
    
    # Create docs directory if it doesn't exist
    os.makedirs("docs", exist_ok=True)
    
    # Save the uploaded file
    file_path = os.path.join("docs", file.filename)
    with open(file_path, "wb") as f:
        content = await file.read()
        f.write(content)
    
    # Process the new document
    loader = UnstructuredLoader(file_path)
    docs = loader.load()
    
    # Update the vector database
    global vectordb
    if vectordb is None:
        vectordb = process_documents(docs)
    else:
        # Add new documents to existing vector store
        splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        chunks = splitter.split_documents(docs)
        vectordb.add_documents(chunks)
    
    return {"message": f"Successfully processed {file.filename}"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
