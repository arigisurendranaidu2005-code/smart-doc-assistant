"""
Main FastAPI backend for the RAG Chatbot project.
"""
from fastapi import FastAPI, UploadFile, File, Form, WebSocket, WebSocketDisconnect, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import logging

# Basic setup for logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Smart Doc Assistant API", version="1.0.0")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request / Response Models
class QueryRequest(BaseModel):
    query: str
    collection_name: str = "default"
    top_k: int = 5

class SourceDocument(BaseModel):
    content: str
    metadata: Dict[str, Any]
    score: float

class QueryResponse(BaseModel):
    answer: str
    sources: List[SourceDocument]

class CollectionInfo(BaseModel):
    name: str
    document_count: int

# Mock Rate Limiter dependency
def rate_limiter():
    # In a real app, implement rate limiting here (e.g., using redis)
    pass

@app.post("/upload", status_code=202)
async def upload_documents(
    files: List[UploadFile] = File(...),
    collection_name: str = Form("default"),
    rate_limit=Depends(rate_limiter)
):
    """
    Upload documents (PDF, DOCX, TXT), process and store in vector DB.
    """
    logger.info(f"Received {len(files)} files for collection: {collection_name}")
    processed_files = []
    
    for file in files:
        if not file.filename.lower().endswith(('.pdf', '.docx', '.txt')):
            raise HTTPException(status_code=400, detail=f"Unsupported file format: {file.filename}")
        
        # Mock processing
        processed_files.append(file.filename)
    
    return {"message": "Files successfully uploaded and processed.", "processed": processed_files}

@app.post("/query", response_model=QueryResponse)
async def query_documents(request: QueryRequest, rate_limit=Depends(rate_limiter)):
    """
    Ask a question, get answer with source citations.
    """
    logger.info(f"Querying collection {request.collection_name}: {request.query}")
    
    # Mock response
    return QueryResponse(
        answer=f"This is a mocked answer for the query: '{request.query}'.",
        sources=[
            SourceDocument(
                content="Mock content from document.",
                metadata={"source": "example.pdf", "page": 1},
                score=0.95
            )
        ]
    )

@app.get("/collections", response_model=List[CollectionInfo])
async def list_collections():
    """
    List all document collections.
    """
    return [
        CollectionInfo(name="default", document_count=10),
        CollectionInfo(name="finance_docs", document_count=5)
    ]

@app.delete("/collections/{name}")
async def delete_collection(name: str):
    """
    Delete a collection.
    """
    logger.info(f"Deleting collection: {name}")
    return {"message": f"Collection '{name}' deleted successfully."}

@app.get("/health")
async def health_check():
    """
    Health check endpoint.
    """
    return {"status": "healthy"}

# WebSocket manager
class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)

    async def send_personal_message(self, message: str, websocket: WebSocket):
        await websocket.send_text(message)

manager = ConnectionManager()

@app.websocket("/ws/chat")
async def websocket_chat(websocket: WebSocket):
    """
    Streaming chat via WebSocket.
    """
    await manager.connect(websocket)
    try:
        while True:
            data = await websocket.receive_text()
            logger.info(f"Received WS message: {data}")
            # Mock streaming response
            await manager.send_personal_message(f"You said: {data}", websocket)
            await manager.send_personal_message("Streaming answer chunk 1...", websocket)
            await manager.send_personal_message("Streaming answer chunk 2...", websocket)
    except WebSocketDisconnect:
        manager.disconnect(websocket)
        logger.info("Client disconnected")
