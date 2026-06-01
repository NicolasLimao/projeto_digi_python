from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from src.api import history_routes, rag_routes
from src.logger import get_logger

logger = get_logger(__name__)

app = FastAPI(
    title="Digi RAG API",
    description="RAG-powered support agent for Digisac analysts",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(history_routes.router)
app.include_router(rag_routes.router)


@app.get("/health")
async def health():
    """Health check endpoint"""
    return {"status": "ok", "service": "digi-rag"}


@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "service": "Digi RAG API",
        "version": "1.0.0",
        "docs": "/docs"
    }


if __name__ == "__main__":
    import os
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
