from pydantic import BaseModel
from typing import Optional, List, Dict, Any


class IngestRequest(BaseModel):
    content: str
    content_type: str  # pdf | txt | text
    source: str = "discord"
    metadata: Dict[str, Any] = {}


class IngestResponse(BaseModel):
    status: str
    chunks_created: int
    total_chars: int
    message: str


class QueryRequest(BaseModel):
    query: str
    mode: Optional[str] = None  # orientacao | resposta-cliente | bug


class QueryResponse(BaseModel):
    response: str
    mode: str
    score: float
    chunks_used: int
    processing_time_ms: float


class LogEntry(BaseModel):
    timestamp: str
    mode: str
    query: str
    score: float
    response_length: int
    processing_time_ms: float
    chunks_used: int


class Document(BaseModel):
    id: str
    content: str
    embedding: List[float]
    metadata: Dict[str, Any]
    score: Optional[float] = None


class HistoryEntry(BaseModel):
    """Single conversation entry stored in database"""
    id: Optional[str] = None
    user_id: str
    pergunta: str
    resposta: str
    modo: str = "orientacao"
    score: float = 0.0
    chunks_used: int = 0
    processing_time_ms: int = 0
    timestamp: Optional[str] = None


class HistoryContextRequest(BaseModel):
    """Request to get formatted history context for injection"""
    user_id: str
    limit: int = 5  # Last 5 conversations


class HistoryContextResponse(BaseModel):
    """Formatted conversation history for prompt injection"""
    formatted_context: str
    entry_count: int
    oldest_timestamp: Optional[str] = None
