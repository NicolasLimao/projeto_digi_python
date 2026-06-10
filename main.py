import os

# Sentry — instrumentação opcional. Só inicializa se SENTRY_DSN estiver definido.
# Mantenha NO TOPO do arquivo, antes de qualquer outro import pesado, para
# capturar erros de inicialização também.
_sentry_dsn = os.environ.get("SENTRY_DSN")
if _sentry_dsn:
    import sentry_sdk
    from sentry_sdk.integrations.fastapi import FastApiIntegration
    from sentry_sdk.integrations.starlette import StarletteIntegration

    sentry_sdk.init(
        dsn=_sentry_dsn,
        integrations=[
            StarletteIntegration(transaction_style="endpoint"),
            FastApiIntegration(transaction_style="endpoint"),
        ],
        traces_sample_rate=float(os.environ.get("SENTRY_TRACES_SAMPLE_RATE", "0.1")),
        profiles_sample_rate=float(os.environ.get("SENTRY_PROFILES_SAMPLE_RATE", "0.1")),
        environment=os.environ.get("ENVIRONMENT", "production"),
        release=os.environ.get("RELEASE_VERSION"),
        send_default_pii=False,  # não enviar dados sensíveis do usuário
    )

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from src.api import history_routes, rag_routes, ingest_routes
from src.logger import get_logger

logger = get_logger(__name__)
if _sentry_dsn:
    logger.info(f"[Sentry] Instrumentação ativa (env={os.environ.get('ENVIRONMENT','production')})")

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
app.include_router(ingest_routes.router)


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
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
