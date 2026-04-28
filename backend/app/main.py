from contextlib import asynccontextmanager
from fastapi import FastAPI
import logging
import structlog
from backend.app.api import auth, chat
from backend.app.core.config import settings
from backend.app.core.database import engine
from backend.app.agent.graph import build_agent

# Structured logging
structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.JSONRenderer()
    ],
    context_class=dict,
    logger_factory=structlog.stdlib.LoggerFactory(),
)
logger = structlog.get_logger()
logging.basicConfig(level=logging.INFO)

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting up...")
    try:
        import joblib
        from sentence_transformers import SentenceTransformer

        logger.info("Loading ML model...")
        app.state.model = joblib.load(settings.MODEL_PATH)

        logger.info("Loading sentence transformer...")
        app.state.embedder = SentenceTransformer('all-MiniLM-L6-v2')

        logger.info("Creating agent...")
        app.state.agent = build_agent(
            app.state.model,
            app.state.embedder,
            settings
        )
        app.state.settings = settings
        logger.info("Startup complete.")
    except Exception as e:
        logger.exception("Fatal startup error")
        raise

    yield

    # Shutdown
    await engine.dispose()
    logger.info("Shutdown complete.")

app = FastAPI(lifespan=lifespan)

 # Include routers (outside lifespan)
app.include_router(auth.router)
app.include_router(chat.router)

@app.get("/health")
async def health():
    return {"status": "ok"}