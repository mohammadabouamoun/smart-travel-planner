from contextlib import asynccontextmanager
from fastapi import FastAPI
import logging
import structlog
from backend.app.api import auth, chat
from fastapi.middleware.cors import CORSMiddleware
from backend.app.core.config import settings
from backend.app.core.database import engine
from backend.app.agent.graph import build_agent
import joblib
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
        
       # from sentence_transformers import SentenceTransformer

        logger.info("Loading ML model...")
        app.state.model = joblib.load(settings.MODEL_PATH)

     #   logger.info("Loading sentence transformer...")
       # app.state.embedder = SentenceTransformer('all-MiniLM-L6-v2')

        logger.info("Creating agent...")
        app.state.agent = build_agent(app.state.model, settings)
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

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

 # Include routers (outside lifespan)
app.include_router(auth.router)
app.include_router(chat.router)

@app.get("/health")
async def health():
    return {"status": "ok"}