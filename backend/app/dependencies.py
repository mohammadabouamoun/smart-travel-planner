from fastapi import Request
from backend.app.core.config import settings

def get_agent(request: Request):
    return request.app.state.agent

def get_model(request: Request):
    return request.app.state.model

def get_embedder(request: Request):
    return request.app.state.embedder

def get_settings(request: Request):
    return request.app.state.settings

# For direct use when request is not available (e.g., in scripts)
def get_settings_sync():
    return settings