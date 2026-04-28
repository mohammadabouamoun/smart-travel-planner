import asyncio
from fastapi import APIRouter, Depends
from pydantic import BaseModel
from langchain_core.messages import HumanMessage
from backend.app.dependencies import get_agent
from backend.app.auth import get_current_user
from backend.app.core.database import get_db
from backend.app.core.models import User, AgentRun
from backend.app.core.config import settings
from backend.app.utils.webhook import send_discord_webhook
from sqlalchemy.ext.asyncio import AsyncSession
import json
import logging
from datetime import datetime, timezone

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/chat", tags=["chat"])

class ChatRequest(BaseModel):
    query: str

class ChatResponse(BaseModel):
    answer: str

async def send_webhook_safe(webhook_url: str, content: str):
    """Wrapper to log any webhook errors without raising."""
    try:
        await send_discord_webhook(webhook_url, content)
    except Exception as e:
        logger.error(f"Discord webhook failed: {e}", exc_info=True)

@router.post("/", response_model=ChatResponse)
async def chat(
    request: ChatRequest,
    agent=Depends(get_agent),
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    inputs = {"messages": [HumanMessage(content=request.query)]}
    result = await agent.ainvoke(inputs)
    final_message = result["messages"][-1].content

    # Extract tool names called
    tools_fired = []
    for msg in result["messages"]:
        if hasattr(msg, "tool_calls") and msg.tool_calls:
            for tc in msg.tool_calls:
                tools_fired.append(tc["name"])   # dictionary key

    # Log the agent run
    agent_run = AgentRun(
        user_id=current_user.id,
        query=request.query,
        answer=final_message,
        tools_fired=json.dumps(tools_fired),
        created_at=datetime.now(timezone.utc)
    )
    db.add(agent_run)
    try:
        await db.commit()
    except Exception:
        await db.rollback()
        logger.exception("Failed to save agent run")

    # Send webhook asynchronously (fire and forget, with error logging)
    if settings.DISCORD_WEBHOOK_URL:
        webhook_content = (
            f"**New Agent Run**\n\n"
            f"**User:** {current_user.email}\n"
            f"**Query:** {request.query}\n"
            f"**Answer:** {final_message[:500]}...\n"
            f"**Tools Fired:** {', '.join(tools_fired)}"
        )
        asyncio.create_task(send_webhook_safe(settings.DISCORD_WEBHOOK_URL, webhook_content))

    return ChatResponse(answer=final_message)