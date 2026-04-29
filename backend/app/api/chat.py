import asyncio
import json
import logging
from datetime import datetime, timezone

from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from langchain_core.messages import HumanMessage, AIMessageChunk
from sqlalchemy.ext.asyncio import AsyncSession

from backend.app.dependencies import get_agent
from backend.app.auth import get_current_user
from backend.app.core.database import get_db, AsyncSessionLocal
from backend.app.core.models import User, AgentRun
from backend.app.core.config import settings
from backend.app.utils.webhook import send_discord_webhook

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/chat", tags=["chat"])

class ChatRequest(BaseModel):
    query: str

class ChatResponse(BaseModel):
    answer: str
    tools_fired: list[str] = []

async def send_webhook_safe(webhook_url: str, content: str):
    """Wrapper to log any webhook errors without raising."""
    try:
        await send_discord_webhook(webhook_url, content)
    except Exception as e:
        logger.error(f"Discord webhook failed: {e}", exc_info=True)

async def log_agent_run(user_id, query, answer, tools_fired):
    """Persist an AgentRun record using a fresh database session."""
    async with AsyncSessionLocal() as db:
        try:
            run = AgentRun(
                user_id=user_id,
                query=query,
                answer=answer,
                tools_fired=json.dumps(tools_fired),
                created_at=datetime.now(timezone.utc),
            )
            db.add(run)
            await db.commit()
        except Exception:
            await db.rollback()
            logger.exception("Failed to save agent run")

@router.post("/", response_model=ChatResponse)
async def chat(
    request: ChatRequest,
    agent=Depends(get_agent),
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    inputs = {"messages": [HumanMessage(content=request.query)]}
    result = await agent.ainvoke(inputs, config={"recursion_limit": 100})
    final_message = result["messages"][-1].content

    tools_fired = []
    for msg in result["messages"]:
        if hasattr(msg, "tool_calls") and msg.tool_calls:
            for tc in msg.tool_calls:
                tools_fired.append(tc["name"] if isinstance(tc, dict) else tc.name)

    try:
        run = AgentRun(
            user_id=current_user.id,
            query=request.query,
            answer=final_message,
            tools_fired=json.dumps(tools_fired),
            created_at=datetime.now(timezone.utc),
        )
        db.add(run)
        await db.commit()
    except Exception:
        await db.rollback()
        logger.exception("Failed to save agent run (non-streaming)")

    if settings.DISCORD_WEBHOOK_URL:
        webhook_content = (
            f"**New Agent Run**\n\n"
            f"**User:** {current_user.email}\n"
            f"**Query:** {request.query}\n"
            f"**Answer:** {final_message[:500]}...\n"
            f"**Tools Fired:** {', '.join(tools_fired)}"
        )
        asyncio.create_task(send_webhook_safe(settings.DISCORD_WEBHOOK_URL, webhook_content))

    return ChatResponse(answer=final_message, tools_fired=tools_fired)

@router.post("/stream")
async def chat_stream(
    request: ChatRequest,
    agent=Depends(get_agent),
    current_user: User = Depends(get_current_user),
):
    async def event_generator():
        inputs = {"messages": [HumanMessage(content=request.query)]}
        tools_fired = []
        full_response = ""
        final_output = None

        def _extract_text(content) -> str:
            """Safely extract text from chunk.content, which may be a string, list of strings, or list of dicts."""
            if isinstance(content, str):
                return content
            if isinstance(content, list):
                parts = []
                for item in content:
                    if isinstance(item, str):
                        parts.append(item)
                    elif isinstance(item, dict):
                        # Gemini-style part: {"text": "..."}
                        parts.append(item.get("text", ""))
                    else:
                        parts.append(str(item))
                return "".join(parts)
            return str(content) if content else ""

        try:
            async for event in agent.astream_events(inputs, config={"recursion_limit": 100}, version="v2"):
                kind = event["event"]
                if kind == "on_chat_model_stream":
                    chunk = event["data"]["chunk"]
                    if isinstance(chunk, AIMessageChunk):
                        content = chunk.content
                        if content:
                            text = _extract_text(content)
                            full_response += text
                            yield f"data: {json.dumps({'type': 'token', 'content': text})}\n\n"
                elif kind == "on_tool_start":
                    tool_name = event["name"]
                    tools_fired.append(tool_name)
                    yield f"data: {json.dumps({'type': 'tool_start', 'name': tool_name})}\n\n"
                elif kind == "on_tool_end":
                    tool_name = event["name"]
                    output = event["data"].get("output")
                    yield f"data: {json.dumps({'type': 'tool_end', 'name': tool_name, 'output': str(output)})}\n\n"
                elif kind == "on_chain_end":
                    # Capture the final graph state (messages may appear here without streaming)
                    output = event["data"]["output"]
                    if isinstance(output, dict) and "messages" in output:
                        final_output = output

            # If no streaming tokens were received, try to get the answer from the final state
            if not full_response and final_output:
                messages = final_output.get("messages", [])
                if messages:
                    last_msg = messages[-1]
                    if hasattr(last_msg, "content") and last_msg.content:
                        full_response = _extract_text(last_msg.content)

            if not full_response:
                full_response = "I'm sorry, I couldn't generate a response. Please try again."

            yield f"data: {json.dumps({'type': 'done', 'answer': full_response, 'tools_fired': tools_fired})}\n\n"

            # Background tasks (logging & webhook) with a fresh DB session
            asyncio.create_task(log_agent_run(current_user.id, request.query, full_response, tools_fired))

            if settings.DISCORD_WEBHOOK_URL:
                webhook_content = (
                    f"**New Agent Run (Stream)**\n\n"
                    f"**User:** {current_user.email}\n"
                    f"**Query:** {request.query}\n"
                    f"**Answer:** {full_response[:500]}...\n"
                    f"**Tools Fired:** {', '.join(tools_fired)}"
                )
                asyncio.create_task(send_webhook_safe(settings.DISCORD_WEBHOOK_URL, webhook_content))

        except Exception as e:
            logger.exception("Streaming error")
            yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")