from fastapi import APIRouter, Depends
from pydantic import BaseModel
from langchain_core.messages import HumanMessage
from backend.app.dependencies import get_agent

router = APIRouter(prefix="/chat", tags=["chat"])

class ChatRequest(BaseModel):
    query: str

class ChatResponse(BaseModel):
    answer: str

@router.post("/", response_model=ChatResponse)
async def chat(request: ChatRequest, agent=Depends(get_agent)):
    inputs = {"messages": [HumanMessage(content=request.query)]}
    result = await agent.ainvoke(inputs)
    final_message = result["messages"][-1].content
    return ChatResponse(answer=final_message)
    