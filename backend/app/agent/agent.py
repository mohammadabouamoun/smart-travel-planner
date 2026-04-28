import logging
from typing import TypedDict, Annotated, List, Any
import operator
import os
from dotenv import load_dotenv
from langgraph.graph import StateGraph, END
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from langchain_core.tools import tool

# Import your existing async functions
from backend.app.tools.rag_tool import rag_tool
from backend.app.tools.style_tool import style_tool
from backend.app.tools.weather_tool import weather_tool

load_dotenv()
logger = logging.getLogger(__name__)

# ---- Wrap tools with @tool decorator ----
@tool
async def rag_tool_wrapper(query: str) -> str:
    """Retrieve relevant destination information based on a natural language query."""
    return await rag_tool(query)

@tool
async def style_tool_wrapper(destination_name: str) -> str:
    """Get the travel style (Adventure, Relaxation, Culture, Budget, Luxury, Family) for a specific destination name."""
    return await style_tool(destination_name)

@tool
async def weather_tool_wrapper(city: str) -> str:
    """Get current weather for a city. Input is a city name."""
    return await weather_tool(city)

tools = [rag_tool_wrapper, style_tool_wrapper, weather_tool_wrapper]

# ---- LLM setup ----
llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0.3,
    api_key=os.getenv("GROQ_API_KEY")
)
llm_with_tools = llm.bind_tools(tools)

# ---- Agent state ----
class AgentState(TypedDict):
    messages: Annotated[List[Any], operator.add]

# ---- Helper to execute a tool ----
async def execute_tool(name: str, args: dict) -> str:
    """Find the right tool wrapper and call it."""
    tool_map = {t.name: t for t in tools}
    tool_fn = tool_map.get(name)
    if not tool_fn:
        return f"Error: Tool '{name}' not found."
    try:
        # @tool-decorated functions are always async-compatible
        result = await tool_fn.ainvoke(args)
        return str(result)
    except Exception as e:
        logger.error(f"Tool '{name}' execution failed: {e}", exc_info=True)
        return f"Tool execution error: {str(e)}"

# ---- Graph nodes ----
async def call_model(state: AgentState):
    response = await llm_with_tools.ainvoke(state["messages"])
    return {"messages": [response]}

async def call_tools(state: AgentState):
    last_message = state["messages"][-1]
    # Tool calls are objects, not dicts
    tool_calls = last_message.tool_calls

    results = []
    for tc in tool_calls:
        tool_result = await execute_tool(tc["name"], tc["args"])
        results.append(ToolMessage(
            content=tool_result,
            tool_call_id=tc["id"],      # id is an attribute
            name=tc["name"]
        ))
    return {"messages": results}

# ---- Conditional edge ----
def should_continue(state: AgentState):
    last_message = state["messages"][-1]
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "tools"
    return "end"

# ---- Build the graph ----
workflow = StateGraph(AgentState)
workflow.add_node("agent", call_model)
workflow.add_node("tools", call_tools)
workflow.set_entry_point("agent")
workflow.add_conditional_edges(
    "agent",
    should_continue,
    {"tools": "tools", "end": END}
)
workflow.add_edge("tools", "agent")

agent = workflow.compile()