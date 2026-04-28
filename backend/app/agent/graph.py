import logging
from langgraph.graph import StateGraph, END
from langchain_groq import ChatGroq
from langchain_core.messages import ToolMessage, AIMessage
from langchain_core.tools import tool
from typing import TypedDict, Annotated, List, Any
import operator

from backend.app.tools.rag_tool import create_rag_tool, RAGToolInput
from backend.app.tools.style_tool import create_style_tool, StyleToolInput
from backend.app.tools.weather_tool import weather_tool, WeatherToolInput

logger = logging.getLogger(__name__)

def build_agent(model, embedder, settings):
    # Create tools with dependencies injected
    rag_tool = create_rag_tool(embedder)
    style_tool = create_style_tool(model, settings.DATA_PATH)   # reuse model

    @tool
    async def rag_wrapper(query: str) -> str:
        """Retrieve information about destinations based on a natural language query."""
        result = await rag_tool(RAGToolInput(query=query))
        return result.result

    @tool
    async def style_wrapper(destination_name: str) -> str:
        """Get the travel style (Adventure, Relaxation, Culture, Budget, Luxury, Family) for a specific destination."""
        result = await style_tool(StyleToolInput(destination_name=destination_name))
        return result.result

    @tool
    async def weather_wrapper(city: str) -> str:
        """Get current weather for a city."""
        result = await weather_tool(WeatherToolInput(city=city))
        return result.result

    tools = [rag_wrapper, style_wrapper, weather_wrapper]

    llm = ChatGroq(
        model="llama-3.3-70b-versatile",
        temperature=0.3,
        api_key=settings.GROQ_API_KEY
    )
    llm_with_tools = llm.bind_tools(tools)

    class AgentState(TypedDict):
        messages: Annotated[List[Any], operator.add]

    async def call_model(state: AgentState):
        try:
            response = await llm_with_tools.ainvoke(state["messages"])
            return {"messages": [response]}
        except Exception as e:
            logger.error(f"LLM call failed: {e}")
            return {"messages": [AIMessage(content="Sorry, I'm having trouble thinking right now.")]}

    async def call_tools(state: AgentState):
        last_message = state["messages"][-1]
        results = []
        for tc in last_message.tool_calls:
            tool_fn = next((t for t in tools if t.name == tc["name"]), None)
            if tool_fn:
                try:
                    res = await tool_fn.ainvoke(tc["args"])
                except Exception as e:
                    logger.error(f"Tool {tc['name']} failed: {e}", exc_info=True)
                    res = f"Tool execution error"
                results.append(
                    ToolMessage(content=res, tool_call_id=tc["id"], name=tc["name"])
                )
            else:
                results.append(
                    ToolMessage(
                        content=f"Error: Tool {tc['name']} not found",
                        tool_call_id=tc["id"],
                        name=tc["name"]
                    )
                )
        return {"messages": results}

    def should_continue(state: AgentState):
        last = state["messages"][-1]
        if hasattr(last, "tool_calls") and last.tool_calls:
            return "tools"
        return "end"

    workflow = StateGraph(AgentState)
    workflow.add_node("agent", call_model)
    workflow.add_node("tools", call_tools)
    workflow.set_entry_point("agent")
    workflow.add_conditional_edges(
        "agent", should_continue, {"tools": "tools", "end": END}
    )
    workflow.add_edge("tools", "agent")

    return workflow.compile()