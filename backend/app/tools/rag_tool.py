import logging
from pydantic import BaseModel, Field
from backend.app.rag.retrieval import retrieve_similar_destinations

logger = logging.getLogger(__name__)

class RAGToolInput(BaseModel):
    query: str = Field(..., description="Natural language query about destinations")

class RAGToolOutput(BaseModel):
    result: str

def create_rag_tool(embedder):
    """Factory that returns an async function with embedder bound."""
    async def rag_tool(input: RAGToolInput) -> RAGToolOutput:
        try:
            results = await retrieve_similar_destinations(input.query, embedder, top_k=3)
        except Exception as e:
            logger.error(f"RAG tool failed: {e}", exc_info=True)
            return RAGToolOutput(result="Sorry, I couldn't retrieve destination info at the moment.")
        if not results:
            return RAGToolOutput(result="No relevant destination information found.")
        output = "Here are some relevant destinations:\n"
        for r in results:
            output += f"- {r['destination_name']}: {r['content']} (similarity: {r['similarity']:.2f})\n"
        return RAGToolOutput(result=output)
    return rag_tool