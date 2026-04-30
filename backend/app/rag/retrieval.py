import httpx
from sqlalchemy import text
from backend.app.core.database import AsyncSessionLocal
from backend.app.core.config import settings

EMBEDDING_MODEL = "intfloat/e5-base-v2"          # 384‑dim, free on OpenRouter
EMBEDDING_API_URL = "https://openrouter.ai/api/v1/embeddings"


async def _get_query_embedding(text: str) -> list[float]:
    """Fetch 384‑dim embedding from OpenRouter (free)."""
    async with httpx.AsyncClient(timeout=30) as client:
        resp = await client.post(
            EMBEDDING_API_URL,
            headers={
                "Authorization": f"Bearer {settings.OPENROUTER_API_KEY}",
                "Content-Type": "application/json",
                "HTTP-Referer": "http://localhost:5173",
                "X-Title": "Smart Travel Planner",
            },
            json={"model": EMBEDDING_MODEL, "input": text},
        )
        resp.raise_for_status()
        data = resp.json()
        return data["data"][0]["embedding"]


async def retrieve_similar_destinations(query: str, top_k: int = 5) -> list:
    """
    Return a list of the top‑k destinations (best chunk per destination)
    using cosine similarity via pgvector.
    """
    query_embedding = await _get_query_embedding(query)

    # Fetch a generous window, then deduplicate in Python
    sql = text("""
        SELECT destination_name, content,
               1 - (embedding <=> :embedding) AS similarity
        FROM destination_documents
        ORDER BY embedding <=> :embedding
        LIMIT 100                     -- fetch more rows so dedup doesn't miss destinations
    """)

    async with AsyncSessionLocal() as db:
        result = await db.execute(
            sql,
            {"embedding": query_embedding, "top_k": top_k},
        )
        rows = result.fetchall()

        # Keep only the best chunk per destination
        best = {}
        for row in rows:
            dest = row.destination_name
            if dest not in best or row.similarity > best[dest]["similarity"]:
                best[dest] = {
                    "destination_name": dest,
                    "content": row.content,
                    "similarity": row.similarity,
                }

        # Sort by similarity (highest first) and take top_k destinations
        sorted_dests = sorted(
            best.values(), key=lambda x: x["similarity"], reverse=True
        )[:top_k]

        return sorted_dests