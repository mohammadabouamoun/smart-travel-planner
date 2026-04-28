import asyncio
from sqlalchemy import text
from backend.app.core.database import AsyncSessionLocal
from sentence_transformers import SentenceTransformer

embedder = SentenceTransformer('all-MiniLM-L6-v2')

async def retrieve_similar_destinations(query: str, top_k: int = 5) -> list:
    # Generate embedding for the query
    query_embedding = embedder.encode(query).tolist()
    # Convert to string format expected by pgvector
    embedding_str = "[" + ",".join(str(x) for x in query_embedding) + "]"

    sql = text("""
        SELECT destination_name, content,
               1 - (embedding <=> CAST(:embedding AS vector)) AS similarity
        FROM destination_documents
        ORDER BY embedding <=> CAST(:embedding AS vector)
        LIMIT 100   -- fetch more to allow deduplication
    """)

    async with AsyncSessionLocal() as db:
        result = await db.execute(sql, {"embedding": embedding_str})
        rows = result.fetchall()

        # Keep only the best chunk per destination
        best_per_dest = {}
        for row in rows:
            dest = row.destination_name
            if dest not in best_per_dest or row.similarity > best_per_dest[dest]["similarity"]:
                best_per_dest[dest] = {
                    "destination_name": dest,
                    "content": row.content,
                    "similarity": row.similarity
                }

        # Sort by similarity and return top_k destinations
        sorted_dests = sorted(best_per_dest.values(), key=lambda x: x["similarity"], reverse=True)[:top_k]
        return sorted_dests