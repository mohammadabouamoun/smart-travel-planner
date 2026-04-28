from sqlalchemy import text
from backend.app.core.database import AsyncSessionLocal

async def retrieve_similar_destinations(query: str, embedder, top_k: int = 5) -> list:
    # Generate embedding as list
    query_embedding = embedder.encode(query).tolist()
    # Convert to string representation for pgvector
    embedding_str = "[" + ",".join(str(x) for x in query_embedding) + "]"

    sql = text("""
        SELECT DISTINCT ON (destination_name)
               destination_name,
               content,
               1 - (embedding <=> CAST(:embedding AS vector)) AS similarity
        FROM destination_documents
        ORDER BY destination_name, embedding <=> CAST(:embedding AS vector)
        LIMIT :top_k
    """)

    async with AsyncSessionLocal() as db:
        result = await db.execute(sql, {"embedding": embedding_str, "top_k": top_k})
        rows = result.fetchall()
        return [
            {
                "destination_name": row.destination_name,
                "content": row.content,
                "similarity": row.similarity
            }
            for row in rows
        ]