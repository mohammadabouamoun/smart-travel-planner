import asyncio
from backend.app.rag.retrieval import retrieve_similar_destinations

async def main():
    query = "I want to go somewhere with beautiful beaches and temples"
    print(f"Query: {query}\n")
    results = await retrieve_similar_destinations(query, top_k=3)
    for r in results:
        print(f"Destination: {r['destination_name']}")
        print(f"Content: {r['content']}")
        print(f"Similarity: {r['similarity']:.4f}")
        print("-" * 50)

if __name__ == "__main__":
    asyncio.run(main())