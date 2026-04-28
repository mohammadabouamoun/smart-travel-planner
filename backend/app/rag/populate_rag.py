import asyncio
import logging
from backend.app.core.database import AsyncSessionLocal
from backend.app.core.models import DestinationDocument
from sentence_transformers import SentenceTransformer
from sqlalchemy import text  # Import the text function

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load embedding model (384-dim)
embedder = SentenceTransformer('all-MiniLM-L6-v2')

# Expanded destinations (10 destinations, total 28 documents)
destinations_content = {
    "Bali, Indonesia": [
        "Bali is known for its volcanic mountains, iconic rice paddies, beaches and coral reefs.",
        "The island is home to many Hindu temples, including Uluwatu and Tanah Lot.",
        "Popular activities include surfing, yoga retreats, and hiking Mount Batur.",
        "The cultural heart of Bali is Ubud, famous for traditional dance and crafts."
    ],
    "Paris, France": [
        "Paris is famous for the Eiffel Tower, Louvre Museum, and Notre-Dame Cathedral.",
        "The city offers world-class cuisine, fashion, and art galleries.",
        "Romantic river cruises on the Seine are a must-do activity.",
        "Montmartre district offers stunning views and a bohemian atmosphere."
    ],
    "Kyoto, Japan": [
        "Kyoto is renowned for its classical Buddhist temples, gardens, and imperial palaces.",
        "The Fushimi Inari Shrine with its thousands of red torii gates is a major attraction.",
        "Arashiyama Bamboo Grove provides a unique walking experience.",
        "Kyoto is also famous for its traditional tea ceremonies and geisha culture."
    ],
    "New York, USA": [
        "New York City is known for Times Square, Central Park, and Broadway theaters.",
        "The Statue of Liberty is an iconic symbol of freedom.",
        "World-class museums include the Metropolitan Museum of Art and MoMA.",
        "Diverse neighborhoods offer a wide range of cuisines and cultures."
    ],
    "Cairo, Egypt": [
        "Cairo is home to the Great Pyramids of Giza and the Sphinx.",
        "The Egyptian Museum houses thousands of ancient artifacts, including Tutankhamun's treasures.",
        "The Nile River flows through the city, offering felucca rides.",
        "Islamic Cairo features historic mosques and bustling bazaars like Khan el-Khalili."
    ],
    "Rome, Italy": [
        "Rome is known for the Colosseum, Roman Forum, and Vatican City.",
        "The city offers incredible art, history, and authentic Italian cuisine.",
        "Throw a coin in the Trevi Fountain to ensure your return to Rome."
    ],
    "Bangkok, Thailand": [
        "Bangkok is famous for its ornate temples like Wat Phra Kaew and Wat Arun.",
        "The street food scene is world-renowned and very affordable.",
        "Take a boat ride through the floating markets or explore Chatuchak weekend market."
    ],
    "Cape Town, South Africa": [
        "Cape Town is dominated by Table Mountain, which can be hiked or climbed by cable car.",
        "The V&A Waterfront offers shopping, dining, and views of the harbor.",
        "Robben Island, where Nelson Mandela was imprisoned, is a short ferry ride away."
    ],
    "Istanbul, Turkey": [
        "Istanbul straddles two continents, with the Hagia Sophia and Blue Mosque as top attractions.",
        "The Grand Bazaar is one of the oldest and largest covered markets in the world.",
        "Take a Bosphorus cruise to see the city's skyline from the water."
    ],
    "Mexico City, Mexico": [
        "Mexico City is home to the Frida Kahlo Museum and the ancient Teotihuacan pyramids.",
        "The city’s historic center is a UNESCO World Heritage site with stunning architecture.",
        "Street food, especially tacos al pastor, is a must-try experience."
    ]
}

async def generate_embeddings(texts):
    """Run embedding in a thread to avoid blocking the event loop."""
    return await asyncio.to_thread(embedder.encode, texts)

async def populate(clear_existing=False):
    """Insert destination documents into the database."""
    async with AsyncSessionLocal() as db:
        if clear_existing:
            await db.execute(text("TRUNCATE destination_documents CASCADE"))
            logger.info("Cleared existing destination documents.")

        total_docs = 0
        for dest, texts in destinations_content.items():
            embeddings = await generate_embeddings(texts)
            docs = []
            for doc_text, emb in zip(texts, embeddings):  # renamed loop variable to doc_text
                doc = DestinationDocument(
                    destination_name=dest,
                    content=doc_text,
                    embedding=emb.tolist()
                )
                docs.append(doc)
            db.add_all(docs)
            await db.commit()
            total_docs += len(docs)
            logger.info(f"Inserted {len(docs)} docs for {dest}")

    logger.info(f"Finished. {total_docs} total documents inserted for {len(destinations_content)} destinations.")

if __name__ == "__main__":
    asyncio.run(populate(clear_existing=True))