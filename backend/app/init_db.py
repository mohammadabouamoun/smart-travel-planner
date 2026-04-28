import asyncio
from backend.app.database import engine, Base
import backend.app.models  # this will load models to register with Base
from sqlalchemy import text

async def init():
    async with engine.begin() as conn:
        await conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
        await conn.run_sync(Base.metadata.create_all)
    print("Database initialized successfully.")

if __name__ == "__main__":
    asyncio.run(init())