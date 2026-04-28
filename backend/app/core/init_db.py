import asyncio
from backend.app.core.database import engine, Base
from backend.app.core import models 
from sqlalchemy import text

async def init():
    async with engine.begin() as conn:
        await conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
        await conn.run_sync(Base.metadata.create_all)
    print("Database initialized successfully.")

if __name__ == "__main__":
    asyncio.run(init())