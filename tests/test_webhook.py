import asyncio
import httpx
import sys

async def test():
    webhook_url = "add your discord webhook url here" 
    content = "Test from Travel Planner – webhook is alive!"
    async with httpx.AsyncClient(timeout=10.0) as client:
        resp = await client.post(webhook_url, json={"content": content})
        print(f"Status: {resp.status_code}")
        if resp.status_code == 204:
            print("Success! Message sent to Discord.")
        else:
            print(f"Error: {resp.text}")

if __name__ == "__main__":
    asyncio.run(test())