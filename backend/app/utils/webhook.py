import httpx
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
    after_log
)
import logging

logger = logging.getLogger(__name__)

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=10),
    retry=retry_if_exception_type((
        httpx.TimeoutException,
        httpx.NetworkError,
        httpx.HTTPStatusError  # retry on server errors (5xx)
    )),
    after=after_log(logger, logging.WARNING)  # log each retry
)
async def send_discord_webhook(webhook_url: str, content: str) -> bool:
    """
    Send a message to a Discord webhook. Returns True if successful, False after all retries fail.
    """
    async with httpx.AsyncClient(timeout=10.0) as client:
        resp = await client.post(webhook_url, json={"content": content})
        resp.raise_for_status()
        return True