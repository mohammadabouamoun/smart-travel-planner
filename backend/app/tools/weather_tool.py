import logging
import httpx
from pydantic import BaseModel, Field
from cachetools import TTLCache
from asyncio import Lock
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from backend.app.core.config import settings

logger = logging.getLogger(__name__)

class WeatherToolInput(BaseModel):
    city: str = Field(..., description="City name")

class WeatherToolOutput(BaseModel):
    result: str

cache = TTLCache(maxsize=100, ttl=settings.WEATHER_CACHE_TTL)
cache_lock = Lock()

@retry(stop=stop_after_attempt(3),
       wait=wait_exponential(multiplier=1, min=1, max=10),
       retry=retry_if_exception_type((httpx.TimeoutException, httpx.NetworkError)))
async def fetch_weather(city: str) -> dict:
    url = "https://api.openweathermap.org/data/2.5/weather"
    params = {"q": city, "appid": settings.OPENWEATHER_API_KEY, "units": "metric"}
    async with httpx.AsyncClient(timeout=10.0) as client:
        resp = await client.get(url, params=params)
        resp.raise_for_status()
        return resp.json()

async def get_cached_weather(city: str):
    if city in cache:
        return cache[city]
    async with cache_lock:
        if city in cache:
            return cache[city]
        data = await fetch_weather(city)
        cache[city] = data
        return data

async def weather_tool(input: WeatherToolInput) -> WeatherToolOutput:
    if not settings.OPENWEATHER_API_KEY:
        return WeatherToolOutput(result="Weather API key not configured.")
    try:
        data = await get_cached_weather(input.city)
        temp = data["main"]["temp"]
        desc = data["weather"][0]["description"]
        humidity = data["main"]["humidity"]
        return WeatherToolOutput(
            result=f"Current weather in {input.city}: {temp}°C, {desc}, humidity {humidity}%."
        )
    except Exception as e:
        logger.error(f"Weather tool failed for '{input.city}': {e}", exc_info=True)
        return WeatherToolOutput(result="Sorry, I couldn't fetch the weather information right now.")