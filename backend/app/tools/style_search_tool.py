import pandas as pd
from pydantic import BaseModel, Field
from backend.app.core.config import settings

DATA_PATH = settings.DATA_PATH

class StyleSearchInput(BaseModel):
    style: str = Field(..., description="Travel style to search for (Adventure, Relaxation, Culture, Budget, Luxury, Family)")

class StyleSearchOutput(BaseModel):
    result: str

def create_style_search_tool():
    """Factory that returns an async function to search destinations by style."""
    df = pd.read_csv(DATA_PATH)

    async def style_search_tool(input: StyleSearchInput) -> StyleSearchOutput:
        style = input.style.strip().title()  # normalise capitalisation
        matching = df[df['style'].str.lower() == style.lower()]
        if matching.empty:
            return StyleSearchOutput(result=f"No destinations found with style '{style}'.")
        names = matching['name'].tolist()
        return StyleSearchOutput(
            result=f"Destinations with style {style}: {', '.join(names)}"
        )
    return style_search_tool