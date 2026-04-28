# backend/app/tools/style_tool.py
import logging
import pandas as pd
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

class StyleToolInput(BaseModel):
    destination_name: str = Field(..., description="Name of the destination")

class StyleToolOutput(BaseModel):
    result: str

def create_style_tool(model, data_path):
    """Factory that accepts a pre-loaded model and dataset path."""
    df = pd.read_csv(data_path)
    logger.info("Style tool ready")

    async def style_tool(input: StyleToolInput) -> StyleToolOutput:
        row = df[df['name'] == input.destination_name]
        if row.empty:
            return StyleToolOutput(result=f"Destination '{input.destination_name}' not found.")
        X = row.drop(columns=['name', 'style'])
        pred = model.predict(X)[0]
        proba = model.predict_proba(X)[0]
        confidence = max(proba)
        return StyleToolOutput(
            result=f"{input.destination_name} is best for {pred} travel (confidence: {confidence:.2f})."
        )
    return style_tool