"""Request and response models. The request is the API's data contract:
it mirrors the Pandera schema used in training (src/data.py)."""

from pydantic import BaseModel, Field


class PredictionRequest(BaseModel):
    value: float = Field(gt=0, description="export value this year, thousand USD")
    value_prev: float = Field(ge=0, description="last year's value; 0 = not exported")
    value_prev2: float = Field(ge=0, description="value two years ago; 0 = not exported")
    years_exported: int = Field(ge=1, le=3, description="years in a row, 3 = 3 or more")
    market_value: float = Field(gt=0, description="destination's imports of the product")
    market_share: float = Field(gt=0, le=1, description="Sri Lanka's share of that market")
    market_suppliers: int = Field(ge=1, description="countries supplying that market")
    world_value: float = Field(gt=0, description="world trade in the product")
    price_ratio: float = Field(ge=0.01, le=100, description="unit price vs world average")
    n_destinations: int = Field(ge=1, description="countries Sri Lanka sells the product to")
    n_products_to_destination: int = Field(ge=1)
    hs_chapter: str = Field(pattern=r"^HS\d{2}$", examples=["HS09"])
    destination: str = Field(pattern=r"^[A-Z0-9]{3}$", examples=["GBR"])


class PredictionResponse(BaseModel):
    request_id: str
    probability: float
    prediction: int
    model_version: str


class FeedbackRequest(BaseModel):
    request_id: str
    actual_outcome: int = Field(ge=0, le=1)
