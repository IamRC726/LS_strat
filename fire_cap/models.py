from pydantic import BaseModel


class Stock(BaseModel):
    """Stock Encapsulation"""

    industry: str
    ticker: str
    market_cap: float
