# environ.py
"""Utility for Configuration Loading from dot_env files"""
import os
from pydantic import BaseModel
from dotenv import dotenv_values

# Global Scope Variable to easily access
# Specific Config
CONFIG = None


class Config(BaseModel):
    """Encapsulation of Valid Configuration Variables"""

    # FRED API Key
    fred_key: str
    # Location of the Database
    db_location: str
    # Interval when fetching data from yfinance
    stock_interval: str
    # Stock to use when comparing against a market price
    market_price_stk: str


def init():
    """
    Initialize Configuration Environment from .env.shared, .env, and os
    environment overrides
    """
    global CONFIG
    if CONFIG is None:
        print("Initializing Config")
        raw = {
            **dotenv_values(".env.shared"),  # shared environment
            **dotenv_values(".env"),  # local overrides
            **os.environ,  # environment overrides
        }
        CONFIG = Config(
            **{
                "fred_key": raw["FRED_KEY"],
                "db_location": raw["DB_LOCATION"],
                "stock_interval": raw["STOCK_INTERVAL"],
                "market_price_stk": raw["MARKET_PRICE_STOCK"],
            }
        )
    else:
        print("Config already initialized")
    return CONFIG


if __name__ == "__main__":
    init()
    print(CONFIG)
