# populate.py
"""
Module used to download stock history and store into the database layer
"""
import yfinance as yf
from datetime import datetime

from db import db

__DEFAULT_START__ = "2010-01-01"


def dl_stk_history(stk: str, interval="1d", start="2024-02-01", end=None):
    """
    Download Stock History
    """
    return yf.download(stk, interval=interval, start=start, end=end)


def populate_stk_history(stk: str, interval="1d", start_date=None, end_date=None):
    """
    Store Stock History, based on the last stored date
    or get all of history from start_date
    """
    earliest_start = db.earliest_stk_history(stk)
    latest_start = db.last_stk_history(stk)

    if start_date is None and latest_start is None:
        start_date = __DEFAULT_START__
    elif start_date is None and latest_start is not None:
        dt_obj = datetime.strptime(latest_start, "%Y-%m-%d %H:%M:%S")
        db.delete_stk_history(stk, latest_start)
        # delete the last entry since it might not be complete for the day
        start_date = f"{dt_obj.date()}"
    df = dl_stk_history(stk, interval=interval, start=start_date, end=end_date)
    db.store_stk_history(stk, df)
