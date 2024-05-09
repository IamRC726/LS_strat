"""Database Operations"""

import pathlib
import sqlite3
import sys

import pandas as pd

import environ.env
from fire_cap.models import Stock


# this is a pointer to the module object instance itself.
this = sys.modules[__name__]

# The connection to this sqlite database
this.__conn__ = None

__TBL_STK_HISTORY__ = "stock_history"
__TBL_MCAP_STK__ = "mcap_stock"
__TBL_RISK__ = "risk_rate"


def initialize_db(explode=False):
    """Initialize the Database"""
    source = environ.env.CONFIG.db_location
    if this.__conn__ is None:
        if explode is True:
            pathlib.Path(source).unlink()
        msg = "Database source {0}"
        print(msg.format(source))
        conn = sqlite3.connect(source)
        if check_db(conn) is False:
            import_schema(conn)
        this.__conn__ = conn
    else:
        msg = "Database is already initialized to {0}"
        raise RuntimeWarning(msg.format(source))
    return conn


def check_db(conn):
    table_name = __TBL_STK_HISTORY__
    query = f"SELECT count(name) FROM sqlite_master WHERE type='table' AND name='{table_name}'"
    rtn = conn.execute(query).fetchone()
    return rtn[0] > 0


def import_schema(conn):
    f = open("sql/schema.sql", "r")
    sql = f.read()
    conn.cursor().executescript(sql)
    conn.commit()


def store_df(data: pd.DataFrame, table_name: str, if_exists="replace"):
    """Store a Dataframe into the Database in table 'table_name'"""
    conn = this.__conn__
    return data.to_sql(table_name, conn, if_exists=if_exists)


def store_mcap_stocks(df: pd.DataFrame, mode="replace"):
    """Store the DF with Market Cap and ETF Tickers"""
    store_df(df, __TBL_MCAP_STK__, if_exists=mode)


def stock_factory(cursor, row):
    """
    Used to convert raw results row into fire_cap.models.Stock object
    """
    fields = [column[0] for column in cursor.description]
    return Stock(**{k: v for k, v in zip(fields, row)})


def arr_mcap_stocks():
    """
    retrieve mcap stocks as array of fire_cap.models.Stock
    """
    conn = this.__conn__
    old_row_factory = conn.row_factory
    conn.row_factory = stock_factory
    query = f"SELECT industry, market_cap, ticker from {__TBL_MCAP_STK__} order by ticker asc"
    rtn = conn.execute(query).fetchall()
    conn.row_factory = old_row_factory
    return list(rtn)


def store_risk_rate(df: pd.DataFrame, mode="append"):
    """Store the DF Risk Rate"""
    store_df(df, __TBL_RISK__, if_exists=mode)


def risk_rate():
    """Retrieve the Latest Risk Rate from the DB"""
    conn = this.__conn__
    query = f"SELECT rate, date from {__TBL_RISK__} order by date desc"
    res = conn.execute(query)
    rtn = res.fetchone()
    return rtn[0]


def arr_risk_rates():
    """Retrieve the All Risk Rates from the DB"""
    conn = this.__conn__
    query = f"SELECT rate, date from {__TBL_RISK__} order by date desc"
    res = conn.execute(query)
    rtn = res.fetchall()
    return rtn


def store_stk_history(stk: str, history_df):
    """
    Store the history for the Stock in DB
    so that we don't need to invoke yfinance every time
    """
    history_df["ticker"] = stk
    store_df(history_df, __TBL_STK_HISTORY__, if_exists="append")


def delete_stk_history(stk, date_str):
    conn = this.__conn__
    query = f"DELETE FROM {__TBL_STK_HISTORY__} where ticker = '{stk}' and Date = '{date_str}';"
    res = conn.execute(query)
    res.fetchone()
    conn.commit()
    return


def clear_stk_history(stk):
    conn = this.__conn__
    query = f"DELETE FROM {__TBL_STK_HISTORY__} where ticker = '{stk}'';"
    res = conn.execute(query)
    res.fetchone()
    conn.commit()
    return


def last_stk_history(stk: str):
    """Get the Date of the last entry for the stock"""
    conn = this.__conn__
    query = f"SELECT Date, ticker from {__TBL_STK_HISTORY__} where ticker = '{stk}' order by Date desc LIMIT 1"
    res = conn.execute(query)
    rtn = res.fetchone()
    return None if rtn is None else rtn[0]


def earliest_stk_history(stk: str):
    """Get the date of the earliest entry for the stock"""
    conn = this.__conn__
    query = f"SELECT Date, ticker from {__TBL_STK_HISTORY__} where ticker = '{stk}' order by Date asc LIMIT 1"
    res = conn.execute(query)
    rtn = res.fetchone()
    return None if rtn is None else rtn[0]


def df_stks_adj_close(stks):
    """
    Return a Dataframe of the combined 'Adj Close' values
    for the Stock Tickers in the argument array. Each column will
    be a stock's adjacent closed, with the stocks joined on date.
    """
    dfs = [*list(map(df_stk_adj_close, stks))]
    rtn = pd.concat(dfs, axis=1, join="inner")
    rtn.dropna(how="any", axis=0, inplace=True)
    return rtn


def df_stk_adj_close(stk: Stock):
    """
    Return a Dataframe of the Date-"Adj Close" of an individual stock
    Indexed on Date, with the Adj Close column renamed to the stock ticker
    """
    conn = this.__conn__
    ticker = stk if isinstance(stk, (str)) else stk.ticker
    data = pd.read_sql_query(
        f'select Date, "Adj Close" from {__TBL_STK_HISTORY__} where ticker = "{ticker}" order by DATE',
        conn,
        index_col="Date",
    )
    return data.rename(columns={"Adj Close": ticker}).dropna(how="any", axis=0)
