"""FireTeam Portfolio Seed"""

import argparse
import datetime
import pandas
import warnings

from environ import env
from db import db
from fire_cap.populate import populate_stk_history


## Main Entry Point
def seed(file: str, risk_rate: float, start_date="2010-01-01", end_date=None):
    """Utility to Seed the Database with the current stock and risk_rate"""
    industries_df = pandas.read_excel(file, sheet_name="Industries")
    ## rename columns to be more SQL friendly
    industries_df.rename(
        columns={
            "ETF tracker": "ticker",
            "Market Cap MM": "market_cap",
            "Industry": "industry",
        },
        inplace=True,
    )

    db.store_mcap_stocks(industries_df)

    risk_df = pandas.DataFrame()
    risk_df["rate"] = [risk_rate]
    risk_df["date"] = [datetime.date.today()]
    db.store_risk_rate(risk_df)

    stocks = db.arr_mcap_stocks()
    for stk in stocks:
        populate_stk_history(
            stk.ticker,
            env.CONFIG.stock_interval,
            start_date=start_date,
            end_date=end_date,
        )
    # populate the market maker stock
    populate_stk_history(
        env.CONFIG.market_price_stk,
        env.CONFIG.stock_interval,
        start_date=start_date,
        end_date=end_date,
    )


def verify_seed(risk_rate):
    """Utility to verify the seed stored the correct last risk rate"""
    rate = db.risk_rate()
    assert rate == risk_rate


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-f",
        "--file",
        help="The Market Cap excel file location",
        default="Marketcap.xlsx",
    )
    parser.add_argument(
        "-r",
        "--risk",
        type=float,
        help="The Risk Rate of the Portfolio",
        default=0.02,
    )
    parser.add_argument(
        "-x",
        "--xplode",
        help="Explode i.e Remove the Database File",
        type=bool,
        default=False,
    )
    parser.add_argument(
        "-s",
        "--start",
        help="Start Date 'yyyy-mm-dd' to start seeding data from",
        default="2010-01-01",
    )
    parser.add_argument(
        "-e",
        "--end",
        help="End Date 'yyyy-mm-dd' to end seeding data from",
        default=None,
    )
    args = parser.parse_args()
    env.init()
    db.initialize_db(args.xplode)

    seed(file=args.file, risk_rate=args.risk, start_date=args.start, end_date=args.end)
    verify_seed(args.risk)
