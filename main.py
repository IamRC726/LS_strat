"""FireTeam Portfolio Optimization"""

import argparse
import warnings
import pandas as pd

from pypfopt import expected_returns, risk_models

from environ import env
from db import db

from fire_cap.analytics.analytics import (
    BLImpliedDiagram,
    BLForecastDiagram,
    Input,
    OptResults,
    OptWeightPerformance,
    VolatilityEFDiagram,
)
from fire_cap.utils import frequency


## Main Entry Point
def main(outputfile, start, end):
    env.init()
    db.initialize_db()
    # % Get Profile
    risk_free_rate = db.risk_rate()
    # % Get Tickers (& associated Market Cap MM) for Profile
    mcap_stks = db.arr_mcap_stocks()
    # % For Each Ticker, Retrieve Financial History
    df_stocks = db.df_stks_adj_close(mcap_stks)
    # % Retrieve Market Price Financial History
    mkt_price = db.df_stk_adj_close(env.CONFIG.market_price_stk)

    # % Prepare for Analysis
    capm_return = expected_returns.capm_return(
        prices=df_stocks,
        market_prices=mkt_price,
        risk_free_rate=risk_free_rate,
        compounding=True,
        frequency=frequency(env.CONFIG.stock_interval),
    )
    sample_cov = risk_models.sample_cov(df_stocks)

    mcaps = {}
    for stk in mcap_stks:
        mcaps[stk.ticker] = stk.market_cap

    input = Input(
        **{
            "risk_free_rate": risk_free_rate,
            "capm_return": capm_return,
            "sample_cov": sample_cov,
            "df_prices": df_stocks,
            "mkt_price": mkt_price,
            "mcap": mcaps,
        }
    )
    trackers = [
        OptWeightPerformance(),
        OptResults(),
        VolatilityEFDiagram(),
        BLImpliedDiagram(),
        BLForecastDiagram(),
    ]
    # % Run Analysis
    for tracker in trackers:
        tracker.exec(input.model_copy())

    # % Generate Reports
    excel_file = pd.ExcelWriter(outputfile)
    for tracker in trackers:
        dfs = tracker.excel_frames()
        for df in dfs:
            data_frame = df
            data_frame["df"].to_excel(excel_file, sheet_name=data_frame["sheet_name"])

    excel_file.close()


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-r",
        "--report",
        help="The report Output location",
        default="out/report.xlsx",
    )
    args = parser.parse_args()
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
    main(args.report, args.start, args.end)
