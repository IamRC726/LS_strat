# -*- coding: utf-8 -*-
"""
Created on Tue Feb 27 23:08:54 2024

@author: rober
"""

import numpy as np
import pandas as pd
from datetime import datetime
import pytz


import vectorbt as vbt


# %%% Define params
symbols = ["AMZN", "NFLX", "GOOG", "AAPL", "NVDA"]
start_date = datetime(2010, 1, 1, tzinfo=pytz.utc)
end_date = datetime(2024, 2, 28, tzinfo=pytz.utc)
num_tests = 2000

vbt.settings.array_wrapper["freq"] = "days"
vbt.settings.returns["year_freq"] = "252 days"
vbt.settings.portfolio["seed"] = 42
vbt.settings.portfolio.stats["incl_unrealized"] = True


# %%% get price

yfdata = vbt.YFData.download(symbols, start=start_date, end=end_date)
print(yfdata.symbols)

ohlcv = yfdata.concat()
print(ohlcv.keys())

price = ohlcv["Close"]

returns = price.pct_change()

# %%% generate weights
np.random.seed(42)

# Generate random weights, n times
weights = []
for i in range(num_tests):
    w = np.random.random_sample(len(symbols))
    w = w / np.sum(w)
    weights.append(w)

print(len(weights))

# %%% column hierarchy

# Build column hierarchy such that one weight corresponds to one price series
_price = price.vbt.tile(
    num_tests, keys=pd.Index(np.arange(num_tests), name="symbol_group")
)
_price = _price.vbt.stack_index(pd.Index(np.concatenate(weights), name="weights"))

print(_price.columns)

# %%% Define order size
size = np.full_like(_price, np.nan)
size[0, :] = np.concatenate(
    weights
)  # allocate at first timestamp, do nothing afterwards

# %%% Run simulation
pf = vbt.Portfolio.from_orders(
    close=_price,
    size=size,
    size_type="targetpercent",
    group_by="symbol_group",
    cash_sharing=True,
)  # all weights sum to 1, no shorting, and 100% investment in risky assets

print(len(pf.orders))


print(size.shape)

# %%% Plot annualized return against volatility, color by sharpe ratio
annualized_return = pf.annualized_return()
annualized_return.index = pf.annualized_volatility()
annualized_return.vbt.scatterplot(
    trace_kwargs=dict(
        mode="markers",
        marker=dict(
            color=pf.sharpe_ratio(),
            colorbar=dict(title="sharpe_ratio"),
            size=5,
            opacity=0.7,
        ),
    ),
    xaxis_title="annualized_volatility",
    yaxis_title="annualized_return",
).show_svg()

# %%% Get index of the best group according to the target metric
best_symbol_group = pf.sharpe_ratio().idxmax()

print(best_symbol_group)
