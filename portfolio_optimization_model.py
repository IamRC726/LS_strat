# -*- coding: utf-8 -*-
"""
Created on Sun Jan  7 20:47:43 2024

@author: rober
"""

# %%% imports

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import yfinance as yf

import environ.env

from pypfopt import EfficientFrontier
from pypfopt import risk_models
from pypfopt import expected_returns
from pypfopt import plotting
from pypfopt import black_litterman
from pypfopt.black_litterman import BlackLittermanModel
from pypfopt import objective_functions

import vectorbt as vbt

from monte_carlo import gbm


environ.env.init()

# %%% portfolio tickers & date

# %%%% Choose time period for analysis
start = "2017-01-01"
end = "2024-04-17"

# %%%% portfolios

rc_sector_pf = [
    "IYE",
    "IYR",
    "IYM",
    "IYJ",
    "IYC",
    "IYK",
    "IYH",
    "IYF",
    "IYW",
    "IDU",
    "IJR",
    "EZU",
    "EWU",
    "EWJ",
    "EMXC",
    "MCHI",
    "VNQI",
    "WM"
]

rc_factor_pf = [
    "MGC",
    "VO",
    "VB",
    "IUSG",
    "IUSV",
    "VWO",
    "VEA",
    "AGG",
    "TIP",
    "VNQ",
    "GSG",
]

leon_pf = [
    "NVDA",
    "META",
    "CCCC",
    "CL",
    "AAPL",
    "TSLA",
    "SPY",
    "AB",
    "ABNB",
    "ALLY",
    "AXP",
    "BOX",
    "COIN",
    "CRS",
    "DHI",
    "GE",
    "GEHC",
    "GME",
    "MACK",
    "MU",
    "RBLX",
    "REGN",
    "UWMC",
    "DIS",
]

simpson_pf = ["SPY", "DKNG", "INTC"]


# %%%% Choose portfolio for analysis
tickers = rc_sector_pf
# tickers = leon_pf
# tickers = simpson_pf


# %%% set global assumptions
risk_free_rate = 0.02
frequency = 252

# solvers
solver = "SCS"

# %%% Read Mkt Cap data via excel

mcap_df = pd.read_excel("capiq_mcap_ts.xlsx", sheet_name="mcap_ts", index_col="Date")
mcap_df = mcap_df.loc[start:end, tickers]


# Replace 'NA' values with NaN to ensure they are recognized as missing values
mcap_df.replace("NA", pd.NA, inplace=True)

# Use forward fill to replace NaN values with the last known value
mcap_df.fillna(method="ffill", inplace=True)


plt.figure(figsize=(25, 20))
# Plotting
ax = mcap_df.plot()
# Moving the legend below the plot
ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.4), shadow=True, ncol=4)
# Adjust layout
plt.tight_layout()
# Show plot
plt.show()

print("mkt cap weights")
print(mcap_df.iloc[-1] / mcap_df.iloc[-1].sum())

# %%% Price data for returns, volatility and correlation

# price data
# close_type = 'Close'
close_type = "Adj Close"
price_list = []
for i in tickers:
    stock = yf.download(i, interval="1d", start=start, end=end)[[close_type]].copy()
    stock.rename(columns={close_type: i}, inplace=True)
    price_list.append(stock)


prices_df = pd.concat(price_list, axis=1)
prices_df.dropna(how="any", axis=0, inplace=True)

correlation_df = (prices_df / prices_df.shift(1) - 1).corr()
volatility = (prices_df / prices_df.shift(1) - 1).std() * np.sqrt(frequency)

# mkt cap from yf

# mcap_list = []
# for i in tickers:
#     mkt_cap = yf.Ticker(i).info['marketCap']
#     mcap_list.append(mkt_cap)
#     print(mkt_cap)

# mcap_df = pd.concat(mcap_list, axis=1)
# mcap_df.dropna(how="any", axis=0, inplace=True)


# %%% optimize portfolio based on CAPM returns

# Calculate expected returns and sample covariance

# historical returns
returns_hist = expected_returns.mean_historical_return(prices_df)

# CAPM returns
# MSCI prices
mkt_price = yf.download("URTH", interval="1d", start=start)[
    [close_type]
]  # this is the global equities portfolio


returns_capm = expected_returns.capm_return(
    prices=prices_df,
    market_prices=mkt_price,
    risk_free_rate=risk_free_rate,
    compounding=True,
    frequency=frequency,
)

mu = returns_capm
cov_matrix = risk_models.sample_cov(prices_df)

hist_volatility = np.sqrt(np.diag(cov_matrix))
asset_historical = pd.DataFrame(
    {
        "historical returns": returns_hist,
        "CAPM returns": returns_capm,
        "volatility": hist_volatility,
    }
)


# Optimize for maximal Sharpe ratio
ef = EfficientFrontier(mu, cov_matrix)


raw_weights = ef.max_sharpe(risk_free_rate=risk_free_rate)
cleaned_weights = ef.clean_weights()
cleaned_weights_df = pd.DataFrame(
    list(cleaned_weights.items()), columns=["Ticker", "Weight"]
)

print()
print("weightings based on simple MVO")
print(cleaned_weights_df)

expected_annual_return, annual_volatility, sharpe_ratio = ef.portfolio_performance(
    verbose=True
)

# store max sharpe solution
max_sharpe_row = {
    "Volatility": annual_volatility,
    "Return": expected_annual_return,
    "Sharpe Ratio": sharpe_ratio,
}

# Add weights for each asset in the portfolio
for ticker, weight in cleaned_weights.items():
    max_sharpe_row[ticker] = weight

# Convert the max_sharpe_row to a DataFrame to ensure compatibility for appending
max_sharpe_df = pd.DataFrame(max_sharpe_row, index=["Max Sharpe"])


# %%% plotting and labling assets

# Optimize the portfolio for maximal Sharpe ratio
ef = EfficientFrontier(mu, cov_matrix, solver=solver)

# Create the efficient frontier plot
fig, ax = plt.subplots()
plotting.plot_efficient_frontier(ef, ax=ax, show_assets=True, show_tickers=True)

ax.set_title("Efficient Frontier (Basic)")

# Calculate risk (volatility) for each asset
asset_risks = np.sqrt(np.diag(cov_matrix))


# Save and show the plot
plt.savefig("efficient_frontier_with_labels.png")
plt.show()

# %%% display weights


# Initialize the Efficient Frontier
ef = EfficientFrontier(mu, cov_matrix, solver=solver)

# Get the minimum and maximum volatility for the portfolios on the frontier

# min_volatility = np.sqrt(np.min(np.diag(cov_matrix)))

min_volatility_portfolio = ef.min_volatility()
min_volatility = ef.portfolio_performance(verbose=False)[1]  # [1] is the volatility

# Reset the frontier to use it again
ef = EfficientFrontier(mu, cov_matrix, solver=solver)
max_volatility = np.sqrt(
    np.max(np.diag(cov_matrix))
)  # A rough estimate of max volatility

# Generate a range of target volatilities
volatility_range = np.linspace(min_volatility, max_volatility, 20)

# Dictionary to hold the optimized weights and performance metrics for each target volatility
portfolio_details = {}

for target_vol in volatility_range:
    ef = EfficientFrontier(mu, cov_matrix, solver=solver)
    ef.efficient_risk(target_volatility=target_vol)
    weights = ef.clean_weights()
    performance = ef.portfolio_performance(verbose=False)
    portfolio_details[f"Volatility: {target_vol:.2f}"] = {
        "Weights": weights,
        "Performance": performance,
    }

# Assuming 'portfolio_details' is your dictionary containing the results
data = {
    "Volatility": [],
    "Return": [],
    "Sharpe Ratio": [],
    # Add keys for weights if needed
}

# Initialize columns for each asset
assets = ef.tickers
for asset in assets:
    data[asset] = []

# Adding to the data dictionary
for vol, details in portfolio_details.items():
    data["Volatility"].append(details["Performance"][1])
    data["Return"].append(details["Performance"][0])
    data["Sharpe Ratio"].append(details["Performance"][2])
    for asset in assets:
        data[asset].append(
            details["Weights"].get(asset, 0)
        )  # Use 0 if the asset is not in the weights

optport_results = pd.DataFrame(data)
optport_results = pd.concat([optport_results, max_sharpe_df], axis=0)

print(optport_results)


# %%% Black-Litterman

# %%%% construct prior view

# market price of is the msci benchmark
cov_matrix = risk_models.CovarianceShrinkage(prices_df).ledoit_wolf()
delta = black_litterman.market_implied_risk_aversion(
    mkt_price[close_type]
)  # note the parameter need to be a Series
print(delta)

plotting.plot_covariance(cov_matrix, plot_correlation=True)

# %%%%% Market Cap
mcaps = dict(zip(mcap_df.columns, mcap_df.iloc[-1, :]))
"""
use if market cap data is from the Capital IQ Excel workbook 
"""
# mcaps = {}
# for t in tickers:
#     stock = yf.Ticker(t)
#     mcaps[t] = stock.info["marketCap"]

print(mcaps)

market_prior = black_litterman.market_implied_prior_returns(mcaps, delta, cov_matrix)

print("prior view")
print(market_prior)


# %%%% Discretionary Views
# %%%%% returns & confidence

# summarize different return expectations to help arrive at discretionary views

returns_summary = pd.DataFrame(
    {"capm": returns_capm, "historical": returns_hist, "BL_mkt_prior": market_prior}
)

"""
will need to build function to apply relative views or input results from the VAR model
"""
# absolute returns & confidence

# for RC
discretionary_inputs = {
    "IYW": (0.16, 0.8),
    "EWJ": (0.10, 0.8),
    "MCHI": (-0.0, 0.9),
    "WM": (0.07, 0.8)
}

# for Leon
# discretionary_inputs = {}
# discretionary_inputs = {"NVDA": (0.45, 0.9), "SPY": (0.08, 0.8), "AAPL": (0.10, 0.6)}

# for Simpson
# discretionary_inputs = {"INTC": (0.14, 0.6)}


# Extracting viewdict and confidences from the combined dictionary
viewdict = {ticker: values[0] for ticker, values in discretionary_inputs.items()}
# Creating the views DataFrame
views_df = pd.DataFrame(list(viewdict.items()), columns=["Ticker", "View"])
views_df.set_index("Ticker", drop=True, inplace=True)

confidences = [values[1] for values in discretionary_inputs.values()]


bl = BlackLittermanModel(
    cov_matrix,
    pi=market_prior,
    absolute_views=viewdict,
    omega="idzorek",
    view_confidences=confidences,
)

fig, ax = plt.subplots(figsize=(7, 7))
im = ax.imshow(bl.omega)

# We want to show all ticks...
ax.set_xticks(np.arange(len(bl.tickers)))
ax.set_yticks(np.arange(len(bl.tickers)))

ax.set_xticklabels(bl.tickers)
ax.set_yticklabels(bl.tickers)
plt.show()

np.diag(bl.omega)

# %%%%%% confidence specified by Standard Deviation

"""
alternative method:
Instead of inputting confidences, calculate the uncertainty matrix directly by specifying 1 standard deviation confidence intervals,
i.e bounds which we think will contain the true return 68% of the time.
# """
# intervals = [
#     (0, 0.25),
#     (0.1, 0.4),
#     (-0.1, 0.15),
#     (-0.05, 0.1),
#     (0.15, 0.25),
#     (-0.1, 0),
#     (0.1, 0.2),
#     (0.08, 0.12),
#     (0.1, 0.9),
#     (0, 0.3),
#     (0, 0.3),
# ]
# variances = []
# for lb, ub in intervals:
#     sigma = (ub - lb) / 2
#     variances.append(sigma**2)

# print(variances)
# omega = np.diag(variances)

# print("omega", omega)

# bl = BlackLittermanModel(
#     cov_matrix,
#     pi="market",
#     market_caps=mcaps,
#     risk_aversion=delta,
#     absolute_views=viewdict,
#     omega=omega,
# )

# %%%% posterior estimate of returns

# Posterior estimate of returns
ret_bl = bl.bl_returns()
rets_df = pd.DataFrame(
    [market_prior, ret_bl, pd.Series(viewdict)], index=["Prior", "Posterior", "Views"]
).T

print(rets_df)
# Assuming rets_df is your DataFrame
ax = rets_df.plot.bar(figsize=(12, 8))

# Adding value labels to each bar
for container in ax.containers:
    ax.bar_label(container, fmt="%.2f")

plt.show()

# covariances
S_bl = bl.bl_cov()
plotting.plot_covariance(S_bl)

# %%%% summarize expected returns and BL outputs

returns_summary = pd.merge(
    left=returns_summary, right=views_df, left_index=True, right_index=True, how="left"
)
returns_summary["BL_posterior"] = ret_bl

print(returns_summary)

# %%%% Black Litterman Weights

ef = EfficientFrontier(ret_bl, S_bl, solver=solver)
ef.add_objective(objective_functions.L2_reg)
ef.max_sharpe()
bl_weights = ef.clean_weights()
print(bl_weights)


# Create a Pandas Series from the dictionary
bl_weights_series = pd.Series(bl_weights)

expected_annual_return, annual_volatility, sharpe_ratio = ef.portfolio_performance(
    verbose=True
)

# store max sharpe solution
bl_max_sharpe_row = {
    "Volatility": annual_volatility,
    "Return": expected_annual_return,
    "Sharpe Ratio": sharpe_ratio,
}

# Add weights for each asset in the portfolio
for ticker, weight in bl_weights.items():
    bl_max_sharpe_row[ticker] = weight

# Convert the max_sharpe_row to a DataFrame to ensure compatibility for appending
bl_max_sharpe_df = pd.DataFrame(bl_max_sharpe_row, index=["Max Sharpe"])


# %%%% show portfolios on the efficient frontier

# Assuming ret_bl and S_bl are your posterior returns and covariance matrix from the Black-Litterman model
mu = ret_bl
cov_matrix = S_bl

# Initialize the Efficient Frontier with the Black-Litterman expected returns and covariance matrix
ef = EfficientFrontier(mu, cov_matrix, solver=solver)
ef.add_objective(objective_functions.L2_reg)
# plot efficient frontier using Black Litterman Model

# Create the efficient frontier plot
fig, ax = plt.subplots()
plotting.plot_efficient_frontier(ef, ax=ax, show_assets=True, show_tickers=True)

ax.set_title("Efficient Frontier with Black-Litterman Model")

plt.savefig("BL_efficient_frontier_with_labels.png")
plt.show()


# Generate a range of target volatilities
ef = EfficientFrontier(mu, cov_matrix, solver=solver)  # reinitiate solver
ef.add_objective(objective_functions.L2_reg)

min_volatility_portfolio = ef.min_volatility()
min_vol = ef.portfolio_performance(verbose=False)[1]
# min_vol = np.sqrt(np.min(np.diag(cov_matrix)))  # use if not able to sovle for min vol
max_vol = np.sqrt(np.max(np.diag(cov_matrix)))
# Estimate of min/max volatility based on the diagonal of the covariance matrix

volatility_range = np.linspace(min_vol, max_vol, 20)

# Dictionary to store optimized weights and performance metrics for each target volatility
portfolio_details = {}

for target_vol in volatility_range:
    ef = EfficientFrontier(
        mu, cov_matrix, solver=solver
    )  # Reinitialize for each iteration
    ef.efficient_risk(target_volatility=target_vol)
    weights = ef.clean_weights()
    performance = ef.portfolio_performance(
        verbose=False
    )  # Set verbose to True to print out the performance
    portfolio_details[f"Volatility: {target_vol:.2%}"] = {
        "Weights": weights,
        "Performance": performance,
    }

# Preparing data for DataFrame
data = {
    "Volatility": [],
    "Return": [],
    "Sharpe Ratio": [],
}

# Initialize columns for each asset
assets = list(weights.keys())  # Extracting asset names from the last set of weights

for asset in assets:
    data[asset] = []

# Populate data dictionary with portfolio details
for detail in portfolio_details.values():
    data["Volatility"].append(detail["Performance"][1])
    data["Return"].append(detail["Performance"][0])
    data["Sharpe Ratio"].append(detail["Performance"][2])
    for asset in assets:
        data[asset].append(detail["Weights"][asset])

# Convert dictionary to DataFrame for a nicer display
bl_optport_results = pd.DataFrame(data)
# bl_optport_results = pd.concat([bl_optport_results, bl_max_sharpe_df])

print(bl_optport_results)


# %%% back test & simulation
# %%%% vbt_backtest function


def vbt_backtest(
    weights,
    prices_df,
    size_type="targetpercent",
    group_by=True,
    cash_sharing=True,
    freq="days",
    years_freq="252 days",
    seed=42,
    incl_unrealized=True,
):
    # settings
    vbt.settings.array_wrapper["freq"] = freq
    vbt.settings.returns["year_freq"] = years_freq
    vbt.settings.portfolio["seed"] = seed
    vbt.settings.portfolio.stats["incl_unrealized"] = incl_unrealized

    # Create dataframe with appropriate size
    port_decision_size = np.full(prices_df.shape, np.nan)
    port_decision_size[
        0, :
    ] = weights  # allocate at first timestamp, do nothing afterwards

    # Backtesting process
    port_backtest = vbt.Portfolio.from_orders(
        close=prices_df,
        size=port_decision_size,
        size_type=size_type,
        group_by=group_by,
        cash_sharing=cash_sharing,
    )

    results = port_backtest.stats()
    return results


"""
select portolio with preferred risk-return profile
"""

# %%%% backtesting

bt_sharpe_ratio = []
bt_Sortino_ratio = []
bt_total_return = []
bt_annualized_return = []
bt_maxdrawdown = []
bt_end_value = []
bt_start_value = []
stored_results = []

for i in bl_optport_results.index:
    port_decision_index = i
    port_decision_wts = bl_optport_results.loc[i, tickers]
    results = vbt_backtest(port_decision_wts, prices_df, "targetpercent")
    bt_sharpe_ratio.append(results["Sharpe Ratio"].round(4))
    bt_Sortino_ratio.append(results["Sortino Ratio"].round(4))
    bt_total_return.append(results["Total Return [%]"].round(3))
    bt_maxdrawdown.append(results["Max Drawdown [%]"].round(3))
    stored_results.append(results)

    # Assuming results['Period'] gives you a Timedelta object
    period_days = results["Period"].days - 1  # Number of days in the period
    period_years = period_days / frequency  # Convert days to years

    # Calculate the Annualized Return for the current iteration
    end_value = results["End Value"]
    start_value = results["Start Value"]
    annualized_return = (end_value / start_value) ** (1 / period_years) - 1
    annualized_return_percent = round(annualized_return * 100, 3)
    bt_annualized_return.append(annualized_return_percent)

# After the loop, create a DataFrame from the collected metrics
bt_metrics_df = pd.DataFrame(
    {
        "BT_Sharpe Ratio": bt_sharpe_ratio,
        "BT_Sortino Ratio": bt_Sortino_ratio,
        "BT_Total Return [%]": bt_total_return,
        "BT_Annualized Return [%]": bt_annualized_return,
        "BT_Max Drawdown [%]": bt_maxdrawdown,
    }
)

# Concatenate the new DataFrame with the existing bl_optport_results DataFrame
bl_optport_results = pd.concat([bl_optport_results, bt_metrics_df], axis=1)

"""
Sharpe Ratio: Measures performance against a risk-free asset, adjusting for risk.
Calmar Ratio: Assesses risk-adjusted returns focusing on downside risk over typically three years.
Omega Ratio: Evaluates the likelihood of achieving returns above a threshold versus falling below it.
Sortino Ratio: Similar to the Sharpe Ratio but focuses only on downside risk.
"""
# %%%% monte carlo simulation

simulation_in_yrs = 1
trials = 50000

paths = []
percentiles = []
stored_simulation = []

for i in bl_optport_results.index:
    simulation = gbm(
        bl_optport_results.loc[i, "Return"],
        simulation_in_yrs,
        trials,
        bl_optport_results.loc[i, "Volatility"],
    )

    stored_simulation.append(simulation)
    paths.append(simulation.paths)
    percentiles.append(simulation.percentiles_df)

pct_df = pd.concat(percentiles, axis=1).T.reset_index(drop=True)
pct_df.columns = [str(col) + " percentile" for col in pct_df.columns]

bl_optport_results = pd.concat([bl_optport_results, pct_df], axis=1)


# %%% examine detailed results for a single portfolio

port_decision_index = 3
port_decision_details = bl_optport_results.iloc[port_decision_index, :].to_frame()

print()
print(stored_results[port_decision_index])

# compare to major indices


def benchmark_compare(benchmark_ticker):
    benchmark = yf.download(benchmark_ticker, interval="1d", start=start, end=end)[
        close_type
    ]
    benchmark_returns = benchmark / benchmark.shift(1) - 1
    benchmark_returns = benchmark_returns.dropna()
    benchmark_avg_return = (benchmark[-1] / benchmark[0]) ** (
        frequency / len(benchmark_returns)
    ) - 1
    benchmark_vol = benchmark_returns.std() * np.sqrt(frequency)
    benchmark_sharpe = (benchmark_avg_return - risk_free_rate) / benchmark_vol
    print()
    print(f"--- comparison ({benchmark_ticker})---")
    print("mean return: ", benchmark_avg_return)
    print("volatility: ", benchmark_vol)
    print("sharpe ratio: ", benchmark_sharpe)


benchmark_compare("SPY")
benchmark_compare("URTH")

# %%%% montecarlo results


simulation = stored_simulation[port_decision_index]

# plot path
simulation.plot_paths()

# plot PDE of CAGR
simulation.plot_distribution()


# %%%% shares to purchase base on decision

port_val = 48000

# get the current minute price
price_list = []
for i in tickers:
    stock = yf.download(i, interval="1m", period="7d")[["Close"]].copy()
    stock.rename(columns={"Close": i}, inplace=True)
    price_list.append(stock)

current_price = pd.concat(price_list, axis=1).ffill().iloc[-1, :]


port_decision_wts = bl_optport_results.loc[port_decision_index, tickers]
port_val_allocations = port_decision_wts * port_val
port_shares = port_val_allocations / current_price

shares_decision_df = pd.DataFrame(
    {
        "shares": port_shares,
        "price": current_price,
        "$amnt": port_val_allocations,
        "weights": port_decision_wts,
    }
)

print("---share to purchase (partial shares)---")
print(shares_decision_df)

rounded_shares = shares_decision_df["shares"].round()
amnt = rounded_shares * shares_decision_df["price"]
wts = amnt / amnt.sum()
cash = port_val - (rounded_shares * current_price).sum()

rounded_shares_decision_df = pd.DataFrame(
    {
        "shares": rounded_shares,
        "price": shares_decision_df["price"],
        "$amnt": amnt,
        "weights": wts,
    }
)

print("---rounded to full shares---")
print(rounded_shares_decision_df)
print("residual cash", cash)

# %%% output to excel
x = pd.ExcelWriter("out/portfolio_optimization.xlsx")


asset_historical.to_excel(x, "asset_historical_perf")
correlation_df.to_excel(x, sheet_name="correlation_matrix")
cov_matrix.to_excel(x, "covariance_matrix")
returns_summary.to_excel(x, sheet_name="returns_summary")
bl_optport_results.to_excel(x, sheet_name="bl_optport_results")
port_decision_details.to_excel(x, "port decision details")
rounded_shares_decision_df.to_excel(x, "shares to purchase")


x.close()
