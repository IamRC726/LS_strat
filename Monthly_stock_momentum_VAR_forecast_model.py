# %%% imports

import matplotlib.pyplot as plt
from matplotlib.dates import MonthLocator, DateFormatter
import numpy as np
import pandas as pd
from pandas.plotting import scatter_matrix
from statsmodels.tools.eval_measures import rmse
from datetime import datetime, timedelta

from pandas.tseries.offsets import MonthEnd
from fredapi import Fred
import quandl as quandl
import yfinance as yf


from dotenv import load_dotenv
import os

# api key management

from macro_data import MacroData
from timeseries_analysis import TimeSeriesAnalysis, logr
import var_analysis

# custom classes

load_dotenv()
FRED_API_KEY = os.getenv("FRED_API_KEY", "dummy")

fred = Fred(api_key=FRED_API_KEY)


# %%% get adjusted closing data from yfinance

security = "IYW"  # the security we want to forecast

"""
additional securities
"""
factor_etfs = ["MTUM"]
asset_etfs = ["GLD"]


tickers = [security] + factor_etfs + asset_etfs

# close_type = 'Adj Close'
close_type = "Close"

stocks_list = []
for i in tickers:
    stock = yf.download(i, interval="1mo")[[close_type]]
    stock.index = stock.index + MonthEnd(0)
    stock.rename(columns={close_type: i}, inplace=True)
    print(stock.tail(5))
    stocks_list.append(stock)

stocks_df = pd.concat(stocks_list, axis=1)
stocks_df.dropna(how="any", axis=0, inplace=True)

# %%% dividends (not used)
# # Define the ticker
# ticker = 'SPY'
# stock = yf.Ticker(ticker)

# # Get historical market data
# hist = stock.history('20y')
# hist.index = hist.index.date
# hist.index = pd.to_datetime(hist.index)

# # Calculate dividend yield time series

# hist['Dividends'].replace(0, np.nan, inplace=True)
# hist['Dividends'] = hist['Dividends'].ffill()

# hist['Dividend Yield'] = hist['Dividends'] / hist['Close'].shift(1)

# # If you want to keep only rows where dividends were paid:
# div_ts = hist['Dividend Yield']
# div_ts = div_ts.resample('M').last()

# print(div_ts)

# %%% dividend implied cost of equity

# g = 0.03
# # div = hist['Dividends'].resample('M').last()
# # div = div.rolling(window=12).sum()

# # price = hist['Close'].resample('M').last()

# # coe = div*(1+g)/price + g
# # coe.name = 'coe'

# # coe.plot()

# div =hist['Dividends'].resample('M').last()
# div = div.rolling(window=12).sum()

# price = hist['Close'].resample('M').mean()

# coe = div*(1+g)/price + g
# coe.name = 'coe'

# coe.plot()


# %%% Macro variable candidates

economic_variables = [
    "GS10",  # 10-Year Treasury Constant Maturity Rate
    "FEDFUNDS",  # Federal Funds Rate
    "CPIAUCSL",  # Consumer Price Index for All Urban Consumers: All Items
    "PPIACO",  # Producer Price Index by Commodity: All Commodities
    "PCEPILFE",  # Personal Consumption Expenditures: Chain-type Price Index: Less Food and Energy
    "UNRATE",  # Civilian Unemployment Rate
    "PAYEMS",  # All Employees: Total Nonfarm Payrolls
    "AHETPI",  # Average Hourly Earnings of All Employees: Total Private
    "GDPC1",  # Real Gross Domestic Product (GDP)
    "INDPRO",  # Industrial Production Index
    "RSXFS",  # Advance Retail Sales: Retail and Food Services
    "HOUST",  # Housing Starts: Total: New Privately Owned Housing Units Started
    "EXHOSLUSM495S",  # Existing Home Sales: Total: Single-family
    "CSUSHPISA",  # S&P/Case-Shiller U.S. National Home Price Index
    "ISMPMI",  # PMI: Manufacturing
    "UMCSENT",  # University of Michigan: Consumer Sentiment
    "CBCCRO1A027NBEA",  # Consumer Confidence Index
    "IPMAN",  # Industrial Production: Manufacturing
    "ISMRSENT",  # ISM: Manufacturing: New Orders Index
    "CUSR0000SA0",  # Real Retail and Food Services Sales
    "CES3000000008",  # Average Weekly Hours of Production and Nonsupervisory Employees: Manufacturing
    "GS1M",  # 1-Month Treasury Constant Maturity Rate
    "GS5",  # 5-Year Treasury Constant Maturity Rate
    "GS20",  # 20-Year Treasury Constant Maturity Rate
    "EXJPUS",  # U.S. / Japan Foreign Exchange Rate
    "EXUSEU",  # U.S. / Euro Foreign Exchange Rate
    "GBTC10",  # 10-Year Breakeven Inflation Rate
    "A939RC0Q052SBEA",  # Real Gross Domestic Product: Total
    "PNFI",  # Federal Debt: Total Public Debt as Percent of Gross Domestic Product
    "M1",  # M1 Money Stock
    "M2",  # M2 Money Stock
    "BOGMBASEW",  # St. Louis Adjusted Monetary Base
    "UMCSENT1M148S",  # University of Michigan: Consumer Sentiment: 1-Month Expectation
    "RETAILMPCSMSA",  # E-commerce Retail Sales
    "RSAFS",  # Advance Retail Sales: Retail Trade
    "ISRATIO",  # Total Business: Inventory to Sales Ratio
    "CPIEALL",  # Consumer Price Index for All Urban Consumers: Energy
    "CUSR0000SAH",  # Real Earnings of Production and Nonsupervisory Employees: Manufacturing
    "EXCSRESNS",  # Crude Oil Prices: West Texas Intermediate (WTI) - Cushing, Oklahoma
    "IR14200",  # Gold Prices
    "APU0000706111",  # Agricultural Prices: All Commodities
    "DTWEXBGS",  # U.S. Dollar Index
]


# %%% create macro data dataframe to combine with SPY adjusted closing

# var = ['SP500','DGS2','DGS10','RSXFS','ICSA','PCE']
# var = ['SP500','DGS2','DGS10','ICSA','PCE']
# var = ['SP500', 'DGS3MO', 'DGS2', 'DGS10', 'PAYEMS', 'PCE']
# var =  ['SP500','DGS3MO','ICSA','PCE']
# var =  ['SP500','DGS2','ICSA','PCE']
# var = ['DGS3MO', 'DGS2', 'DGS10']
# var = ['DGS3MO']
var = ["DGS3MO"]

# var = ['DGS3MO', 'DGS10']

macro = MacroData(var, "m", FRED_API_KEY)
level_df = macro.var_df
level_df.index = level_df.index + MonthEnd(0)
# offset to month end

# level_df['2vs10_spread'] = level_df['DGS10'] - level_df['DGS2']
# level_df = level_df.drop('DGS10', axis=1)
# create spread

level_df = level_df.rename(
    columns={
        "DGS2": "treasury2yr",
        "DGS3MO": "treasury3mo",
        "DGS10": "treasury10yr",
        "RSXFS": "adv_retail_sales",
        "ICSA": "jobless_claims",
        "AAA": "AAA yield",
    }
)


# %%% clean data for missing values and gaps, merge with spy
level_df = level_df[
    :-1
]  # drop latest month data because it only contains partial weeks
level_df.fillna(method="ffill", inplace=True)


level_df = stocks_df.merge(level_df, left_index=True, right_index=True, how="inner")
# level_df = level_df.merge(div_ts, left_index=True, right_index=True, how='inner')
# level_df = level_df.merge(coe, left_index=True, right_index=True, how='inner')

level_df.dropna(inplace=True)

raw_data = (
    level_df.copy()
)  # making a copy of the data as we downloaded it before adjustments


# %%% perform time series analysis on raw data

TimeSeriesAnalysis(level_df).analyze()

# %%% transform data to log returns

"""
pass log_return or log for transformation
"""

transformation_dict = {
    security: "log_return",
    "MTUM": "log_return",
    "GLD": "log_return",
    "treasury3mo": "log_return",
}


# Create a new DataFrame with the same index as level_df
transformed_df = pd.DataFrame(index=level_df.index)

# Apply the specified transformations to each column
for column, transformation_type in transformation_dict.items():
    if transformation_type == "log":
        transformed_df[column] = np.log(level_df[column])
    elif transformation_type == "log_return":
        transformed_df[column] = np.log(level_df[column]) - np.log(
            level_df[column].shift(1)
        )
    else:
        transformed_df[column] = level_df[column]

# Drop rows with NaN values (resulting from the log return calculations)
transformed_df = transformed_df.dropna()


# %%% Adjust for periods

start_date = "2015"

level_df = level_df[start_date:]
log_return_df = transformed_df[start_date:]

scatter_matrix(log_return_df, figsize=(15, 10))


# %%% perform time series analysis on log_transformed data
TimeSeriesAnalysis(log_return_df).analyze()

# %%% perform VAR & AIC grid analysis on trained data


var_analysis.AIC(log_return_df).grid(max_lags=8)
# gen AIC Grid

# %%% Set Model Specs
lags = 1
trend = "n"
alpha = 0.4
train_ratio = 0.8
forecast_steps = 3

# %%%% plot lagged returns in scatter matrix
# Number of lags you have
num_lags = lags  # Change this to the actual number of lags you have

# Create a new DataFrame with all original columns
log_return_lags_df = log_return_df.copy()

# Add lagged log returns as new columns
for asset in log_return_df.columns:
    for lag in range(1, num_lags + 1):
        log_return_lags_df[f"{asset}_lag{lag}"] = log_return_df[asset].shift(lag)

# Print the log_return_lags_df DataFrame
print(log_return_lags_df)

scatter_matrix(log_return_lags_df, figsize=(15, 10))


# %%% In Sample Forecast

# %%% fixed training vs testing
insample = var_analysis.InSample(
    log_return_df, train_ratio=train_ratio, lags=lags, trend=trend
)
insample.run_analysis()
# in sample forecasts

# %%% Dynamic Rolling one step Forecast

# %%%% determin appropriate interval and lag


# Define the range for lag
lag_range = range(1, 4)

# Define the range for interval
interval_range = range(30, 101)

# Initialize variables to store the best combination
best_interval = 0
best_lag = 0
lowest_rmse = float("inf")

# Initialize a dictionary to store RMSE values for each combination
rmse_values = {}

# Loop through intervals and lags
for interval in interval_range:
    for lag in lag_range:
        # Create RollingForecast object with current interval and lag
        back_test = var_analysis.RollingForecast(
            log_return_df,
            lags=lag,
            interval=interval,
            trend=trend,
            conf_alpha=alpha,
            frequency="M",
            price_df=level_df,
        )

        # Calculate RMSE for the current combination
        rmse = back_test.fitness().loc[security, "RMSE"]

        # Check if there are any NaN or negative values in rmse
        if np.isnan(rmse) or rmse < 0:
            print(
                f"Warning: Invalid RMSE value for interval {interval} and lag {lag}: {rmse}"
            )

        # Store RMSE in the dictionary
        rmse_values[(interval, lag)] = rmse

        # Check if the current RMSE is lower than the lowest RMSE found so far
        if rmse < lowest_rmse:
            lowest_rmse = rmse
            best_interval = interval
            best_lag = lag


# Create a DataFrame from the rmse_values dictionary
rmse_df = pd.DataFrame.from_dict(rmse_values, orient="index", columns=["RMSE"])

# Reset the index to have 'interval' and 'lag' as columns
rmse_df.reset_index(inplace=True)
rmse_df.rename(columns={"index": "Interval_Lag"}, inplace=True)

# Split the 'Interval_Lag' column into separate 'Interval' and 'Lag' columns
rmse_df[["Interval", "Lag"]] = pd.DataFrame(
    rmse_df["Interval_Lag"].tolist(), index=rmse_df.index
)
rmse_df.drop(columns="Interval_Lag", inplace=True)

# Pivot the DataFrame to have 'Lag' as columns and 'Interval' as the index
rmse_interval_lag_df = rmse_df.pivot(index="Interval", columns="Lag", values="RMSE")

# Display the resulting DataFrame
print(rmse_interval_lag_df)


# Create a 3D plot of RMSE results
fig = plt.figure(figsize=(15, 12))
ax = fig.add_subplot(111, projection="3d")

intervals, lags = zip(*rmse_values.keys())
rmse_values_list = list(rmse_values.values())

ax.scatter(intervals, lags, rmse_values_list, c=rmse_values_list, cmap="viridis")

ax.set_xlabel("Interval")
ax.set_ylabel("Lag")
ax.set_zlabel("RMSE")
ax.set_title("RMSE vs. Interval and Lag")

# Set tick marks for x and y axes to equal integers
ax.set_xticks(np.arange(min(intervals), max(intervals) + 1, 1))
ax.set_yticks(np.arange(min(lags), max(lags) + 1, 1))

plt.show()

# Print the best combination and RMSE
print(f"Best interval: {best_interval}, Best lag: {best_lag} with RMSE: {lowest_rmse}")

# %%%% perform rolling forecast with chosen interval

lags = 1
interval = 97

# create backtest object
back_test = var_analysis.RollingForecast(
    log_return_df,
    lags=lags,
    interval=interval,
    trend=trend,
    conf_alpha=alpha,
    frequency="M",
    price_df=level_df,
)


# backtest on log returns
backtest_fitness = back_test.fitness()
back_test_forecast = back_test.backtest_by_variable(security)
print("---dynamic forecasting (returns)---")
back_test.plot_backtest(variable=security)
print("log returns : fitness")
print(backtest_fitness)
print("")
print("standard deviation of log returns")
print(log_return_df.std())

# convert log return backtest results to price level

print("---convert to price---")
price_backtest_fitness = back_test.lvl_fitness(security)
back_test.plot_lvl_backtest(security)
print("Price returns : fitness")
print(price_backtest_fitness)
price_backtest = back_test.lvl_backtest_by_variable(security)
print(price_backtest)


# %%% Out of Sample period forecast


# %%%% adjust dates


# Number of months to go back
months_to_go_back = 100  # Change this value as needed

# Calculate the start date by subtracting the specified number of months from the current date
current_date = datetime.now()
start_date = current_date - timedelta(days=months_to_go_back * 30)

# Make sure the start date is the end of the month
start_date = start_date.replace(day=1) - timedelta(days=1)

# Convert start_date to a string in the 'YYYY-MM-DD' format
start_date = start_date.strftime("%Y-%m-%d")

print(f"Start Date: {start_date}")

level_df_adj = level_df[start_date:]
log_return_df_adj = log_return_df[start_date:]

# %%%% forecast

forecast = var_analysis.OutSample(
    log_return_df_adj, lags=lags, steps=forecast_steps, trend=trend
)
forecast.run_analysis(alpha=alpha)
# out of sample forecasts

forecast.params_df()

forecast_lr = forecast.forecast_df
high_lr = forecast.forecast_high(alpha=alpha)
low_lr = forecast.forecast_low(alpha=alpha)
irf_df = forecast.irf_df()


# %%%% forcasted logr with constant alpha


"""
the growth factors confidence interval at a given alpha at a forecasted period is the prior period's forecasted growth factor
multiplied by the current periods growth factor at the confidence interval @ alpha
the exercise here is to arrive at the cumulative growth rates for a constant alpha through the periods and be able to conver this back
to price level
"""
security


lr_interval_df = forecast_lr[[f"{security}_forecast"]].copy()
lr_interval_df[f"{security}_cum_returns"] = lr_interval_df[
    f"{security}_forecast"
].cumsum()
lr_interval_df[f"{security}_high_cum_returns"] = high_lr[
    f"{security}"
].values + lr_interval_df[f"{security}_cum_returns"].shift(1).fillna(0)
lr_interval_df[f"{security}_low_cum_returns"] = low_lr[
    f"{security}"
].values + lr_interval_df[f"{security}_cum_returns"].shift(1).fillna(0)


# %%% converting forecasted log returns to price levels

price_forecasts = logr(forecast_lr, raw_data).get_levels()
return_forecasts = logr(forecast_lr, raw_data).get_returns() - 1
high_price_forecasts = (
    np.exp(lr_interval_df[f"{security}_high_cum_returns"])
    * raw_data.iloc[-1, :][f"{security}"]
)
high_price_forecasts.name = f"{security}_high_price"
low_price_forecasts = (
    np.exp(lr_interval_df[f"{security}_low_cum_returns"])
    * raw_data.iloc[-1, :][f"{security}"]
)
low_price_forecasts.name = f"{security}_low_price"


conf_df = pd.concat(
    [
        price_forecasts[f"{security}_forecast"],
        high_price_forecasts,
        low_price_forecasts,
    ],
    axis=1,
)

last_date = raw_data.index[-1]

# Convert the index of conf_df into a DatetimeIndex with monthly intervals starting from last_date + 1 month
date_range = pd.date_range(
    start=last_date + pd.DateOffset(months=1), periods=len(conf_df), freq="M"
)

# Set the DatetimeIndex as the new index of your DataFrame
conf_df.index = date_range
return_forecasts.index = date_range

# Define the columns you want to plot
columns_to_plot = [
    f"{security}_forecast",
    f"{security}_high_price",
    f"{security}_low_price",
]

# Plot the data
ax = conf_df.plot(figsize=(12, 6))
plt.title(f"{security} Forecast, High Price, and Low Price Over Time")
plt.xlabel("Date")
plt.ylabel("Price")
plt.grid(True)

plt.show()

print("-----------------------------------------------------------------------------")
print(" log return forecasts")
print(return_forecasts)

print("-----------------------------------------------------------------------------")

print(conf_df)

print("-----------------------------------------------------------------------------")
print("Impluse Response")
irf_1 = irf_df[1]
print("Impulse Response period1")
print(irf_1)

irf_2 = irf_df[2]
print("Impulse Response period2")
print(irf_2)

irf_3 = irf_df[3]
print("Impulse Response period3")
print(irf_3)


# %%% Daily opening adjusted forecasts

# # Download daily open prices for the given security
# security = security
# daily_open = yf.download(security, interval='1d')[['Open']]
# daily_open.index = pd.to_datetime(daily_open.index)

# # Get the current date
# current_date = pd.Timestamp.now()
# current_month = current_date.month
# current_year = current_date.year

# # Filter the DataFrame for the current month and year
# current_month_open = daily_open[(daily_open.index.month == current_month) & (daily_open.index.year == current_year)]

# # Create a DataFrame for the adjusted forecast
# daily_adjusted_month_end_forecast = current_month_open.copy()

# # Retrieve the current month end forecasts
# current_month_end_forecast = price_forecasts.iloc[0][f'{security}_forecast']
# current_month_end_forecast_low = low_price_forecasts.iloc[0][security]
# current_month_end_forecast_high = high_price_forecasts.iloc[0][security]

# # Add the forecasts to the DataFrame
# daily_adjusted_month_end_forecast[f'MonthEnd_{security}_forecast'] = current_month_end_forecast
# daily_adjusted_month_end_forecast[f'MonthEnd_{security}_forecast_low'] = current_month_end_forecast_low
# daily_adjusted_month_end_forecast[f'MonthEnd_{security}_forecast_high'] = current_month_end_forecast_high

# # Calculate the number of days past, days in the month, and days remaining in the month
# daily_adjusted_month_end_forecast['Days_past_in_month'] = daily_adjusted_month_end_forecast.index.day
# daily_adjusted_month_end_forecast['Days_in_month'] = daily_adjusted_month_end_forecast.index.days_in_month
# daily_adjusted_month_end_forecast['Days_remaining_in_month'] = daily_adjusted_month_end_forecast['Days_in_month'] - daily_adjusted_month_end_forecast['Days_past_in_month']

# # Calculate the weighted average forecast
# days_past = daily_adjusted_month_end_forecast['Days_past_in_month']
# days_in_month = daily_adjusted_month_end_forecast['Days_in_month']
# days_remaining = daily_adjusted_month_end_forecast['Days_remaining_in_month']
# current_open = daily_adjusted_month_end_forecast['Open']

# weighted_average_forecast = (
#     (days_past / days_in_month) * current_open +
#     (days_remaining / days_in_month) * current_month_end_forecast
# )

# weighted_average_forecast_low = (
#     (days_past / days_in_month) * current_open +
#     (days_remaining / days_in_month) * current_month_end_forecast_low
# )

# weighted_average_forecast_high = (
#     (days_past / days_in_month) * current_open +
#     (days_remaining / days_in_month) * current_month_end_forecast_high
# )

# # Add the adjusted forecasts to the DataFrame
# daily_adjusted_month_end_forecast[f'{security}_adj_forecast'] = weighted_average_forecast
# daily_adjusted_month_end_forecast[f'{security}_adj_low'] = weighted_average_forecast_low
# daily_adjusted_month_end_forecast[f'{security}_adj_high'] = weighted_average_forecast_high

# daily_adjusted_month_end_forecast = daily_adjusted_month_end_forecast.rename(columns={'Open':f'{security}_open'})

# # Select and plot the adjusted forecasts
# daily_adjusted_month_end_forecast = daily_adjusted_month_end_forecast[[f'{security}_open',f'{security}_adj_forecast', f'{security}_adj_low', f'{security}_adj_high']]
# daily_adjusted_month_end_forecast.plot()
# plt.show()

# # Print the adjusted forecasts
# print(f"Current Month's daily open adjusted forecast for {security}:")
# print(daily_adjusted_month_end_forecast)

# %%% BackTest w Trading Strategy


# %%%% Trading Strategy

# %%%%% indicators

# indicators_df = price_backtest
# security = security

# indicators_df[f'{security}_expected_returns'] = indicators_df[f'{security}_forecast']/indicators_df[f'{security}'].shift(1)-1
# indicators_df[f'{security}_actual_returns'] = indicators_df[f'{security}']/indicators_df[f'{security}'].shift(1)-1

# # trailing_vol_log = np.log(raw_data[security]/raw_data[security].shift(1)).rolling(window=12).std().dropna()
# # trailing_vol_simple = np.exp(trailing_vol_log)-1
# # trailing_vol_simple.name = f'{security}_trailing_vol'
# # indicators_df = indicators_df.merge(trailing_vol_simple, left_index=True, right_index=True, how='left')

# # decision nodes
# up = indicators_df[f'{security}_expected_returns'] > 0
# down = indicators_df[f'{security}_expected_returns'] < 0

# # extended =

# %%%% data for backtesting

stock = yf.Ticker(security)
# Get historical market data
hist = stock.history("20y")
hist.index = hist.index.date
hist.index = pd.to_datetime(hist.index)


hist.columns = [security + "_" + col if col != "Date" else col for col in hist.columns]

"""
merge monthly indicators with daily data
"""

# Resample 'indicators_df' to daily frequency, forward-fill missing values
daily_price_forecasts = price_backtest.resample("D").bfill()
daily_price_forecasts.columns = [
    "MonthEnd_" + col if col != "Date" else col for col in daily_price_forecasts.columns
]

# Merge 'daily_indicators_df' with 'hist' DataFrame based on the date
hist = pd.merge(
    hist, daily_price_forecasts, left_index=True, right_index=True, how="left"
)
hist = hist.dropna()

# %%%% apply view
margin = 0.005

# Replace 'MonthEnd_SPY_adj_forecast' with dynamic column names based on the 'security' variable
forecast_column = f"MonthEnd_{security}_adj_forecast"
close_column = f"{security}_Close"

# Define the margin
margin = 0.005

# Calculate the "days past in month" and "days remaining in month"
hist["Days_past_in_month"] = hist.groupby(hist.index.to_period("M")).cumcount()
hist["Days_in_month"] = hist.groupby(hist.index.to_period("M")).transform("count")[
    "Days_past_in_month"
]
hist["Days_remaining_in_month"] = hist["Days_in_month"] - hist["Days_past_in_month"]

# Calculate the weighted average forecast
hist[f"MonthEnd_{security}_adj_forecast"] = (
    hist["Days_past_in_month"] / hist["Days_in_month"]
) * hist[f"{security}_Open"] + (
    hist["Days_remaining_in_month"] / hist["Days_in_month"]
) * hist[f"MonthEnd_{security}_forecast"]

# Calculate the 'view' column based on the given conditions
hist["view"] = hist.apply(
    lambda row: (
        "bull"
        if row[f"MonthEnd_{security}_adj_forecast"] / row[f"{security}_Open"] - 1
        > margin
        else (
            "bear"
            if row[f"MonthEnd_{security}_adj_forecast"] / row[f"{security}_Open"] - 1
            < -margin
            else "flat"
        )
    ),
    axis=1,
)

# Export the DataFrame to Excel
hist.to_excel("data_for_backtesting.xlsx")
print(hist)

# %%%% plot views
forecast_column = f"MonthEnd_{security}_adj_forecast"
close_column = f"{security}_Close"

# Define your custom start and end dates (modify as needed)
start_date = "2022-06-01"
end_date = "2024"

# Filter the DataFrame based on the custom date range
subset_hist = hist[(hist.index >= start_date) & (hist.index <= end_date)]

# Ensure the index is in datetime format
subset_hist.index = pd.to_datetime(subset_hist.index)

# Create the plot
plt.figure(figsize=(10, 6))
plt.plot(
    subset_hist.index,
    subset_hist[forecast_column],
    label=f"{forecast_column}",
    linestyle="-",
    marker="o",
)
plt.plot(
    subset_hist.index,
    subset_hist[close_column],
    label=f"{close_column}",
    linestyle="-",
    marker="o",
)
plt.xlabel("Date")
plt.ylabel("Value")
plt.title(f"{forecast_column} vs. {close_column}")
plt.legend()
plt.grid(True)

# Create a custom date locator and formatter for the x-axis
ax = plt.gca()
ax.xaxis.set_major_locator(MonthLocator(interval=1))
ax.xaxis.set_major_formatter(DateFormatter("%b %Y"))

plt.xticks(rotation=45)  # Rotate x-axis labels for better readability

plt.show()


# %%% forecast & Actual data export to Excel

# %%%% creating dataframes to export

insample_forecasts = insample.forecast().merge(
    log_return_df, left_index=True, right_index=True, how="right"
)
insample_forecasts.sort_index(axis=1, inplace=True)
insample_fitness = insample.fitness_df()
forecast_params = forecast.params_df()

# %%%% ExcelWriter

# """
# uncomment to export analysis to Excel
# """

# out = pd.ExcelWriter('Monthly_stock_momentum_VAR.xlsx')

# level_df.to_excel(out, 'actuals')
# level_df_adj.to_excel(out, 'actuals_for_forecasts')
# insample_forecasts.to_excel(out, 'test_vs_train')
# insample_fitness.to_excel(out, 'insample_fitness')
# forecast_params.to_excel(out, 'outsample_forecast_params')
# back_test_forecast.to_excel(out,'rolling forecast results')
# price_backtest_fitness.to_excel(out, 'rolling forecast fitness')
# price_backtest.to_excel(out, 'rolling forecast price')
# return_forecasts.to_excel(out, 'return_forecasts')
# high_lr.to_excel(out, 'returns_high_end')
# low_lr.to_excel(out, 'returns_low_end')
# conf_df.to_excel(out, 'price_forecasts')
# irf_1.to_excel(out, 'impluse response 1 period')
# irf_2.to_excel(out, 'impluse response 2 period')
# irf_3.to_excel(out, 'impluse response 3 period')

# out.close()
