# -*- coding: utf-8 -*-
"""
Created on Sun Sep 17 16:30:50 2023

@author: rober
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import statsmodels.graphics.tsaplots as tsaplots
from statsmodels.tsa.stattools import adfuller


import quandl as quandl

# %%% class for data prep


class TimeSeriesAnalysis:
    """
    A class to perform various analyses on time series data.

    Attributes:
        data (DataFrame): The data to analyze.
        stationarity_results (dict): Dictionary to store stationarity test results.
    """

    def __init__(self, data):
        """
        Initialize with a DataFrame containing time series data.

        Parameters:
            data (DataFrame): The time series data to analyze.
        """
        if not isinstance(data, pd.DataFrame):
            raise ValueError("Input must be a pandas DataFrame.")

        self.data = data
        self.stationarity_results = {}

    def plot_data(self, figsize=(10, 8), hspace=0.4):
        """
        Plot each column in the DataFrame as a separate subplot.

        Parameters:
            figsize (tuple): Figure size. Default is (10, 8).
            hspace (float): Height space between subplots. Default is 0.4.
        """
        fig, axes = plt.subplots(nrows=len(self.data.columns), ncols=1, figsize=figsize)
        for i, col in enumerate(self.data.columns):
            self.data[col].plot(ax=axes[i])
            axes[i].set_title(col)
        plt.subplots_adjust(hspace=hspace)
        plt.show()

    def test_stationarity(self, timeseries):
        """
        Perform an Augmented Dickey-Fuller test on a given series.

        Parameters:
            timeseries (Series): The time series data to test.

        Returns:
            bool: True if the series is stationary, False otherwise.
        """
        dftest = adfuller(timeseries, autolag="AIC")
        return dftest[1] <= 0.05

    def check_all_stationarity(self):
        """
        Check for stationarity for all columns in the DataFrame.
        Store the results in the stationarity_results attribute and print them.
        """
        self.stationarity_results = {
            col: self.test_stationarity(self.data[col]) for col in self.data.columns
        }
        print("Stationarity results:")
        for col, is_stationary in self.stationarity_results.items():
            print(f"{col}: {'Stationary' if is_stationary else 'Non-Stationary'}")

    def plot_acf_pacf(self):
        """
        Plot AutoCorrelation Function (ACF) and Partial AutoCorrelation Function (PACF)
        for each time series in the DataFrame.
        """
        for col in self.data.columns:
            series = self.data[col]
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))
            tsaplots.plot_acf(series, ax=ax1, zero=False)
            tsaplots.plot_pacf(series, ax=ax2, zero=False)
            ax1.set_title(f"ACF Plot for {col}")
            ax2.set_title(f"PACF Plot for {col}")
            plt.tight_layout()
            plt.show()

    def plot_rolling_correlations(self, window_size=30):
        """
        Calculate and plot rolling correlations with a specified window size.

        Parameters:
            window_size (int): The window size for rolling correlations. Default is 30.
        """
        rolling_corr = self.data.rolling(window=window_size).corr().dropna()
        first_col_name = self.data.columns[0]

        fig, ax = plt.subplots(figsize=(10, 6))
        rolling_corr.iloc[:, 0].unstack().plot(ax=ax, label="Rolling Correlation")
        ax.legend()
        ax.set_title(f"Rolling Correlation of {first_col_name} with other columns")
        plt.show()

    def analyze(self):
        """
        Perform all analyses: plotting data, checking for stationarity,
        and plotting ACF and PACF.
        """
        self.plot_data()
        self.plot_rolling_correlations()
        self.check_all_stationarity()
        self.plot_acf_pacf()


class logr:
    """
    Handle logarithmic returns and associated price data to compute growth levels.

    Attributes:
    -----------
    logr_df : DataFrame
        The DataFrame containing the logarithmic returns.
    price_df : DataFrame
        The DataFrame containing the price data.

    Methods:
    --------
    get_returns():
        Compute and return the simple returns from logarithmic returns.

    get_level():
        Compute and return the growth level data by considering the last price point.
    """

    def __init__(self, logr_df, price_df):
        """
        Initialize the logr class with logarithmic returns and price data.

        Parameters:
        -----------
        logr_df : DataFrame
            The DataFrame containing the logarithmic returns.
        price_df : DataFrame
            The DataFrame containing the price data.
        """
        self.logr_df = logr_df
        self.price_df = price_df

    def get_returns(self):
        """
        Compute and return the simple returns derived from the provided logarithmic returns.

        Returns:
        --------
        DataFrame:
            The simple returns corresponding to the logarithmic returns.
        """
        return np.exp(self.logr_df)

    def get_levels(self):
        """
        Compute the cumulative growth using the logarithmic returns. Multiply each row by the last row
        of the price data to obtain the level data.

        Returns:
        --------
        DataFrame:
            The growth level data for each time point.
        """
        growth_factor = np.exp(self.logr_df)
        cum_growth = growth_factor.cumprod()
        last_price = self.price_df.iloc[-1].values

        return cum_growth.mul(last_price)
