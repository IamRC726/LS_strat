# -*- coding: utf-8 -*-
"""
Created on Sun Sep 17 17:32:35 2023

@author: rober
"""
from sklearn.metrics import mean_squared_error
from math import sqrt
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from statsmodels.tsa.api import VAR


class AIC:
    def __init__(self, df, trend="c"):
        """
        - trend : str, optional (default='c')
        - 'c'   : Include a constant (intercept) in the model.
        - 'ct'  : Include a constant and a linear time trend.
        - 'ctt' : Include a constant, a linear time trend, and a quadratic time trend.
        - 'nc'  : No constant or trend (model is trend-free).
        """
        self.df = df
        self.trend = trend

    def grid(self, max_lags):
        self.max_lags = max_lags
        model = VAR(self.df)
        aic = []
        for i in range(1, self.max_lags + 1):
            result = model.fit(maxlags=i, trend=self.trend)
            aic.append(result.aic)
        plt.plot(list(range(1, self.max_lags + 1)), aic)
        plt.title("Choosing Number of Lags")
        plt.xlabel("Lags")
        plt.ylabel("AIC")
        plt.show()


class InSample:
    def __init__(self, df, train_ratio, lags, trend="c"):
        """
        - trend : str, optional (default='c')
        - 'c'   : Include a constant (intercept) in the model.
        - 'ct'  : Include a constant and a linear time trend.
        - 'ctt' : Include a constant, a linear time trend, and a quadratic time trend.
        - 'nc'  : No constant or trend (model is trend-free).
        """

        self.df = df
        self.train_ratio = train_ratio
        self.lags = lags
        self.trend = trend

    def split_data(self):
        n_obs = int(len(self.df) * self.train_ratio)
        self.train, self.test = self.df[0:n_obs], self.df[n_obs:]

    def plot_series(self):
        self.df.plot(figsize=(10, 6))
        plt.show()

    def fit_model(self):
        model = VAR(self.train)
        self.results = model.fit(maxlags=self.lags, trend=self.trend)

    def get_params(self):
        results_summary = self.results.summary()
        print(results_summary)

    def params_df(self):
        # Parameters & Coefficients
        coefficients = pd.DataFrame(self.results.params)

        # Standard Errors
        std_errors = pd.DataFrame(self.results.stderr)

        # t-values
        t_values = pd.DataFrame(self.results.tvalues)

        # p-values
        p_values = pd.DataFrame(self.results.pvalues)

        # Combine into a single DataFrame
        params_df = pd.concat([coefficients, std_errors, t_values, p_values], axis=1)
        params_df.columns = pd.MultiIndex.from_product(
            [
                ["Coefficient", "Standard Error", "t-value", "p-value"],
                self.results.names,
            ],
            names=["Metric", "Variable"],
        )

        return params_df

    def get_significant_vars(self, crit_val=0.05):
        all_params = pd.DataFrame(self.results.params)
        all_p_val = pd.DataFrame(self.results.pvalues)
        sig_params = all_params.where(all_p_val <= crit_val)
        sig_vars = sig_params.fillna(0).dropna(how="all")
        non_zero_mask = sig_vars != 0
        non_zero_mask = non_zero_mask.reset_index()
        non_zero_mask[["Lag", "Variable"]] = non_zero_mask["index"].str.split(
            ".", expand=True
        )
        sig_var_summary = non_zero_mask.groupby("Lag").sum()
        sig_var_summary["Total"] = sig_var_summary.sum(axis=1)
        print(sig_var_summary)

    def forecast(self):
        lag_order = self.results.k_ar
        forecast_steps = len(self.test)
        forecast_values = self.results.forecast(
            self.train.values[-lag_order:], steps=forecast_steps
        )
        self.forecast_df = pd.DataFrame(
            forecast_values,
            index=self.test.index,
            columns=[f"{col}_forecast" for col in self.test.columns],
        )
        return self.forecast_df

    def plot_forecast(self):
        for col in self.df.columns:
            original = self.df[col]
            forecast = self.forecast_df[col + "_forecast"]
            plt.figure(figsize=(15, 5))
            plt.plot(original, label=f"Original {col}")
            plt.plot(forecast, label=f"Forecast {col}")
            plt.legend()
            plt.show()

    def summary_test_data(self):
        print("Summary of test data")
        print()
        print(self.test.describe())

    def calculate_rmse(self):
        print("Root Mean Square Error")
        for col in self.df.columns:
            original = self.test[col]
            forecast = self.forecast_df[col + "_forecast"].iloc[: len(original)]
            error = sqrt(mean_squared_error(forecast, original))
            print(f"RMSE for {col} is {error:.4f}")

    def calculate_mape(self):
        print("Mean Absolute Percentage Error")
        for col in self.df.columns:
            original = self.test[col]
            forecast = self.forecast_df[col + "_forecast"].iloc[: len(original)]
            error = np.mean(np.abs((original - forecast) / original)) * 100
            print(f"MAPE for {col} is {error:.4f}%")

    def fitness_df(self):
        metrics_data = {"Variable": [], "RMSE": [], "MAPE %s": []}

        for col in self.df.columns:
            original = self.test[col]
            forecast = self.forecast_df[col + "_forecast"].iloc[: len(original)]

            # Calculate RMSE
            rmse_error = sqrt(mean_squared_error(forecast, original))

            # Calculate MAPE
            mape_error = np.mean(np.abs((original - forecast) / original)) * 100

            metrics_data["Variable"].append(col)
            metrics_data["RMSE"].append(rmse_error)
            metrics_data["MAPE %s"].append(mape_error)

        fitness_dataframe = pd.DataFrame(metrics_data)
        fitness_dataframe.set_index("Variable", inplace=True)
        return fitness_dataframe

    def compute_irf(self, steps=10, orth=False):
        self.irf = self.results.irf(steps)
        self.irf_values = self.irf.irfs

    def plot_irf(self, orth=False):
        self.irf.plot(orth=orth, figsize=(20, 15))
        plt.show()

    def irf_df(self):
        dfs = [
            pd.DataFrame(
                self.irf_values[i], columns=self.results.names, index=self.results.names
            )
            for i in range(self.irf_values.shape[0])
        ]
        return dfs

    def run_analysis(self):
        self.split_data()
        self.plot_series()
        self.fit_model()
        try:
            self.get_params()
        except Exception as e:
            print("Unable to retrieve model parameters. Error:", e)
        self.get_significant_vars()
        self.forecast()
        self.plot_forecast()
        self.summary_test_data()
        try:
            self.compute_irf(steps=3)
            self.plot_irf()
            self.irf_df()
        except Exception as e:
            print("Unable to calibrate IRF analysis. Error:", e)

        print(self.fitness_df())


class OutSample:
    conf_alpha = 0.4

    def __init__(self, df, lags, steps, trend="c"):
        """
        - trend : str, optional (default='c')
        - 'c'   : Include a constant (intercept) in the model.
        - 'ct'  : Include a constant and a linear time trend.
        - 'ctt' : Include a constant, a linear time trend, and a quadratic time trend.
        - 'nc'  : No constant or trend (model is trend-free).
        """
        self.df = df
        self.lags = lags
        self.steps = steps
        self.trend = trend

    def plot_series(self):
        self.df.plot(figsize=(10, 6))
        plt.show()

    def fit_model(self):
        model = VAR(self.df)
        self.results = model.fit(self.lags, trend=self.trend)

    def get_params(self):
        results_summary = self.results.summary()
        return results_summary
        print(results_summary)

    def params_df(self):
        # Parameters & Coefficients
        coefficients = pd.DataFrame(self.results.params)

        # Standard Errors
        std_errors = pd.DataFrame(self.results.stderr)

        # t-values
        t_values = pd.DataFrame(self.results.tvalues)

        # p-values
        p_values = pd.DataFrame(self.results.pvalues)

        # Combine into a single DataFrame
        params_df = pd.concat([coefficients, std_errors, t_values, p_values], axis=1)
        params_df.columns = pd.MultiIndex.from_product(
            [
                ["Coefficient", "Standard Error", "t-value", "p-value"],
                self.results.names,
            ],
            names=["Metric", "Variable"],
        )

        return params_df

    def get_significant_vars(self, crit_val=0.05):
        # Same logic as the InSample class
        all_params = pd.DataFrame(self.results.params)
        all_p_val = pd.DataFrame(self.results.pvalues)
        sig_params = all_params.where(all_p_val <= crit_val)
        sig_vars = sig_params.fillna(0).dropna(how="all")
        print(sig_vars)

    def out_of_sample_forecast(self):
        lag_order = self.results.k_ar
        forecast_values = self.results.forecast(
            self.df.values[-lag_order:], steps=self.steps
        )
        self.forecast_df = pd.DataFrame(
            forecast_values, columns=[f"{col}_forecast" for col in self.df.columns]
        )
        return self.forecast_df

    def plot_forecast(self):
        for col in self.df.columns:
            forecast = self.forecast_df[col + "_forecast"]
            plt.figure(figsize=(15, 5))
            plt.plot(forecast, label=f"Out-of-Sample Forecast {col}")
            ax = plt.gca()  # Get current axis
            ax.xaxis.set_major_locator(MaxNLocator(integer=True))
            plt.legend()
            plt.show()

    def compute_irf(self, steps=10, orth=False):
        self.irf = self.results.irf(steps)
        self.irf_values = self.irf.irfs

    def plot_irf(self, orth=False):
        self.irf.plot(orth=orth, figsize=(20, 15))
        # Adjust the x-axis to display integer ticks
        ax = plt.gca()
        ax.set_xticks([int(tick) for tick in ax.get_xticks().tolist()])
        plt.show()

    def irf_df(self):
        dfs = [
            pd.DataFrame(
                self.irf_values[i], columns=self.results.names, index=self.results.names
            )
            for i in range(self.irf_values.shape[0])
        ]
        return dfs

    def forecast_low(self, alpha=None):
        """
        Get the out-of-sample lower forecast confidence intervals.

        Parameters:
        - alpha: significance level for (1 - alpha)100% confidence interval
        """
        if alpha is None:
            alpha = self.conf_alpha

        self.alpha = alpha
        lag_order = self.results.k_ar
        _, lower, _ = self.results.forecast_interval(
            self.df.values[-lag_order:], steps=self.steps, alpha=self.alpha
        )

        low_conf_interval_df = pd.DataFrame(
            lower,
            index=np.arange(len(self.df), len(self.df) + self.steps),
            columns=self.df.columns,
        )

        print(f"low end at {(1-self.alpha)*100} % confidence")
        print(low_conf_interval_df)
        return low_conf_interval_df

    def forecast_high(self, alpha=None):
        """
        Get the out-of-sample upper forecast confidence intervals.

        Parameters:
        - alpha: significance level for (1 - alpha)100% confidence interval
        """
        if alpha is None:
            alpha = self.conf_alpha

        self.alpha = alpha
        lag_order = self.results.k_ar
        _, _, upper = self.results.forecast_interval(
            self.df.values[-lag_order:], steps=self.steps, alpha=self.alpha
        )

        high_conf_interval_df = pd.DataFrame(
            upper,
            index=np.arange(len(self.df), len(self.df) + self.steps),
            columns=self.df.columns,
        )

        print(f"high end at {(1-self.alpha)*100} % confidence")
        print(high_conf_interval_df)
        return high_conf_interval_df

    def run_analysis(self, alpha=conf_alpha):
        self.alpha = alpha
        self.plot_series()
        self.fit_model()
        self.get_params()
        self.get_significant_vars()
        self.out_of_sample_forecast()
        self.plot_forecast()
        try:
            self.compute_irf(steps=3)
            self.plot_irf()
            self.irf_df()
        except:
            print("Unable to calibrate IRF analysis")
        self.forecast_high(self.alpha)
        self.forecast_low(self.alpha)


class RollingForecast(OutSample):
    """
    A class for performing rolling forecasts with VAR models.

    Args:
        df (pd.DataFrame): The time series data.
        lags (int): The number of lags to consider in the VAR model.
        interval (int): The size of the rolling window.
        trend (str, optional): The trend parameter for the VAR model ('c' for constant, 'nc' for no constant). Default is 'c'.
        frequency (str): The frequency of the time series data (e.g., 'D' for daily, 'M' for monthly). Default is None.

    Attributes:
        interval (int): The size of the rolling window.
        trials (int): The number of rolling forecast trials.
        frequency (str, optional): The frequency of the time series data.
    """

    def __init__(
        self,
        df,
        lags,
        interval,
        frequency=None,
        trend="c",
        conf_alpha=0.4,
        price_df=None,
    ):
        super().__init__(df, lags, 1, trend)

        # Ensure interval is an integer
        if not isinstance(interval, int):
            raise ValueError("Interval needs to be an integer")
        self.conf_alpha = conf_alpha
        self.interval = interval
        self.trials = len(self.df.index) - self.interval
        self.frequency = frequency
        self.price_df = price_df

        self.forecast_df = self.forecast()
        self.forecast_low_df = self.forecast_low()
        self.forecast_high_df = self.forecast_high()
        self.fitness_df = self.fitness()
        # instance attributes created by instance methods

    def forecast(self):
        """
        Perform a rolling forecast with VAR models.

        Returns:
            pd.DataFrame: A DataFrame containing the rolling forecasts.
        """

        forecast_df = pd.DataFrame(columns=self.df.columns)

        for i in range(self.trials):
            # Get the rolling window of data
            rolling_df = self.df.iloc[i : i + self.interval]

            # Specify the frequency if provided, otherwise set to None
            if self.frequency:
                rolling_df.index.freq = self.frequency

            # Fit VAR model on the rolling window
            model = VAR(rolling_df)
            results = model.fit(self.lags, trend=self.trend)
            lag_order = results.k_ar

            # Make a one-step forecast
            rolling_forecast_value = results.forecast(
                rolling_df.values[-lag_order:], steps=1
            )

            # Create a DataFrame with the forecasted values and set the index to the original datetime index from self.df
            rolling_forecast_df = pd.DataFrame(
                rolling_forecast_value,
                columns=self.df.columns,
                index=[self.df.index[i + self.interval]],
            )

            # Append the forecast to the result dataframe
            forecast_df = pd.concat([forecast_df, rolling_forecast_df])

        return forecast_df

    def forecast_low(self):
        """
        Get the rolling out-of-sample lower forecast confidence intervals.

        """

        alpha = self.conf_alpha

        forecast_intervals = []

        for i in range(self.trials):
            # Get the rolling window of data
            rolling_df = self.df.iloc[i : i + self.interval]

            # Specify the frequency if provided, otherwise set to None
            if self.frequency:
                rolling_df.index.freq = self.frequency

            # Fit VAR model on the rolling window
            model = VAR(rolling_df)
            results = model.fit(self.lags, trend=self.trend)
            lag_order = results.k_ar

            # Calculate the forecast and confidence interval for the rolling window, result is a tuple of 3 values, we only want the second value
            _, lower, _ = results.forecast_interval(
                rolling_df.values[-lag_order:], steps=self.steps, alpha=alpha
            )

            # Create a DataFrame with the lower confidence interval values and set the index to correspond to the forecasted period
            lower_conf_interval_df = pd.DataFrame(
                lower,
                index=[self.df.index[i + self.interval + j] for j in range(self.steps)],
                columns=self.df.columns,
            )

            # Append the lower confidence interval to the list
            forecast_intervals.append(lower_conf_interval_df)

        # Concatenate the list of lower confidence intervals into a single DataFrame
        low_conf_interval_df_rolling = pd.concat(forecast_intervals)

        return low_conf_interval_df_rolling

    def forecast_high(self):
        """
        Get the rolling out-of-sample higher forecast confidence intervals.

        """

        alpha = self.conf_alpha

        forecast_intervals = []

        for i in range(self.trials):
            # Get the rolling window of data
            rolling_df = self.df.iloc[i : i + self.interval]

            # Specify the frequency if provided, otherwise set to None
            if self.frequency:
                rolling_df.index.freq = self.frequency

            # Fit VAR model on the rolling window
            model = VAR(rolling_df)
            results = model.fit(self.lags, trend=self.trend)
            lag_order = results.k_ar

            # Calculate the forecast and confidence interval for the rolling window, result is a tuple of 3 values, we only want the third value
            _, _, upper = results.forecast_interval(
                rolling_df.values[-lag_order:], steps=self.steps, alpha=alpha
            )

            # Create a DataFrame with the upper confidence interval values and set the index to correspond to the forecasted period
            upper_conf_interval_df = pd.DataFrame(
                upper,
                index=[self.df.index[i + self.interval + j] for j in range(self.steps)],
                columns=self.df.columns,
            )

            # Append the upper confidence interval to the list
            forecast_intervals.append(upper_conf_interval_df)

        # Concatenate the list of upper confidence intervals into a single DataFrame
        high_conf_interval_df_rolling = pd.concat(forecast_intervals)

        return high_conf_interval_df_rolling

    @staticmethod
    def forecast_by_variable(
        variable, actl_df, forecast_df, forecast_low_df, forecast_high_df
    ):
        """
        create DataFrame of a selected variable's actuals, forecasts, and high/low confidence interval

        Parameters
        ----------
        variable : str
            ticker/variable
        actl_df : DataFrame
            the input dataframe
        forecast_df : DataFrame
            From VAR forecast()
        forecast_low_df : DataFrame
            From VAR forecast_low()
        forecast_high_df : DataFrame
            From VAR forecast_high()

        Returns
        -------
        DataFrame of a selected variable's actuals, forecasts, and high/low confidence interval

        """

        variable_df = actl_df[[variable]].merge(
            forecast_df[[variable]].add_suffix("_forecast"),
            left_index=True,
            right_index=True,
            how="inner",
        )
        variable_df = variable_df.merge(
            forecast_low_df[[variable]].add_suffix("_low"),
            left_index=True,
            right_index=True,
            how="inner",
        )
        variable_df = variable_df.merge(
            forecast_high_df[[variable]].add_suffix("_high"),
            left_index=True,
            right_index=True,
            how="inner",
        )

        return variable_df

    @staticmethod
    def plot_forecast(
        variable, actl_df, forecast_df, forecast_low_df, forecast_high_df, conf_alpha
    ):
        """
        plot backtest

        Parameters
        ----------
        variable : str
            ticker/variable
        actl_df : DataFrame
            the input dataframe
        forecast_df : DataFrame
            From VAR forecast()
        forecast_low_df : DataFrame
            From VAR forecast_low()
        forecast_high_df : DataFrame
            From VAR forecast_high()
        conf_alpha : numeric

        Returns
        -------
        None.

        """

        variable_df = RollingForecast.forecast_by_variable(
            variable, actl_df, forecast_df, forecast_low_df, forecast_high_df
        )

        conf_int = 1 - conf_alpha
        plt.figure(figsize=(12, 6))

        # Plot the actual values
        plt.plot(
            variable_df.index,
            variable_df[variable],
            label=f"{variable} Actual",
            color="blue",
        )

        # Plot the forecast with the same color as forecast high/low, and as a solid line
        plt.plot(
            variable_df.index,
            variable_df[f"{variable}_forecast"],
            label=f"{variable} Forecast",
            color="green",
            linestyle="-",
        )

        # Fill the space between the lower and upper confidence intervals with light gray
        plt.fill_between(
            variable_df.index,
            variable_df[f"{variable}_low"],
            variable_df[f"{variable}_high"],
            color="lightgray",
            label=f"{variable} at {int(conf_int*100)}% confidence",
        )  # Include confidence interval level in the legend label

        plt.xlabel("Date")
        plt.ylabel(variable)
        plt.title(f"{variable} Backtest with Forecast and Confidence Intervals")
        plt.legend()
        plt.grid(True)

        plt.show()

    @staticmethod
    def forecast_fitness(actls_df, forecast_df):
        forecast_error_df = forecast_df - actls_df.loc[forecast_df.index, :]
        squared_forecast_erro_df = forecast_error_df**2
        mean_squared_forecast_erro_df = squared_forecast_erro_df.mean()
        rmse_df = np.sqrt(mean_squared_forecast_erro_df)

        pct_error = (forecast_error_df / actls_df.loc[forecast_df.index, :]) * 100
        abs_pct_erro_df = abs(forecast_error_df / actls_df.loc[forecast_df.index, :])
        mape_df = abs_pct_erro_df.mean()

        fitness_df = pd.DataFrame({"RMSE": rmse_df, "MAPE (%)": mape_df})

        return fitness_df

    def backtest_by_variable(self, variable):
        variable = variable
        actl_df = self.df
        forecast_df = self.forecast_df
        forecast_low_df = self.forecast_low_df
        forecast_high_df = self.forecast_high_df

        return RollingForecast.forecast_by_variable(
            variable, actl_df, forecast_df, forecast_low_df, forecast_high_df
        )

    def plot_backtest(self, variable):
        variable = variable
        actl_df = self.df
        forecast_df = self.forecast_df
        forecast_low_df = self.forecast_low_df
        forecast_high_df = self.forecast_high_df
        conf_alpha = self.conf_alpha

        RollingForecast.plot_forecast(
            variable,
            actl_df,
            forecast_df,
            forecast_low_df,
            forecast_high_df,
            conf_alpha,
        )

    def fitness(self):
        actls_df = self.df
        forecast_df = self.forecast_df

        return RollingForecast.forecast_fitness(actls_df, forecast_df)

    def lvl_forecast(self):
        growth_factor_df = np.exp(self.forecast_df)
        lvl_forecast_df = self.price_df.shift(1) * growth_factor_df
        lvl_forecast_df = lvl_forecast_df.dropna(axis=0)

        return lvl_forecast_df

    def lvl_forecast_low(self):
        """
        Convert the lower confidence interval forecast (log returns) to price levels.

        Args:
            price_df (pd.DataFrame): A DataFrame containing the historical price data.

        Returns:
            pd.DataFrame: A DataFrame containing the lower confidence interval forecast in price levels.
        """
        growth_factor_low_df = np.exp(self.forecast_low_df)
        lvl_forecast_low_df = self.price_df.shift(1) * growth_factor_low_df
        lvl_forecast_low_df = lvl_forecast_low_df.dropna(axis=0)

        return lvl_forecast_low_df

    def lvl_forecast_high(self):
        """
        Convert the upper confidence interval forecast (log returns) to price levels.

        Args:
            price_df (pd.DataFrame): A DataFrame containing the historical price data.

        Returns:
            pd.DataFrame: A DataFrame containing the upper confidence interval forecast in price levels.
        """
        growth_factor_high_df = np.exp(self.forecast_high_df)
        lvl_forecast_high_df = self.price_df.shift(1) * growth_factor_high_df
        lvl_forecast_high_df = lvl_forecast_high_df.dropna(axis=0)

        return lvl_forecast_high_df

    def lvl_backtest_by_variable(self, variable):
        variable = variable
        actl_df = self.price_df
        forecast_df = self.lvl_forecast()
        forecast_low_df = self.lvl_forecast_low()
        forecast_high_df = self.lvl_forecast_high()

        return RollingForecast.forecast_by_variable(
            variable, actl_df, forecast_df, forecast_low_df, forecast_high_df
        )

    def plot_lvl_backtest(self, variable):
        variable = variable
        actl_df = self.price_df
        forecast_df = self.lvl_forecast()
        forecast_low_df = self.lvl_forecast_low()
        forecast_high_df = self.lvl_forecast_high()
        conf_alpha = self.conf_alpha

        RollingForecast.plot_forecast(
            variable,
            actl_df,
            forecast_df,
            forecast_low_df,
            forecast_high_df,
            conf_alpha,
        )

    def lvl_fitness(self, variable):
        actls_df = self.price_df
        forecast_df = self.lvl_forecast()

        return RollingForecast.forecast_fitness(actls_df, forecast_df)
