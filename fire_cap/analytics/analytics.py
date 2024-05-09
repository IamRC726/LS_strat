from abc import ABC, abstractmethod
from pydantic import BaseModel

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from pypfopt import (
    BlackLittermanModel,
    black_litterman,
    EfficientFrontier,
    plotting,
    risk_models,
    objective_functions,
)

from environ import env


class Input(BaseModel, arbitrary_types_allowed=True):
    risk_free_rate: float
    capm_return: any
    sample_cov: any
    df_prices: any
    mkt_price: any
    mcap: any


def e_f(input: Input):
    return EfficientFrontier(input.capm_return, input.sample_cov)


class Analytics(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def exec(self, input: Input):
        pass

    @abstractmethod
    def excel_frames(self):
        pass


class OptWeightPerformance(Analytics):
    cleaned_weights: any = None
    performance: any = None

    def exec(self, input: Input):
        ef = e_f(input)
        ef.max_sharpe(risk_free_rate=input.risk_free_rate)
        cleaned_weights = ef.clean_weights()
        self.cleaned_weights = pd.DataFrame(
            list(cleaned_weights.items()), columns=["Ticker", "Weight"]
        )

        print(self.cleaned_weights)
        (
            expected_annual_return,
            annual_volatility,
            sharpe_ratio,
        ) = ef.portfolio_performance(verbose=True)

        self.performance = pd.DataFrame(
            {
                "Metric": [
                    "Expected Annual Return",
                    "Annual Volatility",
                    "Sharpe Ratio",
                ],
                "Value": [expected_annual_return, annual_volatility, sharpe_ratio],
            }
        )
        print(self.performance)

    def excel_frames(self):
        return [
            {
                "df": self.cleaned_weights,
                "sheet_name": "optport_weights",
            },
            {
                "df": self.performance,
                "sheet_name": "port_performance",
            },
        ]


class OptResults(Analytics):
    data_frame: any

    def exec(self, input: Input):
        ef = e_f(input)

        # Get the minimum and maximum volatility for the portfolios on the frontier
        ef.min_volatility()
        min_volatility = ef.portfolio_performance(verbose=False)[
            1
        ]  # [1] is the volatility

        max_volatility = np.sqrt(
            np.max(np.diag(input.sample_cov))
        )  # A rough estimate of max volatility

        # Generate a range of target volatilities
        volatility_range = np.linspace(min_volatility, max_volatility, 20)

        # Dictionary to hold the optimized weights and performance metrics for each target volatility
        portfolio_details = {}

        for target_vol in volatility_range:
            ef = e_f(input)
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
        assets = (
            ef.tickers
        )  # Assuming ef.tickers contains your list of tickers. Adjust as necessary.
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

        self.data_frame = pd.DataFrame(data)

    def excel_frames(self):
        return [
            {
                "df": self.data_frame,
                "sheet_name": "optport_results",
            },
        ]


class VolatilityEFDiagram(Analytics):
    def exec(self, input: Input):
        ef = e_f(input)
        # Create the efficient frontier plot
        fig, ax = plt.subplots()
        plotting.plot_efficient_frontier(
            ef, ax=ax, show_assets=True
        )  # Temporarily disable showing assets

        # Calculate risk (volatility) for each asset
        asset_risks = np.sqrt(np.diag(input.sample_cov))

        # Add text annotation for each asset
        df_prices = input.df_prices
        for ticker in df_prices.columns:
            volatility = asset_risks[
                df_prices.columns.get_loc(ticker)
            ]  # Get volatility for the asset
            expected_return = input.capm_return[
                ticker
            ]  # Get expected return for the asset
            ax.scatter(
                volatility, expected_return, marker="o", color="red"
            )  # Mark the asset on the plot
            ax.text(volatility, expected_return, ticker, fontsize=9)  # Label the asset

        # Save and show the plot
        plt.savefig("out/efficient_frontier_with_labels.png")

    def excel_frames(self):
        return []


class BLImpliedDiagram(Analytics):
    def exec(self, input: Input):
        shrink = risk_models.CovarianceShrinkage(input.df_prices).ledoit_wolf()
        delta = black_litterman.market_implied_risk_aversion(
            input.mkt_price[env.CONFIG.market_price_stk]
        )
        print(delta)
        plotting.plot_covariance(shrink, plot_correlation=True)

        market_prior = black_litterman.market_implied_prior_returns(
            input.mcap, delta, shrink
        )
        print(market_prior)
        market_prior.plot.barh(figsize=(10, 5))

    def excel_frames(self):
        return []


class BLForecastDiagram(Analytics):
    def view_dict(self, input):
        return {
            "IYE": 0.10,
            "IYR": 0.15,
            "IYM": 0.20,
            "IYJ": 0.05,
            "IYC": 0.20,
            "IYK": 0.10,
            "IYH": 0.2,
            "IYF": 0.15,
            "IYW": 0.04,
            "IXP": 0.05,
            "IDU": 0.05,
        }

    def intervals(self, input):
        return [
            (0, 0.25),
            (0.1, 0.4),
            (-0.1, 0.15),
            (-0.05, 0.1),
            (0.15, 0.25),
            (-0.1, 0),
            (0.1, 0.2),
            (0.08, 0.12),
            (0.1, 0.9),
            (0, 0.3),
            (0, 0.3),
        ]

    def exec(self, input: Input):
        shrink = risk_models.CovarianceShrinkage(input.df_prices).ledoit_wolf()
        delta = black_litterman.market_implied_risk_aversion(
            input.mkt_price[env.CONFIG.market_price_stk]
        )
        market_prior = black_litterman.market_implied_prior_returns(
            input.mcap, delta, shrink
        )

        variances = []
        intervals = self.intervals(input)
        viewdict = self.view_dict(input)
        for lb, ub in intervals:
            sigma = (ub - lb) / 2
            variances.append(sigma**2)

        omega = np.diag(variances)
        bl = BlackLittermanModel(
            shrink,
            pi="market",
            market_caps=input.mcap,
            risk_aversion=delta,
            absolute_views=viewdict,
            omega=omega,
        )

        # Posterior estimate of returns
        ret_bl = bl.bl_returns()
        rets_df = pd.DataFrame(
            [
                market_prior,
                ret_bl,
                pd.Series(viewdict),
            ],
            index=["Prior", "Posterior", "Views"],
        ).T
        rets_df.plot.bar(figsize=(12, 8))
        S_bl = bl.bl_cov()
        plotting.plot_covariance(S_bl)

        ef = EfficientFrontier(ret_bl, S_bl)
        ef.add_objective(objective_functions.L2_reg)
        ef.max_sharpe()
        weights = ef.clean_weights()
        print(weights)

        pd.Series(weights).plot.pie(figsize=(10, 10))

    def excel_frames(self):
        return []
