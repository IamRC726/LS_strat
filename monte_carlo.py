# -*- coding: utf-8 -*-
"""
Created on Sat Mar  2 21:15:42 2024

@author: rober
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# %%% Geometric Brownian Motion


class gbm:
    def __init__(
        self,
        mu: float,
        years: int,
        trials: int,
        sigma: float,
        steps: int = 252,
        beg_val: float = 100,
    ):
        """
        Initialize the GBM model with given parameters.

        :param mu: Annual drift coefficient.
        :param years: Total simulation time in years.
        :param trials: Number of simulation trials.
        :param sigma: Annual volatility.
        :param steps: Number of steps per year.
        :param beg_val: Initial stock price.
        """
        self.mu = mu
        self.steps = steps
        self.years = years
        self.trials = trials
        self.sigma = sigma
        self.beg_val = beg_val
        self.dt = self.years / self.steps
        self.paths = self.paths()
        self.end_val = self.paths[-1, :]
        self.cagr = self.cagr()
        self.percentiles_df = self.percentiles()

    def paths(self) -> np.ndarray:
        """
        Generate stock price paths using Geometric Brownian Motion.

        :return: A numpy array of shape (trials, steps+1) containing simulated stock price paths.
        """
        growth = np.exp(
            (self.mu - self.sigma**2 / 2) * self.dt
            + self.sigma
            * np.random.normal(0, np.sqrt(self.dt), size=(self.trials, self.steps)).T
        )
        growth = np.vstack([np.ones(self.trials), growth])
        paths = self.beg_val * growth.cumprod(axis=0)
        return paths

    def cagr(self) -> np.ndarray:
        """
        Calculate the Compound Annual Growth Rate (CAGR) for each simulation path.

        :return: A numpy array containing the CAGR for each path.
        """
        cagr = (self.end_val / self.beg_val) ** (1 / self.years) - 1
        return cagr

    def percentiles(self, percentiles=[5, 20, 50, 80, 95]):
        """
        Calculate specified percentiles of CAGR.

        :param percentiles: A list of percentiles to calculate.
        :return: A pandas DataFrame with the specified percentiles of CAGR.
        """
        percentile_values = np.percentile(self.cagr, percentiles)
        percentile_df = pd.DataFrame(
            percentile_values, index=percentiles, columns=["CAGR"]
        )
        percentile_df.index = [f"{p}%" for p in percentiles]
        percentile_df.index.name = "Percentile"
        return percentile_df

    def plot_distribution(self) -> None:
        """
        Plot the probability density of CAGR.
        """
        sns.set(style="whitegrid")
        plt.figure(figsize=(10, 6))
        ax = sns.kdeplot(self.cagr, bw_adjust=0.5, color="black")
        plt.title("Probability Density of CAGR", color="black")
        plt.xlabel("Percentile", color="blue")

        # Extract the percentile CAGR values for setting x-ticks
        percentile_cagr_values = self.percentiles_df["CAGR"].values
        # Extract the percentile labels for x-ticks
        percentile_labels = [f"{p}" for p in self.percentiles_df.index]

        # Set custom ticks on the x-axis at the percentile CAGR values
        ax.set_xticks(percentile_cagr_values)
        # Label these x-ticks with the corresponding percentile labels
        ax.set_xticklabels(percentile_labels)

        ylim = plt.ylim()
        # Draw vertical lines and label them with the CAGR values
        for value in percentile_cagr_values:
            ax.axvline(x=value, color="red", linestyle="--")
            plt.text(
                value,
                ylim[1] * 0.50,
                f"{value*100:.2f}%",
                color="black",
                ha="center",
                fontsize=14,
                rotation=45,
            )

        plt.ylim(ylim[0], ylim[1])
        plt.show()

    def plot_paths(self) -> None:
        time = np.linspace(0, self.years, self.steps + 1)
        plt.figure(figsize=(10, 6))
        plt.plot(time, self.paths, linewidth=0.5, alpha=0.4, color="gray")

        plt.xlabel("Years $(t)$", fontsize=14)
        plt.ylabel("value $(S_t)$", fontsize=14)
        plt.title(
            f"Realizations of Geometric Brownian Motion\n$dS_t = \mu S_t dt + \sigma S_t dW_t$\n$S_0 = {self.beg_val}, \mu = {self.mu}, \sigma = {self.sigma}$",
            fontsize=14,
        )
        plt.tight_layout()
        plt.show()


# %%% example

# gbm_obj = gbm(0.10, 3, 50000, 0.17)

# cagr = gbm_obj.cagr
# paths = gbm_obj.paths
# percentile_df = gbm_obj.percentiles_df
# gbm_obj.plot_paths()
# gbm_obj.plot_distribution()
