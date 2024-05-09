# -*- coding: utf-8 -*-
"""
Created on Sun Sep 17 15:30:19 2023

@author: rober
"""

import pandas as pd

from fredapi import Fred

# %%% Macro Data Class


class MacroData:
    def __init__(self, variables, freq, key):
        """
        initiate class to downlaod macro data from FRED and perform analysis

        Parameters:
        - variables (list): list of macrovariables avaialbe on Fred

        - freq (str): Data interval ()

        One of the following values: 'd', 'w', 'bw', 'm', 'q', 'sa', 'a', 'wef', 'weth', 'wew', 'wetu', 'wem', 'wesu', 'wesa', 'bwew', 'bwem'
        Frequencies without period descriptions:

        d = Daily
        w = Weekly
        bw = Biweekly
        m = Monthly
        q = Quarterly
        sa = Semiannual
        a = Annual

        Frequencies with period descriptions:

        wef = Weekly, Ending Friday
        weth = Weekly, Ending Thursday
        wew = Weekly, Ending Wednesday
        wetu = Weekly, Ending Tuesday
        wem = Weekly, Ending Monday
        wesu = Weekly, Ending Sunday
        wesa = Weekly, Ending Saturday
        bwew = Biweekly, Ending Wednesday
        bwem = Biweekly, Ending Monday

        - key: FRED api key
        """
        self.variables = variables
        self.freq = freq
        self.key = key
        self.var_df = self.variable_df()

    def variable_df(self):
        frames = {}
        fred = Fred(api_key=self.key)
        for i in self.variables:
            frames[i] = fred.get_series(i, frequency=self.freq)
            merged = pd.concat(frames, axis=1)
        return merged
