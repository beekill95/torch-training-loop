from __future__ import annotations

import pandas as pd


def assert_dataframes_equal(df1: pd.DataFrame, df2: pd.DataFrame):
    for idx in df1.index:
        assert df1.loc[idx].to_dict() == df2.loc[idx].to_dict()
