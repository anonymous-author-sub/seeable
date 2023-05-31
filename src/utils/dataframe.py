from numpy import not_equal
import pandas as pd
from typing import Any, List, Optional, Callable, Dict

from .types import check_type


def get_col_unique(df: pd.DataFrame, col: str) -> Optional[List[str]]:
    """get unique values of a column in the df"""
    if col not in df:
        return
    col_values = df[col].unique().tolist()
    col_values = [x for x in col_values if check_type(x)]
    return col_values


def describe_col(df: pd.DataFrame, col: str) -> Optional[pd.DataFrame]:
    values = get_col_unique(df, col)
    if values is None:
        return
    # metrics
    count = df[col].value_counts()
    freq = (count / count.sum()).apply(lambda x: f"({100*x:.2f} %)")
    # generate string
    desc = pd.concat([count, freq], axis=1)
    if not desc.empty:
        return desc


def describe(df: pd.DataFrame, columns: List[str] = ["label", "fold", "split"]):
    """
    {'fold': [['method_A', 3975, '(75.71 %)'],
              ['original_videos', 1131, '(21.54 %)'],
              ['method_B', 144, '(2.74 %)']],
    'label': [[1, 4119, '(78.46 %)'], [0, 1131, '(21.54 %)']],
    'split': [['train', 4473, '(85.20 %)'], ['test', 777, '(14.80 %)']]}
    """
    print("[Describe dataframe]")
    print(f"total: {len(df)} rows")
    print()
    out = {}
    for col in columns:
        print(">>>", col)
        col_df = describe_col(df, col)
        if col_df is not None:
            desc_str = col_df.to_string(header=False)
            print(desc_str, "\n")
            out[col] = [
                [idx] + [c for c in row]
                for idx, row in zip(col_df.index, col_df.to_numpy())
            ]
    return out


def convert_partial_int(df):
    def convert_obj(c):
        try:
            return c.is_integer()
        except:
            pass
        if c is True or c is False:
            return False
        if isinstance(c, str) and c.isnumeric():
            return True
        if isinstance(c, int):
            return True
        if isinstance(c, float):
            return c.is_integer()
        return False
        # raise Exception(f'unkown type : {type(c)}')

    # columns that miss some values
    missing = df.columns[df.isnull().any()].tolist()

    # columns that are all int
    convert_list = []
    for col in missing:
        C = df[col].dropna().to_list()
        cond = all([convert_obj(c) for c in C])
        # print(col, cond)
        if cond:
            convert_list.append(col)

    # convert them
    for col in convert_list:
        try:
            df[col] = df[col].astype(pd.Int64Dtype())
        except:
            print(f"failed to convert {col}")
            print(df[col].unique().tolist())

    return df
