import pandas as pd
import numpy as np
from os import listdir
from os.path import isfile, join


def get_query_cols(q_config):
    cols = []
    for id in q_config:
        cols.extend(list(q_config[id].keys()))
    return list(set(cols))


def get_filenames(path):
    files = []
    for f in listdir(path):
        if isfile(join(path, f)) and ('csv' in f or 'tbl' in f):
            files.append(f)
    return files


def load_csv(file, config):
    cat_cols = config["cat_cols"]
    dtypes = {}
    for col in cat_cols:
        dtypes[col] = "str"
    for col in config["num_cols"]:
        if 'KEY' in col:
            dtypes[col] = "int"
        else:
            dtypes[col] = "float"
    for col in config["date_cols"]:
        dtypes[col] = "str"
    for col in config["int_cols"]:
        dtypes[col] = "int"
    if "bool_cols" in config:
        for col in config["bool_cols"]:
            dtypes[col] = "bool"
    df = pd.read_csv(
        file,
        delimiter=config["delimiter"],
        header=0,
        usecols=list(dtypes.keys()),
        dtype=dtypes
    )
    for col in cat_cols:
        if not col in config["int_cols"]:
            df[col] = df[col].fillna("nan").str.strip()
    for col in config["date_cols"]:
        df[col] = df[col].fillna("nan")
    for col in config["num_cols"]:
        df[col] = df[col].fillna(0)
    return df


def load_df(config, concat=True):
    fnames = get_filenames(config["path"])
    files = [join(config["path"], f) for f in fnames]
    dtypes = {}
    for col in config["cat_cols"]:
        dtypes[col] = "str"
    for col in config["num_cols"]:
        if 'KEY' in col:
            dtypes[col] = "int"
        else:
            dtypes[col] = "float"
    for col in config["date_cols"]:
        dtypes[col] = "str"
    for col in config["int_cols"]:
        dtypes[col] = "int"
    if "bool_cols" in config:
        for col in config["bool_cols"]:
            dtypes[col] = "bool"
    dfs = []
    for file in files:
        df = pd.read_csv(
            file,
            delimiter=config["delimiter"],
            header=0,
            usecols=list(dtypes.keys()),
            dtype=dtypes
        )
        for col in config["cat_cols"]:
            if not col in config["int_cols"]:
                df[col] = df[col].fillna("nan").str.strip()
        for col in config["date_cols"]:
            df[col] = df[col].fillna("nan")
        for col in config["num_cols"]:
            df[col] = df[col].fillna(0)
        dfs.append(df)
    if concat:
        return df, config["ds"]
    else:
        return dfs, fnames
