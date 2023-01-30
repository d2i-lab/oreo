import numpy as np
import dateutil.parser
import datetime
import math


class MetaNode:
    """Partition-level metadata"""
    def __init__(self, df, cfg, pid, compute_meta=False, card_lim=1000):
        self.indices = df.index.values
        self.size = len(df)
        self.pid = pid
        self.cfg = cfg
        self.num_cols = cfg["num_cols"]
        self.date_cols = []
        if "date_cols" in cfg:
            self.date_cols = cfg["date_cols"]
        self.cat_cols = []
        self.hash_cols = []
        self.card_lim = card_lim
        self.meta = {}
        if compute_meta:
            self.populate_metadata(df)

    def populate_metadata(self, df):
        meta = {}
        for col in self.num_cols:
            meta[col] = [np.nanmin(df[col]), np.nanmax(df[col])]
        for col in self.cfg["cat_cols"]:
            vals = set(df[col].unique())
            # Store ranges of hash values for columns with large cardinality
            if len(vals) > self.card_lim:
                hashes = list(map(hash, vals))
                meta[col] = [np.min(hashes), np.max(hashes)]
                self.hash_cols.append(col)
            else:
                meta[col] = vals
                self.cat_cols.append(col)
        for col in self.date_cols:
            vals = set(df[col].unique())
            if "nan" in vals:
                vals.remove("nan")
            if len(vals) == 0:
                meta[col] = ["nan", "nan"]
            else:
                meta[col] = [min(vals), max(vals)]
        self.meta = meta

    def update_metadata(self, df):
        for col in self.num_cols:
            min_v = min(self.meta[col][0], np.nanmin(df[col]))
            max_v = max(self.meta[col][1], np.nanmax(df[col]))
            self.meta[col] = [min_v, max_v]
        for col in self.cat_cols:
            self.meta[col].update(df[col].unique())
        for col in self.hash_cols:
            vals = set(df[col].unique())
            hashes = list(map(hash, vals))
            min_v = min(self.meta[col][0], np.min(hashes))
            max_v = max(self.meta[col][1], np.max(hashes))
            self.meta[col] = [min_v, max_v]


class BoundingBox:
    def __init__(self, cfg, card_lim=1000):
        self.size = 0
        self.cfg = cfg
        self.num_cols = cfg["num_cols"]
        self.date_cols = []
        if "date_cols" in cfg:
            self.date_cols = cfg["date_cols"]
        self.cat_cols = []
        self.hash_cols = []
        self.card_lim = card_lim
        self.meta = {}
        
        meta = {}
        for col in self.num_cols:
            meta[col] = [math.inf, -math.inf]
        for col in self.date_cols:
            meta[col] = [datetime.datetime(9999, 12, 31), datetime.datetime(1, 1, 1)]
        for col in self.cfg["cat_cols"]:
            meta[col] = set()
            self.cat_cols.append(col)
        self.meta = meta

    def __repr__(self):
        return str(self.meta)
    
    def update_row(self, row):
        self.size += 1
        for col in self.num_cols:
            min_v = min(self.meta[col][0], row[col])
            max_v = max(self.meta[col][1], row[col])
            self.meta[col] = [min_v, max_v]
        
        for col in self.hash_cols:
            min_v = min(self.meta[col][0], hash(row[col]))
            max_v = max(self.meta[col][1], hash(row[col]))
            self.meta[col] = [min_v, max_v]

        changes = []
        for col in self.cat_cols:
            self.meta[col].add(row[col])
            if len(self.meta[col]) > self.card_lim:
                # Change it to a hash col
                hashes = list(map(hash, self.meta[col]))
                meta[col] = [np.min(hashes), np.max(hashes)]
                self.hash_cols.append(col)
                changes.append(col)
                
        for col in changes:
            # A bit slow, but should be fine if number of columns isn't large
            self.cat_cols.remove(col)