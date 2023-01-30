import numpy as np
import pandas as pd
import pickle
from utils.meta import *


def cost(parts, workload, N, avg=True):
    if parts[0] is None:
        if avg:
            return 1, []
        else:
            return [1] * len(workload), []

    read = [0] * len(workload)
    read_pids = []
    for i, query in enumerate(workload):
        pids = []
        for part in parts:
            intersect = query.intersect(part)
            # No overlap with predicate => can skip
            if intersect:
                read[i] += part.size
                pids.append(part.pid)
        read_pids.append(pids)

    read = np.array(read) * 1.0 / N
    if avg:
        return np.average(read), read_pids
    else:
        return read, read_pids


class Layout:
    def __init__(self, df, cfg, k):
        self.df = df
        self.cfg = cfg
        self.k = k
        self.N = len(df)
        self.path = ""
        self.parts = []

    def save_labels(self, path):
        pickle.dump(self.labels, open(path, "wb"))

    def save_by_path(self, path):
        self.path = path

    def load_by_path(self, path):
        self.path = path

    def make_partitions(self):
        pass

    def load_by_labels(self, labels):
        self.parts = []
        ids = set(labels)
        labels = np.array(labels)
        for i in ids:
            pid = int(i)
            indices = np.where(labels == pid)
            part_df = self.df.iloc[indices]
            node = MetaNode(part_df, self.cfg, pid, True)
            self.parts.append(node)
        self.labels = labels

    def eval(self, queries, avg=True):
        return cost(self.parts, queries, self.N, avg)