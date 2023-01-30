import numpy as np
import pandas as pd
import random

from utils.data import *
from utils.meta import *
from enum import Enum


class LogicalOp(Enum):
    AND = 1
    OR = 2

def build_predicate(cfg, preds, ands=1):
    if ands:
        op = LogicalOp.AND
    else:
        op = LogicalOp.OR
    pred = None
    for p in preds:
        if isinstance(p, dict):
            expr = Expr(cfg, p)
        else:
            expr = p
        if pred is None:
            pred = expr
        else:
            pred = Predicate(pred, expr, op)
    return pred

class Predicate:
    def __init__(self, left, right, op):
        self.left = left
        self.right = right
        self.op = op

    def get_leaves(self):
        leaves = []
        leaves.extend(self.left.get_leaves())
        leaves.extend(self.right.get_leaves())
        return leaves

    def eval(self, df):
        m1 = self.left.eval(df)
        m2 = self.right.eval(df)
        if self.op == LogicalOp.AND:
            return m1 & m2
        elif self.op == LogicalOp.OR:
            return m1 | m2
        else:
            print("NO OP", self.left, self.right, self.op)
            return m1
        
    def intersect(self, node):
        left_overlap = self.left.intersect(node)
        if self.op == LogicalOp.AND and not left_overlap:
            return False
        elif self.op == LogicalOp.OR and left_overlap:
            return True
        right_overlap = self.right.intersect(node)
        return right_overlap

    def __repr__(self):
        if self.op == LogicalOp.AND:
            return ("(%s) AND (%s)" % (self.left, self.right))
        else:
            return ("(%s) OR (%s)" % (self.left, self.right))


class Expr:
    def __init__(self, cfg, pred):
        self.cfg = cfg
        self.col = list(pred.keys())[0]
        self.vals = pred[self.col]
        self.expr = pred
        self.is_cat = self.col in self.cfg["cat_cols"]
        self.is_num = self.col in self.cfg["num_cols"]

    # Please use the member variables instead of these functions
    # These are just here to maintain backwards compat. with saved
    # pickle files.
    def _is_num(self):
        return self.col in self.cfg["num_cols"]

    def _is_cat(self):
        return self.col in self.cfg["cat_cols"]

    def _is_date(self):
        return self.col in self.cfg["date_cols"]

    def __repr__(self):
        col = self.col
        vals = self.vals
        if self._is_cat():
            quoted = ['\"%s\"' % v for v in vals]
            cond = '%s IN (%s)' % (col, ','.join(quoted))
        else:
            if self._is_date():
                quoted = []
                for v in vals:
                    if v != v:
                        quoted.append(v)
                    else:
                        quoted.append('\"%s\"' % v)
                vals = quoted
            if len(vals) == 1:  # Point query:
                cond = '%s == %s' % (col, vals[0])
            else:
                if vals[0] != vals[0]:
                    cond = "%s <= %s" % (col, vals[1])
                elif vals[1] != vals[1]:
                    cond = "%s > %s" % (col, vals[0])
                else:
                    cond = "%s >= %s AND %s <= %s" % (col, vals[0], col, vals[1])
        return cond

    def eval(self, df):
        if self._is_cat():
            if len(self.vals) == 1:
                mask = (df[self.col] == self.vals[0])
            else:
                column = df[self.col].values
                m = []
                for v in column:
                    if v in self.vals:
                        m.append(True)
                    else:
                        m.append(False)
                mask = pd.Series(m, index=df.index, name=self.col)
        else:
            if len(self.vals) == 1: # Point query
                mask = (df[self.col] == self.vals[0])
            else:              # Range query
                if self.vals[0] != self.vals[0]:
                    mask = (df[self.col] <= self.vals[1])
                elif self.vals[1] != self.vals[1]:
                    mask = (df[self.col] > self.vals[0])
                else:
                    mask = (df[self.col] >= self.vals[0]) & (df[self.col] <= self.vals[1])
        return mask

    def intersect(self, node):
        vals = self.vals
        meta = node.meta[self.col]
        if self.is_cat:
            # High cardinality categorical columns stored as hash values in metadata
            if self.col in node.hash_cols:
                hashes = list(map(hash, vals))
                hmin, hmax = min(hashes), max(hashes)
                return not (hmax <= meta[0] or hmin >= meta[1])
            # Categorical columns stored as dictionaries in metadata
            else:
                for v in vals:
                    if v in meta:
                        return True
                return False

        # numeric and date columns
        if len(vals) == 1:
            return not (meta[0] > vals[0] or vals[0] > meta[1])
        else:
            vmin, vmax = vals
            if vmax != vmax:      # (>= vmin)
                return meta[1] >= vmin
            elif vmin != vmin:      # (<= vmax)
                return meta[0] < vmax
            else:
                return not (vmax <= meta[0] or vmin > meta[1])

    def get_leaves(self):
        # For backwards compatibility, since some preds did not have is_num
        # and is_cat attributes in previous pickle files. This will remake
        # the pred with new member variables
        if not hasattr(self, 'is_cat'):
            self.is_cat = self.col in self.cfg["cat_cols"]
            self.is_num = self.col in self.cfg["num_cols"]

        if not self.is_cat and len(self.vals) == 2:
            vmin, vmax = self.vals
            if vmin != vmin or vmax != vmax:
                return [self]
            else:
                # Break range query into two predicates
                e1 = Expr(self.cfg, {self.col: [vmin, np.nan]})
                e2 = Expr(self.cfg, {self.col: [np.nan, vmax]})
                return [e1, e2]
        else:
            return [self]









