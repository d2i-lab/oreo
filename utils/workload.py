import numpy as np
import pickle
import datetime

from dateutil.relativedelta import relativedelta
from utils.predicates import *
from utils.meta import *
from collections import defaultdict

def flatten_lists(nested):
    flattened = []
    for l in nested:
        flattened.extend(l)
    return flattened


def load_workload_from_pickle(fname, templates=False):
    if not templates:
        workload = []
        queries = pickle.load(open(fname, "rb"))
        n = len(queries)
        for i in range(n):
            workload.append(queries[str(i)])
    else:
        workload = flatten_lists(load_template_workload_from_pickle(fname))
    return workload


def load_template_workload_from_pickle(fname):
    workload = []
    queries = pickle.load(open(fname, "rb"))
    for key in queries:
        qs = queries[key]
        n = len(qs)
        tmp = []
        for i in range(n):
            tmp.append(qs[str(i)])
        workload.append(tmp)
    return workload


def sample_workload(gen, data_config, spec, n=1000):
    preds = gen.gen_pred(spec, n)
    workload = []
    for pred in preds:
        workload.append(build_predicate(data_config, pred, 1))
    return workload


def gen_random_workload_from_template(gen, data_config, q_config, total_queries):
    wl = WorkloadSimulator(len(q_config)).gen_workload(total_queries)
    keys = list(q_config.keys())
    preds = []
    for [id, n] in wl:
        spec = q_config[keys[id]]
        queries = gen.gen_pred(spec, n)
        preds.extend(queries)
    workload = []
    for pred in preds:
        clause = build_predicate(data_config, pred, 1)
        workload.append(clause)
    return workload


def gen_random_query_config(cols, num_wl, data_config, use_all=False):
    query_config = {}
    for i in range(num_wl):
        if not use_all:
            k = np.random.randint(min(1, len(cols)), min(len(cols) // 2 + 2, 7))
            selected = np.random.choice(cols, k, replace=False)
        else:
            selected = cols
        spec = {}
        for col in selected:
            if col in data_config["cat_cols"]:
                spec[col] = "=="
            else:
                spec[col] = ">"
        query_config[str(i)] = spec
    return query_config


def gen_random_workload(cols, gen, data_config, num_wl, total_queries, use_all=False):
    q_config = gen_random_query_config(cols, num_wl, data_config, use_all)
    queries = gen_random_workload_from_template(gen, data_config, q_config, total_queries)
    return queries

def get_splits(df, workload, min_size):
    preds_by_col = defaultdict(list)
    N = len(df)
    # Dedup predicates
    seen_queries = {}
    for i, query in enumerate(workload):
        preds = query.get_leaves()
        for pred in preds:
            vals = pred.vals
            col = pred.col
            
            # Special case: change point number query to range queries
            # as splitting criteria
            if len(vals) == 1 and pred.is_num:
                pred = Expr(pred.cfg, {col: [vals[0], np.nan]})
                vals = pred.vals
                
            expr = str(pred)
            if not expr in seen_queries:
                seen_queries[expr] = 1
                if len(vals) == 2 and vals[1] == vals[1]:
                    pred.vals.reverse()
                preds_by_col[col].append(pred)
                
    splits = []
    for col in preds_by_col:
        pred = preds_by_col[col][0]
        if pred.is_cat and np.unique(df[col]).size > 2000:
            print(col, 'excluded as it was a hash column.')
            continue
        preds_by_col[col].sort(key=lambda x: x.vals[0])
        values = np.sort(df[col].values)
        
        for pred in preds_by_col[col]:
            if pred.is_cat:            
                x = np.searchsorted(values, pred.vals[0], side='left')
                y = np.searchsorted(values, pred.vals[0], side='right')
                size = y - x
            else:
                size = np.searchsorted(values, pred.vals[0], side='right')

            if size > min_size and N - size > min_size:
                splits.append(pred)
    return splits

def _combine_workload(wl):
    combined = []
    i = 0
    while i < len(wl):
        j = i + 1
        n = wl[i][1]
        while j < len(wl) and wl[j][0] == wl[i][0]:
            n += wl[j][1]
            j += 1
        combined.append([wl[i][0], n])
        i = j
    return combined


class WorkloadSimulator:
    def __init__(self, k):
        self.k = k
        self.interval = 200
        self.prob = []
        for i in range(k):
            # p_self: probability of staying in the current state
            p_self = 0.6
            raw = np.random.rand(k-1)
            raw = list(raw / np.sum(raw) * (1-p_self))
            self.prob.append(raw[:i] + [p_self] + raw[i:])
        self.curr_state = np.random.randint(0, k)

    def next(self):
        self.curr_state = np.random.choice(list(range(self.k)), 1, p=self.prob[self.curr_state])[0]

    def gen_workload(self, n, simple=True):
        if self.k == 1:
            return [[0, n]]
        workload = []
        if simple:
            m = n // self.k
            extra = n - m * self.k
            for i in range(self.k):
                if i == 0:
                    workload.append([i, m+extra])
                else:
                    workload.append([i, m])
        else:
            tot = 0
            for _ in range(n // self.interval):
                workload.append([self.curr_state, self.interval])
                tot += self.interval
                self.next()
            if tot < n:
                workload.append([self.curr_state, n - tot])
        return _combine_workload(workload)


class PredicateGenerator:
    def __init__(self, cfg, fname, load=False):
        self.cfg = cfg
        self.fname = fname
        if load:
            self.load_meta()

    def compute_meta(self, df):
        self.meta = MetaNode(df, self.cfg, True).meta
        self.save_meta()

    def save_meta(self):
        pickle.dump(self.meta, open("resources/data/%s.p" % self.fname, "wb"))

    def load_meta(self):
        self.meta = pickle.load(open("resources/data/%s.p" % self.fname, "rb"))

    def _cat_query(self, col, k=1):
        """
        Generate a single predicate clause in the form of (col IN [val1, val2, ...])
        :return: {col: [vals]}
        """
        all_vals = sorted(list(self.meta[col]))
        prob = np.random.uniform(0, 1, len(all_vals))
        indices = np.argsort(prob)
        selected = []
        for i in range(k):
            selected.append(all_vals[indices[i]])
        return {col: selected}

    def _num_query(self, col, op):
        mmin, mmax = self.meta[col]
        v1 = np.random.uniform(mmin, mmax)
        if op == "==":              # (col == val)
            return {col: [int(v1)]}
        elif op in ["<", "<="]:     # (col < val)
            return {col: [np.nan, v1]}
        elif op in [">", ">="]:     # (col > val)
            return {col: [v1, np.nan]}
        else:               # (v1 < col < v2)
            if op == "<>":  # no selectivity specification
                v2 = np.random.uniform(mmin, mmax)
                return {col: [min(v1,v2), max(v1,v2)]}
            else:           # generate query with specific width
                if op[0] == "s":
                    delta = (mmax - mmin) * float(op[1:])
                else:
                    delta = float(op)
                v1 = np.random.uniform(mmin, mmax-delta)
                return {col: [v1, v1 + delta]}

    def _random_date(self, col):
        start_date = dateutil.parser.parse(self.meta[col][0])
        end_date = dateutil.parser.parse(self.meta[col][1])
        # dates = list(map(dateutil.parser.parse, vals))
        time_range = (end_date - start_date).days
        time_delta = np.random.randint(1, time_range)
        new_date = start_date + datetime.timedelta(days=time_delta)
        return datetime.datetime.strftime(new_date, self.cfg["date_format"])

    def _date_to_str(self, date):
        return datetime.datetime.strftime(date, self.cfg["date_format"])

    def _date_query(self, col, op):
        state_date, end_date = self.meta[col]
        new_date = self._random_date(col)
        if op == "==":              # (col == val)
            return {col: [new_date]}
        elif op in ["<", "<="]:     # (col < val)
            return {col: [np.nan, new_date]}
        elif op in [">", ">="]:     # (col > val)
            return {col: [new_date, np.nan]}
        else:
            if op == "<>":  # no selectivity specification
                new_date2 = self._random_date(col)
                return {col: [min(new_date, new_date2), max(new_date, new_date2)]}
            else:           # generate range query that is X days apart
                val = int(op[1:])
                if op[0] == 'y':     # years
                    delta = relativedelta(years=val)
                elif op[0] == 'm':   # month
                    delta = relativedelta(months=val)
                else:
                    delta = relativedelta(days=val)
                end_date_parsed = dateutil.parser.parse(end_date)
                while dateutil.parser.parse(new_date) + delta > end_date_parsed:
                    new_date = self._random_date(col)
                new_end = dateutil.parser.parse(new_date) + delta
            return {col: [new_date, self._date_to_str(new_end)]}

    def pred(self, col, op):
        if col in self.cfg["num_cols"]:
            return self._num_query(col, op)
        elif "date_cols" in self.cfg and col in self.cfg["date_cols"]:
            return self._date_query(col, op)
        else:
            if op == "<" or op == ">" or op == "<>":
                return self._num_query(col, op)
            k = 1
            if "IN" in op: # IN query
                k = int(op.split(",")[1])
            return self._cat_query(col, k)

    def gen_pred(self, spec, n):
        # Connect by ANDs by default
        preds = []
        for i in range(n):
            pred = []
            for col in spec:
                pred.append(self.pred(col, spec[col]))
            preds.append(pred)
        return preds