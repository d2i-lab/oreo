from offline.layout import *
from sklearn.preprocessing import MinMaxScaler
import math
import collections
from tqdm import tqdm


def decimalToBinary(n, max_bits):
    raw = bin(n).replace("0b", "")
    return '0' * (max_bits-len(raw)) + raw


def order_mapping(order, lim):
    ncols = len(lim)
    idx = [0] * ncols
    mapping = []
    for o in order:
        mapping.append(idx[o] + sum(lim[:o]))
        idx[o] += 1
    return mapping


def get_row_code(vals, lim, mapping):
    bins = []
    for i, val in enumerate(vals):
        bins.append(decimalToBinary(int(val), lim[i]))
    code = ''.join(bins)

    zorder = []
    for i in mapping:
        zorder.append(code[i])
    return ''.join(zorder)


def code_to_label(z_order, k):
    indices = np.argsort(z_order)
    part_size = len(z_order) // k + 1
    bids = np.zeros(len(z_order))
    for i in range(k):
        if i == k-1:
            bids[indices[i*part_size:]] = i
        else:
            bids[indices[i*part_size:(i+1)*part_size]] = i
    return bids


def get_top_columns(cfg, workload, cutoff=3):
    cnt = {}
    for q in workload:
        leaves = q.get_leaves()
        for leaf in leaves:
            col = leaf.col
            if not col in cnt:
                cnt[col] = 0
            cnt[col] += 1
    # Heuristic: date and numeric columns before categorical columns
    for col in cnt:
        if col in cfg["date_cols"]:
            cnt[col] += 0.2
        elif col in cfg["num_cols"]:
            cnt[col] += 0.1
        if col in cfg["sort_cols"]:
            cnt[col] += 0.1
    sorted_cnt = dict(sorted(cnt.items(), key=lambda item: item[1], reverse=True))
    return list(sorted_cnt.keys())[:min(cutoff, len(sorted_cnt))]


class Zorder(Layout):
    def __init__(self, df, cfg, k, max_bits=32, workload=None, verbose=True):
        super().__init__(df, cfg, k)
        self.max_bits = max_bits
        self.lim = []
        self.sort_cols = cfg["sort_cols"]
        self.verbose = verbose
        if workload is not None:
            self.sort_cols = get_top_columns(cfg, workload)

    def _remove_unused_cols(self, cfg, df):
        # Extract columns that are involved in the query
        cols = []
        idx = []
        for i, col in enumerate(df.columns):
            if not col in self.sort_cols:
                continue
            cols.append(col)
            idx.append(i)
        df = df[cols]
        # Store column types
        types = []
        for col in df.columns:
            if col in cfg["num_cols"]:
                types.append("num")
            elif col in cfg["cat_cols"]:
                types.append("cat")
            else:
                types.append("date")
        self.d = len(cols)
        self.types = types
        return df

    def _transform(self, df):
        """Prepare columns values for Z-ordering"""
        new_df = collections.OrderedDict()
        for i, col in enumerate(df.columns):
            vals = df[col].values
            # Normalize numeric values to integers in [0, 2^(max_bits)]
            if self.types[i] == "num":
                scaler = MinMaxScaler()
                vals = vals.reshape(-1, 1)
                scaler.fit(vals)
                new_vals = np.squeeze(scaler.transform(vals)) * (
                        np.power(2, self.max_bits) - 1)
                new_vals = new_vals.astype(int)
                self.lim.append(self.max_bits)
            # Transform strings into binary by keeping track of unique values
            else:
                # Integer encode strings via alphabetical order
                # This is needed since date columns have orders
                mapping = {}
                unique = sorted(list(set(vals)))
                for val in unique:
                    mapping[val] = len(mapping)
                new_vals = []
                for val in vals:
                    new_vals.append(mapping[val])
                num_bits = int(math.log2(len(mapping)))
                if math.pow(2, num_bits) < len(mapping):
                    num_bits += 1
                self.lim.append(min(self.max_bits, num_bits))
            new_df[col] = new_vals
        new_dataframe = pd.DataFrame.from_dict(new_df)
        return new_dataframe.values

    def _gen_ordering(self):
        ordering = []
        lim = list(self.lim)
        for i in range(self.max_bits):
            for j in range(self.d):
                if lim[j] > 0:
                    ordering.append(j)
                    lim[j] -= 1
        return ordering

    def _get_codes(self):
        if self.verbose:
            print("Computing Z-order...")
        mapping = order_mapping(self.order, self.lim)
        zorder = []
        if self.verbose:
            progress = tqdm(total=self.N, miniters=1000)
        for i in range(self.N):
            zorder.append(get_row_code(self.vals[i], self.lim, mapping))
            if self.verbose:
                progress.update()
        return zorder

    def _get_labels(self):
        df = self._remove_unused_cols(self.cfg, self.df)
        self.vals = self._transform(df)
        self.order = self._gen_ordering()
        self.codes = np.array(self._get_codes())
        self.labels = code_to_label(self.codes, self.k)

    def make_partitions(self):
        self._get_labels()
        self.load_by_labels(self.labels)

    def save_by_path(self, path):
        self.path = path
        pickle.dump(self.labels, open(self.path, "wb"))

    def load_by_path(self, path):
        labels = pickle.load(open(path, "rb"))
        self.path = path
        self.load_by_labels(labels)
