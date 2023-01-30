import multiprocessing
import os
import time
from collections import defaultdict
from offline.layout import *
from utils.predicates import *
from utils.workload import *
from tqdm import tqdm
from collections import deque


def build_edges(parent, left_child, right_child):
    parent.children[0] = left_child
    parent.children[1] = right_child
    left_child.parent = parent
    right_child.parent = parent
    left_child.depth = parent.depth + 1
    right_child.depth = parent.depth + 1
    return parent, left_child, right_child


def output_tree(nodes):
    tree = {}
    depth = 0
    while len(nodes) > 0:
        remaining = []
        level = {}
        for n in nodes:
            if n.depth == depth:
                if not n.is_leaf():
                    level[str(n.pid)] = [n.cond, n.left().pid, n.right().pid]
            else:
                remaining.append(n)
        nodes = remaining
        if len(level) > 0:
            tree[str(depth)] = level.copy()
        depth += 1
    return tree


def _split(node_df, cfg, pred, pid, min_size):
    N = len(node_df)
    mask = pred.eval(node_df)
    idx1 = np.nonzero(mask.values)[0]
    if len(idx1) >= min_size and N - len(idx1) >= min_size:
        df1 = node_df[mask]
        df2 = node_df[~mask]
        node1 = TreeNode(df1, cfg, None, pid, True)
        node2 = TreeNode(df2, cfg, None, pid+1, True)
        return node1, node2
    else:
        return None, None


def _split_by_col(node_df, col, cfg, pid):
    N = len(node_df)
    vals = node_df[col].values
    sorted_indices = np.argsort(vals)
    index = N // 2 - 1
    median_val = vals[sorted_indices[index]]
    if median_val == max(vals):
        while index > 0 and vals[sorted_indices[index]] == median_val:
            index -= 1
    median_val = vals[sorted_indices[index]]
    pred = Expr(cfg, {col: [median_val, np.nan]})
    mask = pred.eval(node_df)
    df1 = node_df[mask]
    if 0 < len(df1) < N:
        df2 = node_df[~mask]
        node1 = TreeNode(df1, cfg, None, pid, True)
        node2 = TreeNode(df2, cfg, None, pid+1, True)
        return node1, node2, pred
    else:
        return None


def _build_node(node_df, cfg, pred, pid):
    mask = pred.eval(node_df)
    df1 = node_df[mask]
    df2 = node_df[~mask]
    node1 = TreeNode(df1, cfg, None, pid)
    node2 = TreeNode(df2, cfg, None, pid+1)
    return node1, node2


def build_tree(tree_info, df, cfg):
    root_id = int(list(tree_info["0"].keys())[0])
    root = TreeNode(df, cfg, None, root_id)
    node_map = {str(root_id): root}
    level = 0
    pid = root_id
    while str(level) in tree_info:
        tree_level = tree_info[str(level)]
        for nid in tree_level:
            cond, lid, rid = tree_level[nid]
            parent = node_map[nid]
            node_df = df.iloc[parent.indices]
            n1, n2 = _build_node(node_df, cfg, cond, 0)
            n1.pid = lid
            n2.pid = rid
            parent, n1, n2 = build_edges(parent, n1, n2)
            parent.cond = cond
            node_map[str(lid)] = n1
            node_map[str(rid)] = n2
            pid = np.max([pid, lid, rid])
        level += 1
    return root, pid


class TreeNode(MetaNode):
    def __init__(self, df, cfg, parent, id, compute_meta=False):
        super().__init__(df, cfg, id, compute_meta)
        self.parent = parent
        self.depth = 0
        if parent is not None:
            self.depth = parent.depth + 1
        self.children = [None, None]
        self.cond = None

    def update_metadata(self, df):
        super().update_metadata(df)

    def is_leaf(self):
        return (self.children[0] is None) and (self.children[1] is None)

    def left(self):
        return self.children[0]

    def right(self):
        return self.children[1]

    def set_left(self, node):
        self.children[0] = node

    def set_right(self, node):
        self.children[1] = node

    def get_leaves(self):
        leafs = []
        for child in self.children:
            if child is not None:
                if child.is_leaf():
                    leafs.append(child)
                else:
                    leafs.extend(child.get_leaves())
        return leafs


class QDTree(Layout):
    """Greedy construction for qd-tree"""
    def __init__(self, df, cfg, workload, k, preds=None, verbose=True):
        super().__init__(df, cfg, k)
        self.workload = workload
        self.root = TreeNode(df, cfg, None, 0, True)
        self.pid = 1
        self.verbose = verbose
        self.min_size = len(df) // k // 2
        self.preds = preds

    def _resample_splits(self):
        sampled = []
        orig = {}
        for i, pred in enumerate(self.preds):
            col = pred.col
            if not col in orig:
                orig[col] = []
            if col in self.cfg["num_cols"] or col in self.cfg["date_cols"]:
                orig[col].append(pred)
            else:
                sampled.append(pred)
        for col in orig:
            splits = orig[col]
            # Reduce granularity of numerical splits
            if len(splits) > self.k * 10:
                new_splits = np.random.choice(splits, self.k * 10, replace=False)
                sampled.extend(list(new_splits))
            else:
                sampled.extend(splits)
        if self.verbose:
            print("Resample predicates (%d): %d => %d" % (
                len(self.workload), len(self.preds), len(sampled)))
        self.preds = sampled

    def _split_node(self, node, pid):
        # Try each predicate as a split
        min_read = 1
        card_lim = 1000
        node_df = self.df.iloc[node.indices]
        cat_cols, num_cols, date_cols, hash_cols = list(self.cfg['cat_cols']), list(self.cfg['num_cols']), [], []
        if 'date_cols' in self.cfg:
            date_cols = list(self.cfg['date_cols'])
                
        for col in cat_cols:
            series = (~node_df[col].duplicated()).cumsum()
            if series.values[-1] > card_lim:
                hash_cols.append(col)
        for col in hash_cols:
            cat_cols.remove(col)
            
        workload = []
        if pid > 1:
            for query in self.workload:
                if query.intersect(node):
                    workload.append(query)
        else:
            workload = self.workload
        
        new_df = node_df.copy()
        new_df[hash_cols] = new_df[hash_cols].applymap(hash) 
                    
        preds_by_col = defaultdict(set)
        col_to_pred = {}
        for pred in self.preds:
            vals = [x for x in pred.vals if x == x]
            preds_by_col[pred.col].add(vals[0])
            col_to_pred[(pred.col, vals[0])] = pred
            
        L = len(node_df)
        best_p = None
        optimize_search = True

        for col in preds_by_col:
            if len(workload) == 0:
                continue
            preds_by_col[col] = list(preds_by_col[col])
            if not optimize_search or col in self.cfg['cat_cols']:
                for p in preds_by_col[col]:
                    pred = col_to_pred[(col, p)]
                    n1, n2 = _split(node_df, self.cfg, pred, pid, self.min_size)
                    c, _ = cost([n1, n2], self.workload, self.N)
                    if c < 0.99 * min_read:
                        min_read = c
                        best_p = pred
            else:
                preds_by_col[col].sort()
                new_df.sort_values(by=[col], inplace=True)
                
                all_uniques = set()
                uniques = pd.DataFrame()
                for ccol in cat_cols:
                    uniques[ccol] = (~new_df[ccol].duplicated()).cumsum()    
                has_unique = ~uniques.duplicated()
                
                uniques = pd.DataFrame()
                for ccol in cat_cols:
                    uniques[ccol] = (~new_df[ccol][::-1].duplicated()).cumsum()    
                rhas_unique = ~uniques.duplicated()
                
                left_catcol, right_catcol = defaultdict(list), defaultdict(list)
                left_bpts, right_bpts = defaultdict(deque), defaultdict(deque)
                left_set, right_set = {col: 0 for col in cat_cols}, {col: -1 for col in cat_cols}
                for ccol in cat_cols:
                    left_catcol[ccol].append(set())
                    right_catcol[ccol].append(set())
                    
                cat_values = new_df[cat_cols].values
                for i in has_unique.to_numpy().nonzero()[0]:
                    for j, ccol in enumerate(cat_cols):
                        if cat_values[i, j] not in left_catcol[ccol][-1]:
                            copy = set(left_catcol[ccol][-1])
                            copy.add(cat_values[i, j])
                            left_catcol[ccol].append(copy)
                            left_bpts[ccol].append(i)
                            all_uniques.add(i)
                
                for i in rhas_unique.to_numpy().nonzero()[0]:
                    for j, ccol in enumerate(cat_cols):
                        if cat_values[L-i-1, j] not in right_catcol[ccol][-1]:
                            copy = set(right_catcol[ccol][-1])
                            copy.add(cat_values[L-i-1, j])
                            right_catcol[ccol].append(copy)
                            right_bpts[ccol].append(L-i-1)
                            all_uniques.add(L-i-1)
                                
                reduce_cols = num_cols + hash_cols + date_cols
                lmin_df, lmax_df = new_df[reduce_cols].cummin(), new_df[reduce_cols].cummax()
                rmin_df, rmax_df = new_df[reduce_cols][::-1].cummin(), new_df[reduce_cols][::-1].cummax()
                
                index, last_index = 0, -1
                sort_values = new_df[col].values
                left_bbox, right_bbox = lambda: None, lambda: None
                left_bbox.hash_cols, right_bbox.hash_cols = hash_cols, hash_cols
                for p in preds_by_col[col]:
                    while index < L and sort_values[index] <= p:
                        if index in all_uniques:
                            # Update categorical columns set
                            for ccol in cat_cols:
                                if left_bpts[ccol] and index == left_bpts[ccol][0]:
                                    left_set[ccol] += 1
                                    left_bpts[ccol].popleft()
                                if right_bpts[ccol] and index == right_bpts[ccol][-1]:
                                    right_set[ccol] -= 1
                                    right_bpts[ccol].pop()
                        index += 1

                    if index < self.min_size or index == last_index:
                        continue
                    elif index > L - self.min_size:
                        break

                    # Evaluate cost function
                    # At this point, the left node should have indices [0, ..., index)
                    # The right should have [index, ... N)
                    # Create the bounding boxes
                    left_meta, right_meta = {}, {}
                    left_size, right_size = index, L - index
                    for ccol in num_cols + date_cols + hash_cols:
                        left_meta[ccol] = [lmin_df[ccol].values[index-1], lmax_df[ccol].values[index-1]]
                        right_meta[ccol] = [rmin_df[ccol].values[-index-1], rmax_df[ccol].values[-index-1]]
                    for ccol in cat_cols:
                        lvals, rvals = left_catcol[ccol][left_set[ccol]], right_catcol[ccol][right_set[ccol]]
                        left_meta[ccol] = lvals
                        right_meta[ccol] = rvals    
                    left_bbox.meta, right_bbox.meta = left_meta, right_meta
                    
                    read = 0
                    scale = 1. / (self.N * len(workload))
                    for query in workload:
                        read += left_size * scale if query.intersect(left_bbox) else 0
                        read += right_size * scale if query.intersect(right_bbox) else 0
                        if read > 0.99 * min_read:
                            break
                    
                    if read < 0.99 * min_read:
                        min_read = read
                        best_p = col_to_pred[(col, p)]
                    last_index = index
                    
        # All else failed: split by half according to sort column
        if best_p is None:
            cols = list(self.cfg["sort_cols"])
            cols.extend(self.cfg["num_cols"])
            cols.extend(self.cfg["date_cols"])
            best_split = None
            for sort_col in cols:
                vmin = node_df[sort_col].min()
                vmax = node_df[sort_col].max()
                if vmin == vmax:
                    continue
                best_split = _split_by_col(node_df, sort_col, self.cfg, pid)
                if best_split is not None:
                    break
        else:
            n1, n2 = _split(node_df, self.cfg, best_p, pid, self.min_size)
            best_split = (n1, n2, best_p)
        return best_split

    def _get_all_nodes(self):
        all_nodes = []
        to_visit = [self.root]
        while len(to_visit) > 0:
            all_nodes.extend(to_visit)
            next_level = []
            while len(to_visit) > 0:
                node = to_visit.pop(0)
                for child in node.children:
                    if child is not None:
                        next_level.append(child)
            to_visit = next_level
        return all_nodes

    def make_partitions(self):
        # Get splits
        if self.preds is None:
            self.preds = get_splits(self.df, self.workload, self.min_size)
        self._resample_splits()

        level = [self.root]
        part_count = 1
        if self.verbose:
            progress = tqdm(total=self.k)
            progress.update()

        while len(level) > 0 and part_count < self.k:
            # Sort nodes by size * # queries
            sizes = []
            costs = []
            next_level = []
            # Special case: if no queries intersect with node
            for node in level:
                s = node.size
                c, _ = cost([node], self.workload, s)
                sizes.append(s)
                costs.append(s * c)
            indices = np.argsort(costs)[::-1]

            single_thread = False
            if single_thread:
                # Single thread
                results, params = [], []
                for idx in indices:
                    if sizes[idx] >= self.min_size * 2:
                        params.append([level[idx], self.pid])
                        results.append(self._split_node(level[idx], self.pid))
                        self.pid += 2
            else:       
                # Multi thread
                p = multiprocessing.Pool(1)
                params = []
                for idx in indices:
                    if sizes[idx] >= self.min_size * 2:
                        params.append([level[idx], self.pid])
                        self.pid += 2
                results = p.starmap(self._split_node, params)
                p.close()

            for i, item in enumerate(results):
                split = results[i]
                n1, n2, cond = split
                node, n1, n2 = build_edges(params[i][0], n1, n2)
                node.cond = cond
                next_level.extend([n1, n2])
                part_count += 1
                if self.verbose:
                    progress.update()
                if part_count == self.k:
                    break
            level = next_level
        if self.verbose:
            progress.close()
        # Get leaf nodes
        self.parts = self.root.get_leaves()

    def save_by_path(self, path):
        nodes = self._get_all_nodes()
        tree_info = output_tree(nodes)
        pickle.dump(tree_info, open(path, "wb"))
        self.path = path

    def load_by_path(self, path):
        tree_info = pickle.load(open(path, "rb"))
        root, pid = build_tree(tree_info, self.df, self.cfg)
        self.root = root
        self.parts = self.root.get_leaves()
        N = len(self.df)
        labels = np.zeros(N)
        for part in self.parts:
            indices = part.indices
            labels[indices] = part.pid
            node_df = self.df.iloc[part.indices]
            part.populate_metadata(node_df)
        self.labels = labels
        self.pid = pid + 1
        self.path = path

    def print_by_level(self):
        nodes = self._get_all_nodes()
        depth = 0
        while len(nodes) > 0:
            print("Lvl: %d" % depth)
            remaining = []
            for n in nodes:
                if n.depth == depth:
                    if not n.is_leaf():
                        print('%d (%d, %s): %d, %d' % (
                            n.pid, n.size, str(n.cond), n.left().pid, n.right().pid))
                    else:
                        print('%d (%d): leaf' % (n.pid, n.size))
                else:
                    remaining.append(n)
            nodes = remaining
            depth += 1

    def print_splits(self):
        nodes = self._get_all_nodes()
        depth = 0
        sizes = []
        splits = []
        while len(nodes) > 0:
            remaining = []
            for n in nodes:
                if n.depth == depth:
                    if not n.is_leaf():
                        splits.append(str(n.cond))
                    else:
                        sizes.append(n.size)
                else:
                    remaining.append(n)
            nodes = remaining
            depth += 1
        print(splits, sum(sizes), sizes)

