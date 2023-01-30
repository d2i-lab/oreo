from utils.setup import *
from utils.tree import *
from utils.config import *
from offline.states import *
from offline.zorder import *
import numpy as np
import argparse


def compute_and_eval(df, df_sample, ds, args, k, queries):
    method = args.method
    load = args.load
    schedule = {}
    N = len(df)
    if method == "z":
        z = Zorder(df, config, k, 32)
        z.make_partitions()
        read, bids = z.eval(queries)
        offline_label = '%s/labels/offline/%s-%s-%s-%d-%s.p-label'
        path = offline_label % ("resources", config["ds"], fname, args.q, k, "z")
        print("Saving labesl to", path)
        z.save_by_path(path)
        schedule["move"] = [[-1, path]]
    else:
        tb = TreeBuilder(df, df_sample, config, args, k, "")
        if not load:
            _ = tb.compute_offline_oracle(ds, queries, True)
        eval_tree = tb.load_offline_oracle(ds, queries)
        schedule["move"] = [[-1, eval_tree.path]]
        read, bids = eval_tree.eval(queries)
    schedule["query"] = bids
    return read, N, schedule


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Static baseline.')
    parser.add_argument('--config', default="demo", help="Config File Path")
    args = Args(parser.parse_args().config)
    # Fix the random seeds for our experiments
    np.random.seed(args.seed)

    method_str = "Z-order"
    if args.method != "z":
        method_str = "QD-tree"

    fnames, files, parts, config = setup_perfile(args)
    queries = get_workload_perfile(config, files, fnames, args)
    skipped = 0
    size = 0
    for i, fname in enumerate(fnames):
        df, df_sample, k = get_data(config, args, parts, files[i], fname)
        outfile = "%s-%s" % (config["ds"], fnames[i])
        read, N, schedule = compute_and_eval(df, df_sample, outfile, args, k, queries)
        skipped += (1 - read) * N
        size += N
        print("%s,%.3f,%d" % (fname, read, N))
        pickle.dump(schedule, open("resources/schedule/offline/%s-%s-%s-%d-%s.p" % (
            config["ds"], fname, args.q, k, args.method), "wb"))
    skipped = skipped / size
    print("[%s offline] skipped: %.3f, total cost: %.3f" % (
        method_str, skipped, (1-skipped) * len(queries)))


