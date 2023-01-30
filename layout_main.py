from utils.setup import *
from utils.tree import *
from utils.config import *
from offline.states import *
from online.counter import *
import numpy as np
import argparse
import os


def gen_layout(queries, out, k):
    tb = TreeBuilder(df, df_sample, config, args, k, out)
    if args.policy == 'oracle':
        queries = pickle.load(open("resources/query/%s-oracle.p" % config["ds"], "rb"))
        _ = tb.get_init_states(queries)
    else:
        init_states = tb.get_init_states()
        sg = StateGenerator(tb, init_states, args.interval)
        sg.reset_reservoir(args.res)
        T = len(queries) // args.interval
        for i in range(T):
            new_queries = queries[i * args.interval:(i + 1) * args.interval]
            _ = sg.process_queries(new_queries)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Layout generator.')
    parser.add_argument('--config', default="demo", help="Config File Path")
    args = Args(parser.parse_args().config)
    # Fix the random seeds for our experiments
    np.random.seed(args.seed)

    # Partition per input file
    fnames, files, parts, config = setup_perfile(args)
    queries = get_workload_perfile(config, files, fnames, args)
    print("# queries: %d" % len(queries))
    dir_template = "%s/%s-%s-%d-%d-%s"
    for i, fname in enumerate(fnames):
        print(fname)
        df, df_sample, k = get_data(config, args, parts, files[i], fname)
        output_dir = dir_template % (config["ds"], fname, args.q, args.interval, k, args.method)
        gen_layout(queries, output_dir, k)



