from utils.setup import *
from utils.tree import *
from utils.config import *
from offline.states import *
from offline.zorder import *
from online.regret import *
import numpy as np
import argparse


def run_regret(df, df_sample, out, args, k, queries):
    T = len(queries) // args.interval
    tb = TreeBuilder(df, df_sample, config, args, k, out)
    init_states = tb.get_init_states()
    sg = StateGenerator(tb, init_states, args.interval, epsilon=0, load=True)
    regret = Regret(tb, init_states, args.alpha, 1)
    for i in range(T):
        new_queries = queries[i*args.interval:(i+1)*args.interval]
        regret.process_queries(new_queries)
        new_states = sg.process_queries(new_queries)
        for s in new_states:
            regret.add_state(s)
    return regret.schedule, regret.query_cost, regret.movement_cost


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Regret baseline.')
    parser.add_argument('--config', default="demo", help="Config File Path")
    args = Args(parser.parse_args().config)
    # Fix the random seeds for our experiments
    np.random.seed(args.seed)

    fnames, files, parts, config = setup_perfile(args)
    queries = get_workload_perfile(config, files, fnames, args)
    print("#queries: %d" % len(queries))
    total_query = 0
    total_size = 0
    total_movement = 0
    for i, fname in enumerate(fnames):
        df, df_sample, k = get_data(config, args, parts, files[i], fname)
        N = len(df)
        output_dir = "%s/%s-%s-%d-%d-%s" % (config["ds"], fname, args.q, args.interval, k, args.method)
        schedule, q, m = run_regret(df, df_sample, output_dir, args, k, queries)
        total_size += N
        total_query += q * N
        total_movement += m * N
        print("[%s] Query: %f, Movement: %f" % (fname, q, m))
        pickle.dump(schedule, open("resources/schedule/regret/%s-%s-%s-%d-%s-%d.p" % (
            config["ds"], fname, args.q, k, args.method, args.alpha), "wb"))
    print("[Regret (%s,%d)] Query: %f, Movement: %f" % (
        args.policy, args.interval, total_query / total_size, total_movement / total_size))


