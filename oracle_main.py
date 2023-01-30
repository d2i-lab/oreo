from utils.setup import *
from utils.tree import *
from utils.config import *
from offline.states import *
from online.counter import *
import numpy as np
import argparse
import os

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Gap to optimal comparison.')
    parser.add_argument('--config', default="demo", help="Config File Path")
    args = Args(parser.parse_args().config)
    args.set_policy('sw,oracle')

    fnames, files, parts, config = setup_perfile(args)
    fname = fnames[0]
    df, df_sample, k = get_data(config, args, parts, files[0], fname)
    queries = get_workload_perfile(config, files, fnames, args)
    alpha = args.alpha
    eps = args.eps
    print("alpha=%d, epsilon=%.3f" % (alpha, eps))
    qfile = args.q
    output_dir = "%s/%s-%s-%d-%d-%s" % (
        config["ds"], fname, args.q, args.interval, args.k, args.method)
    tb = TreeBuilder(df, df_sample, config, args, args.k, output_dir)

    # TPCH workload
    if qfile == "tpch1":
        n =[731, 2152, 2007, 2710,  572, 1529,  691,  683,  812,  539, 2421, 1973, 2628,
             2235, 799, 2559, 2090,  245, 1403, 1221]
        keys = ['q3', 'q14', 'q19', 'q7', 'q19', 'q14', 'q17', 'q21', 'q17', 'q5', 'q17', 'q12', 'q12',
                'q10', 'q6', 'q4', 'q19', 'q7', 'q8', 'q1']
        templates = ['q1', 'q10', 'q12', 'q14', 'q17', 'q19', 'q21', 'q3', 'q4', 'q5', 'q6', 'q7', 'q8']
    # TPCDS workload
    else:
        n = [759,  501, 1083, 941,  935, 1101, 1473,  387, 1340, 2745,  960,  514, 2253, 1919,
                2709, 2609, 2743,  765, 2466, 1797]
        keys = ['q27', 'q96', 'q53', 'q68', 'q7', 'q28', 'q46', 'q79', 'q13', 'q68', 'q19',
                'q36', 'q89', 'q7', 'q48', 'q36', 'q98', 'q34', 'q88', 'q3']
        templates = ['q13', 'q19', 'q27', 'q28', 'q3', 'q34', 'q36', 'q46', 'q48', 'q53', 'q68',
                     'q7', 'q79', 'q88', 'q89', 'q96', 'q98']

    schedule = {}
    schedule["ro"] = pickle.load(open(
                "resources/schedule/random/%s-%s-%s-%d-%s-%d-oracle-%d.p" % (config["ds"],
                    fname, qfile, k, args.method, args.alpha, args.gamma), "rb"))
    schedule["random"] = pickle.load(open(
                "resources/schedule/random/%s-%s-%s-%d-%s-%d-%.2f-%d.p" % (config["ds"],
                    fname, qfile, k, args.method, args.alpha, args.eps, args.gamma), "rb"))
    schedule["offline"] = pickle.load( open("resources/schedule/offline/%s-%s-%s-%d-%s.p" % (
                config["ds"], fname, qfile, k, args.method), "rb"))

    results = pickle.load(open("resources/schedule/oracle/%s-%s-%d.p" % (
        config["ds"], qfile, args.k), "rb"))

    # Offline oracle that is allowed to move
    q_idx = 0
    query = []
    reorg = []
    for i in range(len(n)):
        qs = queries[q_idx:q_idx+n[i]]
        reorg.append(q_idx)
        q_idx += n[i]

        s_idx = templates.index(keys[i])
        path = '%s/%d.p' % (tb.dir["oracle"], s_idx)
        tree = tb.load_by_path(path)
        read, _ = tree.eval(qs, avg=False)
        query.extend(list(read))
    print("Query: %f, Movement: %f" % (sum(query), len(reorg) * alpha))
    results["online"] = [query, reorg]

    # Random and Random Optimal
    for baseline in ["random", "ro"]:
        print(baseline)
        results[baseline] = []
        query = []
        reorg = []
        s = schedule[baseline]
        t = list(s.keys())[0]
        print(t, s[t].keys())
        moves = s[t]["move"]
        moves.append([30000, ''])
        for j in range(1, len(moves)):
            path = moves[j-1][1]
            qs = queries[max(0, moves[j-1][0]):moves[j][0]]
            eval_tree = tb.load_by_path(path)
            read, _ = eval_tree.eval(qs, avg=False)
            query.extend(list(read))
            reorg.append(moves[j-1][0])
        results[baseline].append([query, reorg])
        pickle.dump(results, open("resources/schedule/oracle/%s-%s-%d.p" % (
            config["ds"], qfile, args.k), "wb"))

    # Offline
    for baseline in ["offline"]:
        print(baseline)
        results[baseline] = []
        query = []
        reorg = []
        moves = schedule[baseline]["move"]
        path = moves[0][1]
        eval_tree = tb.load_by_path(path)
        read, _ = eval_tree.eval(queries, avg=False)
        query.extend(list(read))
        print(len(query), len(read))
        reorg.append(0)

        results[baseline].append([query, reorg])
        pickle.dump(results, open("resources/schedule/oracle/%s-%s-%d.p" % (
                config["ds"], qfile, args.k), "wb"))







