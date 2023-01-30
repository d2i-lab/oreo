from utils.setup import *
from utils.config import *
import pyspark
from pyspark.sql import SparkSession
from e2e.partition import *
import pickle
import argparse
import time


def reorg(move_schedule, rewrite, fname=None):
    # Replay reorg
    reorg_time = []
    moves = []
    # Make initial layout (same across different algorithms)
    if fname is None:
        data_dir = config["path"]
        in_dir = "%s/%s/%d" % (root, config["ds"], 0)
    else:
        data_dir = join(config["path"], fname)
        in_dir = "%s/%s/%s-%d" % (root, config["ds"], fname, 0)
    labels = {}
    # Create layouts used
    indices = []
    for [t, path] in move_schedule:
        if t > 0:
            indices.append(t)
        print(t, path)
        label_path = path
        lid = 0
        if 'offline' in path:
            policy = 'offline'
        else:
            lid = int((path.split('/')[-1]).split('.')[0])
            policy = 'res'
            if 'sw/' in path:
                policy = 'sw'
        if not '-label' in label_path:
            label_path += '-label'
        bids = pickle.load(open(label_path, "rb"))
        if t == 0:
            if not os.path.exists(in_dir) and rewrite:
                pm.load_and_reorg_with_labels(bids, data_dir, in_dir, True)
            moves.append([0, in_dir])
            labels[label_path] = in_dir
            continue
        if label_path in labels:
            out_dir = labels[label_path]
        else:
            if fname is None:
                out_dir = "%s/%s/%s-%d" % (root, config["ds"], policy, lid)
            else:
                out_dir = "%s/%s/%s-%s-%d" % (root, config["ds"], fname, policy, lid)
            if args.method == 'z':
                out_dir += "-z"
            labels[label_path] = out_dir
            if rewrite and not os.path.exists(out_dir):
                t0 = time.time()
                pm.load_and_reorg_with_labels(bids, in_dir, out_dir, t == 0)
                t1 = time.time()
                reorg_time.append(t1 - t0)
                print("[T=%d] Creating layout %d in %f" % (t, lid, t1 - t0))
        if True:
            moves.append([t, out_dir])
    print("%s avg reorg time: %f" % (fname, np.average(reorg_time)))
    if rewrite:
        if fname is None:
            pickle.dump({"idx": indices, "time": reorg_time}, open("results/e2e/%s-%s-%s-%s-reorg.p" % (
                config["ds"], qfile, args.alg, args.method), 'wb'))
        else:
            pickle.dump({"idx": indices, "time": reorg_time}, open("results/e2e/%s-%s-%s-%s-%s-reorg.p" % (
                config["ds"], fname, qfile, args.alg, args.method), 'wb'))
    return moves, reorg_time


def query(query_schedule, moves, sqls, N, fname=None):
    query_time = []
    j = 1
    in_dir = moves[j - 1][1]
    np.random.seed(0)
    samples = sorted(np.random.choice(len(query_schedule), N, replace=False))
    print(samples[-10:])
    for q_idx in samples:
        while j < len(moves) and q_idx >= moves[j][0]:
            j += 1
            in_dir = moves[j-1][1]
        bids = query_schedule[q_idx]
        t0 = time.time()
        pm.run_query(in_dir, bids, sqls[q_idx])
        t1 = time.time()
        print(q_idx, in_dir, len(bids), t1-t0)
        query_time.append(t1 - t0)
    print("%s avg query time: %f" % (fname, np.average(query_time)))
    if fname is None:
        pickle.dump({"idx": samples, "time": query_time}, open("results/e2e/%s-%s-%s-%s-query.p" % (
            config["ds"], qfile, args.alg, args.method), 'wb'))
    else:
        pickle.dump({"idx": samples, "time": query_time}, open("results/e2e/%s-%s-%s-%s-%s-query.p" % (
            config["ds"], fname, qfile, args.alg, args.method), 'wb'))
    return query_time


def run(num_trails, fname):
    q = []
    m = []
    for trials in range(num_trails):
        if args.alg == "random":
            ds = config["ds"]
            if fname is not None:
                ds = fname
            move_schedule = schedule["%s-%d" % (ds, trials)]["move"]
            query_schedule = schedule["%s-%d" % (ds, trials)]["query"]
        else:
            move_schedule = schedule["move"]
            query_schedule = schedule["query"]

        # Reorg replay
        moves, reorg_time = reorg(move_schedule, args.rewrite, fname)
        # Query replay
        query_time = query(query_schedule, moves, sqls, args.n, fname)

        # Add cost
        q.append(np.average(query_time) * len(query_schedule))
        m.append(np.average(reorg_time) * (len(moves) - 1))

    return q, m


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Schedule replayer.')
    parser.add_argument('--config', default="demo", help="Config File Path")
    parser.add_argument('--rewrite', action='store_true')
    parser.add_argument('--n', type=int, default=2000)
    parser.add_argument('--alg', default='offline')
    parser.add_argument('--root', default="/mnt/1/partition")
    tmp = parser.parse_args()
    args = Args(tmp.config)
    args.rewrite = tmp.rewrite
    args.alg = tmp.alg
    args.m = tmp.n
    root = tmp.root
    qfile = args.q
    alpha = args.alpha
    eps = args.eps
    session = SparkSession.builder.config("spark.master", "local[4]").config("spark.driver.memory", "4g").getOrCreate()
    with open("resources/query/%s.sql" % qfile, "r") as file:
        sqls = [line.rstrip() for line in file]

    fnames, files, parts, config = setup_perfile(args)
    for i, fname in enumerate(fnames):
        k = parts[fname]
        pm = PartitionManager(session, k)
        # Load schedule
        if args.alg == "random":
            num_trails = 3
            schedule = pickle.load(open("resources/schedule/%s/%s-%s-%s-%d-%s-%d-%.2f-%d.p" % (
                args.alg, config["ds"], fname, qfile, args.k, args.method, alpha, eps, args.gamma), "rb"))
        else:
            num_trails = 1
            if args.alg == "offline":
                schedule = pickle.load(open("resources/schedule/%s/%s-%s-%s-%d-%s.p" % (
                    args.alg, config["ds"], fname, qfile, args.k, args.method), "rb"))
            else:
                schedule = pickle.load(open("resources/schedule/%s/%s-%s-%s-%d-%s-%d.p" % (
                    args.alg, config["ds"], fname, qfile, args.k, args.method, alpha), "rb"))

        # Main
        q, m = run(num_trails, fname)
        print("[%s] Query: %f, Movement: %f" % (fname, np.average(q), np.average(m)))














