import requests
import click
import json
import subprocess
import sys
import tempfile
import tarfile
import pickle
import logging
import os
import collections

from itertools import groupby
from operator import itemgetter
from pathlib import Path

import pandas as pd
import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# We do this cause the models file can be directly called to run
sys.path.append(str(Path(__file__).parent.absolute()))

from internals.epmeans import ecdf, epmeans_cluster
from internals.stats import calc_ia_time

from internals.orcglobals import RAW_DATA_URL, RAW_DATA_DIR, NUM_DATA_FILES
from internals.orcglobals import FORMAT, DEFAULT_DAYS, CAPTURE_WINDOW
from internals.orcglobals import INVOCATION_CUTOFF, MODEL_DIR
from internals.stats import calc_invocs_list

from internals.run import get_jobs

raw_data_dir = RAW_DATA_DIR

logging.basicConfig(format=FORMAT, level=os.environ.get("LOGLEVEL", "INFO"))
log = logging.getLogger(__name__)


def graph(group, ax=None):
    fig = None
    if ax is None:
        fig, ax = plt.subplots()

    df = group["center"]
    log.info("Graphing CDFS of models")
    for m in group["functions"]:
        e = m["ecdf"]
        ax.plot(e.ecdf.x, e.ecdf.y, alpha=0.3)

    ax.plot(df.ecdf.x, df.ecdf.y, label="Centroid", color="black")
    ax.set_xscale("log")

    vals = []

    # Rough evaluation
    for c in group["functions"]:
        vals.append(np.mean(c["sub"]))
    log.info("Real ", np.mean(vals))
    s = group["center"].sample(100000)
    cm = np.cumsum(s)
    ma = int(max(cm) + 1)
    ra = np.arange(ma + 1, dtype="float32")
    bins, _ = np.histogram(cm, bins=ra)
    log.info("Model", np.mean(bins))

    return fig


def parse_data(days=DEFAULT_DAYS):
    # Tuples of prefixes for files as well as what we want to group the data on
    days_full = ["{}".format(str(x).zfill(2)) for x in range(1, 13)]
    days = ["{}".format(str(x).zfill(2)) for x in days]
    prefixes = [
        ("app_memory_percentiles.anon.d", "HashApp", days_full),
        ("function_durations_percentiles.anon.d", "HashFunction", days_full),
        ("invocations_per_function_md.anon.d", "HashFunction", days),
    ]

    # Grab all the files for specific data and read it in
    g = []
    total = (len(days_full) * 2) + len(days)
    timers_count = 0
    with click.progressbar(length=total, label="Loading Data") as bar:
        for pre in prefixes:
            files = [f"{pre[0]}{d}.csv" for d in pre[2]]
            invoc = []
            for i, f in enumerate(files):
                df = pd.read_csv(os.path.join(raw_data_dir, f))
                df["days"] = [i + 1] * len(df.index)
                # Filter away our timer triggers
                before = len(df)
                try:
                    df = df[df["Trigger"] != "timer"]
                except:
                    pass
                after = len(df)
                timers_count += before - after
                invoc.append(df)
                bar.update(1)

            con = pd.concat(invoc, axis=0, ignore_index=True)
            g.append(con.groupby(pre[1]))

    return g[0], g[1], g[2]


def download_data():
    global raw_data_dir
    if len(os.listdir(raw_data_dir)) == NUM_DATA_FILES:
        log.info("Data already downloaded and extracted, continuing..")
        return

    log.info("Downloading data")
    response = requests.get(RAW_DATA_URL)
    f = tempfile.NamedTemporaryFile()
    f.write(response.content)
    log.info("Extracting data")
    with tarfile.open(f.name, mode="r:xz") as tf:
        tf.extractall(path=raw_data_dir)
    f.close()


def aggregate_data():
    app, durations, invocations = parse_data()
    # Set up the invocation
    duration_keys = set([name for name, _ in durations])
    invocation_keys = set([name for name, _ in invocations])
    common = list(duration_keys.intersection(invocation_keys))
    log.info(f"{len(duration_keys)}, {len(invocation_keys)}, {len(common)}")
    functions = []

    exception_cont = 0
    with click.progressbar(length=len(common), label="Parsing Data") as bar:
        for group in common:
            bar.update(1)
            # Get timelines of each application.
            trigger = None
            hashApp = None
            hashFunction = None
            hashOwner = None
            t = [[0] * 1440] * len(DEFAULT_DAYS)
            for row in invocations.get_group(group).to_numpy().tolist():
                t[row[-1] - 1] = np.asarray(list(map(int, row[4:-1])), dtype="uint32")
                hashOwner = row[0]
                hashApp = row[1]
                hashFunction = row[2]
                trigger = row[3]

            t = np.asarray([i for sl in t for i in sl], dtype="uint32")

            # Aggregate data into one object
            d = dict(
                timeline=t,
                hash=hashFunction,
                owner=hashOwner,
                app=hashApp,
                trigger=trigger,
            )

            percentiles = [1, 25, 50, 75, 99, 100]
            percentiles_cpu = ["percentile_Average_{}".format(x) for x in percentiles]
            percentiles_ram = [
                "AverageAllocatedMb_pct{}".format(x) for x in percentiles
            ]
            retrieve_cpu = ["Average", "Minimum", "Maximum"] + percentiles_cpu
            retrieve_ram = ["AverageAllocatedMb"] + percentiles_ram
            cpu = []

            # Attach CPU and Memory data to the function
            try:
                for _, row in durations.get_group(group).iterrows():
                    cpu.append(np.array([row[x] for x in retrieve_cpu]))
                s = len(cpu)
                cpu = np.sum(cpu, axis=0) / s
                mem = []

                for _, row in app.get_group(hashApp).iterrows():
                    mem.append(np.array([row[x] for x in retrieve_ram]))
                s = len(mem)
                mem = np.sum(mem, axis=0) / s

                for ii, x in enumerate(retrieve_cpu):
                    d[x] = cpu[ii]
                for ii, x in enumerate(retrieve_ram):
                    d[x] = mem[ii]

            # If function does not have an associated app then discard it
            except:
                exception_cont += 1
                continue

            functions.append(d)

    return functions


def filter_and_ia(functions):
    for func in functions:
        calc_ia_time(func)

    return functions


def invert_and_ecdf(functions):
    # Cluster on our Frequency Data
    for func in functions:
        vals = ecdf(np.reciprocal(func["ia"], dtype="float32"))
        func["ecdf"] = vals
        del func["ia"]

    return functions


snapshot_one = raw_data_dir / "snapshot_one.pickle"
snapshot_two = raw_data_dir / "snapshot_two.pickle"
snapshot_three = raw_data_dir / "snapshot_three.pickle"
snapshot_three_b = raw_data_dir / "snapshot_three_b.pickle"
snapshot_four = raw_data_dir / "snapshot_four.pickle"


def s1(functions):
    download_data()
    functions = aggregate_data()
    with open(snapshot_one, "wb") as f:
        pickle.dump(functions, f)

    return functions


def split_and_collect(final, command, NUM_PER=200):
    global raw_data_dir
    funcs = [final[i : i + NUM_PER] for i in range(0, len(final), NUM_PER)]
    path = raw_data_dir / "tmp"
    path.mkdir(parents=True, exist_ok=True)
    this_file = os.path.abspath(__file__)
    procs = []
    files = []
    for i, f in enumerate(funcs):
        p = path / f"{i}.pickle"
        files.append(p)
        with open(p, "wb") as file:
            pickle.dump(f, file)
        log.info(p)
        os.sync()

        cmd = f"{sys.executable} {this_file} {command} {p}"
        proc = subprocess.Popen(
            cmd,
            shell=True,
            stdin=None,
            stdout=sys.stdout,
            stderr=sys.stderr,
            close_fds=True,
        )
        procs.append(proc)

    for i, p in enumerate(procs):
        p.wait()
        log.info(files[i])

    functions = []
    for f in files:
        with open(f, "rb") as file:
            functions.extend(pickle.load(file))

    for f in files:
        os.unlink(f)

    return functions


def s2(functions):
    log.info("Filtering and calculating interarrival times")
    final = []
    # Apply minimum invocation filter
    for func in functions:
        chunks = np.arange(720, len(DEFAULT_DAYS) * 1440, 1440)
        overall = []
        check = []
        for x in chunks:
            t = func["timeline"]
            overall.append(np.sum(t[x : x + CAPTURE_WINDOW]))
            check.extend(t[x : x + CAPTURE_WINDOW])

        # If there doesnt exist a window of time in any of the
        # days we are looking at the has at least INVOCATION_CUTOFF
        # invocations then we don't use it
        if not any([i > INVOCATION_CUTOFF for i in overall]):
            continue

        func["sub"] = check
        final.append(func)

    functions = split_and_collect(final, "ia")
    del final

    with open(snapshot_two, "wb") as f:
        pickle.dump(functions, f)

    return functions


def s3(functions):
    log.info("Calculate top 1%")
    weights = []
    for func in functions:
        weights.append(func["ia"].size)
    ninenine = np.percentile(weights, 99)

    functions_top = [func for func in functions if func["ia"].size >= ninenine]

    functions_bottom = [func for func in functions if func["ia"].size < ninenine]

    functions_bottom = split_and_collect(functions_bottom, "ecdf")
    functions_top = split_and_collect(functions_top, "ecdf")
    with open(snapshot_three, "wb") as f:
        pickle.dump((functions_top, functions_bottom), f)


def group_up(functions, centers, labels):
    log.info(collections.Counter(labels))
    log.info(len(centers))
    zipped = list(zip(functions, labels))
    zipped.sort(key=itemgetter(1))
    group = groupby(zipped, itemgetter(1))
    groups = []
    for key, data in group:
        d = {"center": centers[key], "functions": [d[0] for d in data]}
        groups.append(d)

    return groups


def s4(functions):
    log.info("Use EP-Means Clustering")
    functions_top = functions[0]
    functions_bottom = functions[1]

    functions_bottom = [f for f in functions_bottom if f["ecdf"] is not None]
    functions_top = [f for f in functions_top if f["ecdf"] is not None]

    data = [f["ecdf"] for f in functions_bottom]
    log.info("Use EP-Means Clustering low")
    centers, labels, error = epmeans_cluster(data, k=20)
    group_bottom = group_up(functions_bottom, centers, labels)

    data = [f["ecdf"] for f in functions_top]
    log.info("Use EP-Means Clustering high")
    centers, labels, error = epmeans_cluster(data, k=5)
    group_top = group_up(functions_top, centers, labels)

    group_bottom.extend(group_top)

    log.info("Snapshotting")
    with open(snapshot_four, "wb") as f:
        pickle.dump(group_bottom, f)

    return group_bottom


def create_models(data):
    global raw_data_dir
    global snapshot_one
    global snapshot_two
    global snapshot_three
    global snapshot_three_b
    global snapshot_four

    raw_data_dir = Path(data)
    snapshot_one = raw_data_dir / "snapshot_one.pickle"
    snapshot_two = raw_data_dir / "snapshot_two.pickle"
    snapshot_three = raw_data_dir / "snapshot_three.pickle"
    snapshot_four = raw_data_dir / "snapshot_four.pickle"

    functions = None
    retrieve_map = {
        snapshot_one: s1,
        snapshot_two: s2,
        snapshot_three: s3,
        snapshot_four: s4,
    }

    snaps = list(retrieve_map.keys())
    vals = list(retrieve_map.values())
    for cur in range(0, len(snaps)):
        if snaps[cur].exists():
            log.info(f"{snaps[cur]} exists")
            continue
        else:
            if (cur - 1) != -1:
                with open(snaps[cur - 1], "rb") as f:
                    functions = pickle.load(f)

            functions = vals[cur](functions)

    if functions == None:
        with open(snaps[-1], "rb") as f:
            functions = pickle.load(f)

    log.info("Graphing Models")
    models_dir = Path("graphs")
    models_dir.mkdir(parents=True, exist_ok=True)
    for i, group in enumerate(functions):
        fig = graph(group)
        fig.savefig(models_dir / f"{i}-model.svg")
        plt.close(fig)

    log.info("Create Models")
    for i, model in enumerate(functions):
        log.info("Creating - ", i)
        mbs = [func["AverageAllocatedMb"] for func in model["functions"]]
        cpu = [func["Average"] for func in model["functions"]]
        model["mbs"] = ecdf(mbs).inverse
        model["cpu"] = ecdf(cpu).inverse

    models_dir = MODEL_DIR
    models_dir.mkdir(parents=True, exist_ok=True)
    st = 720 + 2880
    end = st + CAPTURE_WINDOW
    evaluation = {}
    for i, group in enumerate(functions):
        k = f"{i}-model"
        p = models_dir / k
        model = group["center"]
        with open(p, "w") as f:
            m = dict()
            m["inverse_ecdf"] = model.inverse
            m["num_functions"] = len(group["functions"])
            m["mbs"] = group["mbs"]
            m["cpu"] = group["cpu"]
            json.dump(m, f)

        arr = np.zeros(30)
        for f in group["functions"]:
            t = np.asarray(f["timeline"][st:end])
            arr = np.add(arr, t)
        evaluation[k] = arr

    models_dir = Path("evaluation")
    models_dir.mkdir(parents=True, exist_ok=True)
    job_map = get_jobs(100, 1.0, 0)
    if job_map is None:
        log.error("Could not get models - are the models there?")
        exit(0)

    xs = np.arange(0, 30, 1)
    for name, jobs in job_map.items():
        p = models_dir / name
        trace = evaluation[name]
        arr = np.zeros(30)

        for j in jobs:
            t = np.asarray(calc_invocs_list(j.run_trace(30), 30))
            arr = np.add(arr, t)
        fig, ax = plt.subplots()

        ax.plot(xs, trace, label="Trace")
        ax.plot(xs, arr, label="Model")
        ax.legend()
        fig.savefig(p)


if __name__ == "__main__":
    if sys.argv[1] == "ecdf":
        with open(sys.argv[2], "rb") as f:
            functions = pickle.load(f)
            functions = invert_and_ecdf(functions)

        with open(sys.argv[2], "wb") as f:
            pickle.dump(functions, f)

    elif sys.argv[1] == "ia":
        with open(sys.argv[2], "rb") as f:
            functions = pickle.load(f)
            functions = filter_and_ia(functions)

        with open(sys.argv[2], "wb") as f:
            pickle.dump(functions, f)
