"""Filesystem paths, experiment-design constants, the paired-sweep loader, and streaming / RAM
helpers shared across the analysis scripts.

Data facts and IO only -- no statistics (those live in :mod:`utils`). Importing this module is
inexpensive: pyarrow/psutil are imported lazily inside the helpers that need them.
"""
import os
import time

import numpy as np
import pandas as pd

# ------------------------------------------------------------------ paths
ANALYSIS_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # this file lives in analysis/lib/
REPO_ROOT = os.path.dirname(ANALYSIS_DIR)
DATA_DIR = os.path.join(REPO_ROOT, "data")
RUNS_DIR = os.path.join(ANALYSIS_DIR, "runs")
SOBOL_DIR = os.path.join(REPO_ROOT, "sobol")
DESIGN_CSV = os.path.join(SOBOL_DIR, "design.csv")

# Latest run dirs holding the evidence that make_figures.py renders the structural-virality,
# equivalence, and switchover figures from (sv_decomposition.py / tost_blocks.py / switchover_audit.py).
SVDECOMP_DIR = os.path.join(RUNS_DIR, "svdecomp")
TOST_DIR = os.path.join(RUNS_DIR, "tost_blocks")
SWITCHOVER_DIR = os.path.join(RUNS_DIR, "switchover_audit")
MEANFIELD_DIR = os.path.join(RUNS_DIR, "mean_field")
# The streamed per-sim node/cascade cache that build_evidence.py reduces from by default.
NODECACHE_DIR = os.path.join(RUNS_DIR, "nodecache")

# The two paired sweeps (uniform-random vs top-degree hub seeding); identical Saltelli
# design and per-simulation seeds, so simulations match one-to-one on global_sim_id.
BASELINE_SWEEP = os.path.join(DATA_DIR, "baseline")
INFLUENCER_SWEEP = os.path.join(DATA_DIR, "influencer")
BASELINE_SVD_CSV = os.path.join(DATA_DIR, "baseline_svd", "simulations.csv.gz")
INFLUENCER_SVD_CSV = os.path.join(DATA_DIR, "influencer_svd", "simulations.csv.gz")

# ------------------------------------------------------------------ design constants
N_PAIRS, N_DESIGNS, N_BLOCKS = 491_520, 16_384, 1_024
BLOCK, REPS, PAIRS_PER_BLOCK = 16, 30, 480
N_NODES = 300                                         # agents per simulation = network size (Julia N_AGENTS)
N_CASC_PER_SIM = 1000                                 # recorded post-burn-in cascades per simulation
SESOI = 0.05                                          # fake-news reach equivalence band, raw units (TOST)
FACTORS = ["v_cost", "loss", "tpr", "fpr", "p_fake", "rationality", "loss_aversion"]
NODE_COLS = ["global_sim_id", "payoff", "best_payoff", "worst_payoff", "true_shares",
             "fake_shares", "tpr_alpha", "tpr_beta", "true_alpha", "true_beta"]  # nodes.arrow projection
CASC_METRICS = ["cascade_size", "max_depth", "seed_degree", "shares", "verifications"]
PFAKE_EDGES = np.linspace(0.05, 0.95, 10)             # nine equal-width prevalence bins
LAMBDA_EDGES = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])  # four log10-rationality bins
PUBLISHED_MEANS = (0.22555, 0.22817)                  # baseline, influencer avg_fake_cascade
MEAN_TOL = 5e-5


# ------------------------------------------------------------------ paired-sweep loader
def load_seeding_pairs(cols, base_csv=BASELINE_SVD_CSV, infl_csv=INFLUENCER_SVD_CSV,
                       check_means=True):
    """Load the paired baseline/influencer per-simulation tables aligned on ``global_sim_id``.

    Returns ``(a, b)`` sorted/reset DataFrames after the common-random-number alignment asserts
    (equal length N_PAIRS; identical global_sim_id and design_id vectors) and, when
    ``check_means``, the published-mean gate on ``avg_fake_cascade``. ``cols`` must include
    ``global_sim_id``, ``design_id``, and ``avg_fake_cascade``.
    """
    a = pd.read_csv(base_csv, usecols=cols).sort_values("global_sim_id").reset_index(drop=True)
    b = pd.read_csv(infl_csv, usecols=cols).sort_values("global_sim_id").reset_index(drop=True)
    assert len(a) == len(b) == N_PAIRS, (len(a), len(b))
    assert np.array_equal(a["global_sim_id"].values, b["global_sim_id"].values)
    assert np.array_equal(a["design_id"].values, b["design_id"].values)
    if check_means:
        mb, mi = float(a["avg_fake_cascade"].mean()), float(b["avg_fake_cascade"].mean())
        assert abs(mb - PUBLISHED_MEANS[0]) < MEAN_TOL and abs(mi - PUBLISHED_MEANS[1]) < MEAN_TOL, (mb, mi)
    return a, b


# ------------------------------------------------------------------ streaming / RAM helpers
if os.name == "nt":
    import ctypes
    _psapi = ctypes.WinDLL("psapi")
    _kernel32 = ctypes.WinDLL("kernel32")
    _kernel32.GetCurrentProcess.restype = ctypes.c_void_p
    _psapi.EmptyWorkingSet.argtypes = [ctypes.c_void_p]
    _psapi.EmptyWorkingSet.restype = ctypes.c_int

    def trim_working_set():
        """Return clean memory-mapped pages to the OS standby list. Windows trims working sets
        lazily, so a full multi-GB mmap scan otherwise drives 'available' RAM to zero and trips
        the guard. Explicit argtypes matter: the GetCurrentProcess pseudo-handle (-1) truncates
        without them and the call fails."""
        _psapi.EmptyWorkingSet(_kernel32.GetCurrentProcess())
else:
    def trim_working_set():
        pass


def ram_guard():
    """Throttle -- never abort -- when available system RAM falls below 10% of the machine's
    total, i.e. overall usage may reach at most 90% of the device's RAM. Machine-relative and
    re-evaluated at every check; there is deliberately no absolute threshold. On low memory the
    caller is paused: clean memory-mapped pages are returned to the OS and the check repeats
    once per second until availability recovers above the floor."""
    import psutil
    vm = psutil.virtual_memory()
    floor = 0.10 * vm.total
    waited = 0
    while vm.available < floor:
        if waited == 0:
            print(f"THROTTLE: available RAM {vm.available / 2**30:.2f} GiB < floor "
                  f"{floor / 2**30:.2f} GiB (10% of {vm.total / 2**30:.2f} GiB total); "
                  "pausing until memory frees", flush=True)
        trim_working_set()
        time.sleep(1)
        waited += 1
        vm = psutil.virtual_memory()
    if waited:
        print(f"THROTTLE: resumed after {waited}s (avail {vm.available / 2**30:.2f} GiB)", flush=True)


def open_arrow(path):
    """Memory-map an Arrow IPC file (zero-copy) and return ``(record batch, n_rows)``."""
    import pyarrow as pa
    batch = pa.ipc.open_file(pa.memory_map(path, "r")).get_batch(0)
    return batch, batch.num_rows


def stream_chunks(batch, n_rows, chunk_rows, label="", print_every=1):
    """Yield ``(offset, slice)`` over the first ``n_rows`` of a memory-mapped record batch in
    ``chunk_rows``-sized pieces: the RAM guard runs before each chunk, the working set is
    trimmed after it, and a progress line prints every ``print_every`` chunks (``label``
    prefixes the line; an empty label streams silently). The per-chunk accumulator logic
    stays in the caller; this generator owns only the walk/guard/trim/progress scaffold."""
    import psutil
    t0 = time.time()
    for i, off in enumerate(range(0, n_rows, chunk_rows)):
        ram_guard()
        n = min(chunk_rows, n_rows - off)
        yield off, batch.slice(off, n)
        trim_working_set()
        done = off + n
        if label and (i % print_every == 0 or done == n_rows):
            rate = done / max(time.time() - t0, 1e-9) / 1e6
            print(f"{label}: {done:,}/{n_rows:,} rows ({rate:.1f} M rows/s, "
                  f"avail {psutil.virtual_memory().available / 2**30:.1f} GiB)", flush=True)
