"""Filesystem paths, experiment-design constants, the paired-sweep loader, and streaming / RAM
helpers shared across the analysis scripts.

Data facts and IO only -- no statistics (those live in :mod:`utils`). Importing this module is
inexpensive: pyarrow/psutil are imported lazily inside the helpers that need them.
"""
import os
import sys

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
SVDECOMP_DIR = os.path.join(RUNS_DIR, "svdecomp_2026_07_09")
TOST_DIR = os.path.join(RUNS_DIR, "tost_blocks_2026_07_04")
SWITCHOVER_DIR = os.path.join(RUNS_DIR, "switchover_audit_2026_07_04")
MEANFIELD_DIR = os.path.join(RUNS_DIR, "mean_field_2026_07_09")

# The two paired sweeps (uniform-random vs top-degree hub seeding); identical Saltelli
# design and per-simulation seeds, so simulations match one-to-one on global_sim_id.
BASELINE_SWEEP = os.path.join(DATA_DIR, "sweep_2026_06_27_1641_baseline")
INFLUENCER_SWEEP = os.path.join(DATA_DIR, "sweep_2026_06_27_1633_influencer")
BASELINE_SVD_CSV = os.path.join(DATA_DIR, "sweep_2026_06_27_1641_baseline_svd", "simulations.csv")
INFLUENCER_SVD_CSV = os.path.join(DATA_DIR, "sweep_2026_06_27_1633_influencer_svd", "simulations.csv")

# ------------------------------------------------------------------ design constants
N_PAIRS, N_DESIGNS, N_BLOCKS = 491_520, 16_384, 1_024
BLOCK, REPS, PAIRS_PER_BLOCK = 16, 30, 480
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


def ram_guard(min_avail_gb=1.0):
    """Abort cleanly (exit code 3) if available system RAM drops below ``min_avail_gb`` GiB,
    turning a near-OOM condition into a controlled stop rather than an abrupt termination."""
    import psutil
    avail = psutil.virtual_memory().available / 2 ** 30
    if avail < min_avail_gb:
        print(f"ABORT: available RAM {avail:.2f} GiB < {min_avail_gb} GiB", flush=True)
        sys.exit(3)


def open_cascades(sweep_dir):
    """Memory-map ``<sweep_dir>/cascades.arrow`` (zero-copy) and return ``(batch, n_rows)``."""
    import pyarrow as pa
    path = os.path.join(sweep_dir, "cascades.arrow")
    batch = pa.ipc.open_file(pa.memory_map(path, "r")).get_batch(0)
    return batch, batch.num_rows
