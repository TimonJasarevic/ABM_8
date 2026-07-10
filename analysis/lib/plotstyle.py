"""Shared figure styling, brand palette, and manuscript-figure copy for the analysis figures.

Kept separate from :mod:`utils` (the citable statistical-methods library) so that stats-only
consumers need not import matplotlib. Import this module for the house Matplotlib style and the
UvA brand colours used across the figure scripts.
"""
import os
import shutil

import matplotlib.pyplot as plt
import seaborn as sns

# UvA brand hues and the Matplotlib default accents used across the figure scripts.
CRIMSON, UVABLUE = "#BC0031", "#1B6FBC"
BLUE, ORANGE, RED, GREY = "#1f77b4", "#ff7f0e", "#d62728", "#7f7f7f"
DARK = "#404040"
# Seeding-configuration colours (baseline = uniform-random, influencer = hub seeding).
ARM_COLORS = {"baseline": UVABLUE, "influencer": CRIMSON}

# The manuscript figures directory (.../report/JASSS.../figures), resolved from this file's
# location (analysis/lib/). Figure scripts copy their rendered PDFs/PNGs here when it exists.
MANUSCRIPT_FIG_DIR = os.path.abspath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "report",
    "JASSS__Agent_Based_Modelling_07_06_545pm", "figures"))


def set_pub_style(tick=13, label=15, legend=12, figsize=(8, 6)):
    """Install the house Matplotlib style shared by the figure scripts.

    Defaults reproduce ``make_figures.py`` exactly; ``tost_blocks.py`` calls
    ``set_pub_style(12, 14, 11, figsize=None)`` (it never set ``figure.figsize``). ``figsize=None``
    leaves the seaborn-whitegrid default in place.
    """
    sns.set_context("notebook", font_scale=1.2)
    plt.style.use("seaborn-v0_8-whitegrid")
    rc = {
        "font.size": 15, "figure.dpi": 100, "grid.alpha": 0.3, "axes.axisbelow": True,
        "mathtext.fontset": "cm",
        "xtick.labelsize": tick, "ytick.labelsize": tick,
        "axes.labelsize": label, "legend.fontsize": legend,
    }
    if figsize is not None:
        rc["figure.figsize"] = figsize
    plt.rcParams.update(rc)
    plt.rc("text", usetex=False)
    plt.rc("font", family="serif")


def copy_to_manuscript(paths):
    """Copy the given figure files into the manuscript figures directory when it exists."""
    if os.path.isdir(MANUSCRIPT_FIG_DIR):
        for f in paths:
            dst = os.path.join(MANUSCRIPT_FIG_DIR, os.path.basename(f))
            shutil.copy2(f, dst)
            print("COPIED ->", dst)
    else:
        print("NOTE: manuscript figures dir not found, skipped copy:", MANUSCRIPT_FIG_DIR)
