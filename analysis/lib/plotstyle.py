"""Shared figure styling and brand palette for the analysis figures.

Kept separate from :mod:`utils` (the citable statistical-methods library) so that stats-only
consumers need not import matplotlib. Import this module for the house Matplotlib style and the
UvA brand colours used across the figure scripts.
"""
import matplotlib.pyplot as plt
import seaborn as sns

# UvA brand hues and the Matplotlib default accents used across the figure scripts.
CRIMSON, UVABLUE = "#BC0031", "#1B6FBC"
BLUE, ORANGE, RED, GREY = "#1f77b4", "#ff7f0e", "#d62728", "#7f7f7f"
DARK = "#404040"
# Seeding-configuration colours (baseline = uniform-random, influencer = hub seeding).
ARM_COLORS = {"baseline": UVABLUE, "influencer": CRIMSON}


def set_pub_style(tick=13, label=15, legend=12, figsize=(8, 6)):
    """Install the house Matplotlib style shared by the figure scripts.

    Defaults reproduce ``make_figures.py``'s descriptive group exactly; its equivalence/TOST and
    switchover groups call ``set_pub_style(12, 14, 11, figsize=None)`` (those figures never set
    ``figure.figsize``). ``figsize=None`` leaves the seaborn-whitegrid default in place.
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
