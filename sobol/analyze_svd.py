"""Sobol sensitivity indices for the two-part structural-virality decomposition.

Runs the canonical sobol/analyze.py machinery unchanged, but on the decomposed outcome
metrics: the ignition rate P(cascade size >= 2) and the conditional structural virality
E[SV | spread] of spreading cascades, both produced per simulation by
analysis/sv_decomposition.py and merged into the *_svd sweep copies by --write-sobol-csv.
Answers whether the dominance of fake-news prevalence over the pooled
avg_structural_virality operates through ignition or through cascade shape. Simulations
with no spreading cascade have undefined conditional virality; the design-point replicate
mean skips those NaNs (their count is reported by sv_decomposition.py).

Usage (run in the Sobol-SA environment, numpy<2, see requirements.txt):
  python sobol/analyze_svd.py --sweep data/baseline_svd \
      --problem sobol/problem.json --out sobol/results/baseline_svd_<date>
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import analyze

analyze.OUTPUTS = ["ignition_rate", "cond_sv"]
analyze.OUTPUT_LABELS = {"ignition_rate": "Ignition\nrate",
                         "cond_sv": "Conditional\nstruct. virality"}

if __name__ == "__main__":
    analyze.main()
