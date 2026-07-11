# Fake-news percolation

An agent-based model (ABM) of how fake news, including deepfake-style content, spreads, is
verified, and is shared across a social network, and how that interacts with users' verification
cost, reputation concerns, bounded rationality, and loss aversion. This branch holds the code
and the frozen intermediate data behind the manuscript "Influencers, bounded rationality, and
the spread of fake news in social networks" (July 2026): the Julia simulation model, the
experiment runners, and the Python analysis and figure pipeline. The simulation engine is the
deepfake-percolation model of Dupont, C. (2026), written in Julia
(https://github.com/charlesaugdupont/deepfake-percolation), extended here with two paired
seeding runners, `run_experiment_baseline.jl` (uniform-random seeding) and
`run_experiment_influencers.jl` (hub seeding). The initial Python/Mesa prototype mentioned in
the manuscript is on the `main` branch.

The repository supports two downstream analyses:

1. **Global sensitivity analysis (Sobol).** Variance-based attribution of five outcome metrics
   (`veracity_differential`, `avg_verify_rate`, `avg_payoff`, `sen_welfare`,
   `avg_structural_virality`) to the seven model inputs, with a deterministic/stochastic
   decomposition (Approach IV).
2. **Seeding comparison.** A paired comparison of two cascade-seeding rules, uniform-random versus
   high-degree "influencer" hubs, on welfare, belief calibration, and misinformation reach.

## Reproduce everything

The repository is self-contained: it bundles the frozen intermediate data (~350 MB of
per-simulation tables and bounded-memory accumulators) from which every manuscript figure,
table, and quantitative claim is computed. One command rebuilds them all:

```bash
bash reproduce_analysis.sh
```

Runtime is roughly half an hour on a laptop (the paired comparison in `build_evidence.py
compare` is the slowest step at about 15 minutes), with a few GB of RAM. Each compute step
self-verifies against published reference numbers through internal assertion gates (the TOST
verifier's cross-checks, the structural-virality criteria V1-V7, the switchover gates S1-S7,
and the loss-edge mean assertions); a failed gate aborts the run. Outputs land in `analysis/runs/`,
`analysis/figures/`, and `sobol/results/`, which are all gitignored, so a reproduction run
leaves the working tree clean.

Two Python environments are needed (see **Requirements**). By default the script uses `python`
on PATH as the main stack; the Sobol step needs a second interpreter:

```bash
PY_MAIN="conda run -n <main-env> --no-capture-output python" \
PY_SOBOL="conda run -n <salib-env> --no-capture-output python" bash reproduce_analysis.sh
```

If `PY_SOBOL` is unset, the Sobol step is skipped with a notice. The raw per-node and
per-cascade Arrow tables (~174 GB) are not bundled; the optional `RUN_JULIA_SWEEP=1` and
`STREAM_FROM_ARROW=1` tiers of the script regenerate them from the model and re-stream the
caches from them (see section A).

## Re-running the simulation itself (scaled tiers)

Full-fidelity re-simulation is HPC-scale (measured 4.5-5 h per seeding configuration on a
192-core node; roughly two weeks on an 8-core laptop), so `reproduce.sh` offers scaled tiers
that re-run the **unchanged model** at reduced sweep breadth and then push the reviewer's own
sweep through the complete analysis chain:

```bash
bash reproduce.sh            # detects cores, prints ETAs, suggests a tier, asks to confirm
bash reproduce.sh --tier quick --yes
```

Only the Saltelli base sample N and the replicate count R shrink; network size (300 agents),
cascades per simulation (2,000), burn-in (1,000, MSER-validated), and every model parameter
stay at their published values. The scaled design is the first `16 N` rows of the committed
`sobol/design.csv`, which is byte-identical to the Saltelli sample SALib generates for that N
with the same seed, so a scaled run visits a verbatim subset of the published design with the
same common-random-number pairing.

| Tier | N | R | Simulations | What it can check |
|---|---|---|---|---|
| `smoke` | 8 | 2 | 512 | the pipeline end-to-end (statistics too thin to interpret) |
| `quick` | 128 | 2 | 8,192 | the flagship reach-equivalence TOST (128 clusters) and the S1/ST factor ranking |
| `overnight` | 1024 | 2 | 65,536 | everything at full design resolution: all prevalence bins, the switchover profile, S2 interactions; the flagship TOST standard error is only ~5% above the published run |
| `stochastic` | 1024 | 8 | 262,144 | additionally the stochastic-Sobol factor attributions |

ETAs are printed at launch from a deliberately conservative throughput constant measured on
the saturated production nodes (6.6 s per simulation per thread); a 16-thread desktop ran the
`smoke` tier's simulations in under a minute, and every run prints its own measured constant.
Each run is self-contained under `scaled_runs/<tier>_<tag>/` (its own sliced design, sweep,
caches, evidence JSONs, figures, and Sobol tables) and touches nothing outside it.

Two caveats, by design. First, scaled runs produce statistically valid but wider-CI numbers:
compare them to the manuscript qualitatively (dominant factors and their ordering, the TOST
conclusion, the direction and shape of the switchover profile), not digit-for-digit; the
exact-value reproduction gates and anchor cross-checks apply only to the canonical full-scale
data and are skipped with a notice on scaled runs. Second, no laptop-sized tier reaches the
manuscript's designed power (alpha = 0.01, power = 0.95 at the SESOI); that requires the full
491,520-pair sweep.

## Relation to the manuscript

Where each headline result is produced; steps refer to `reproduce_analysis.sh`:

| Manuscript item | Produced by | Step |
|---|---|---|
| Table 3 (Sobol indices) | `sobol/analyze.py` on the bundled `data/*_svd` tables | 7 |
| Table 4 (paired seeding contrasts) | `analysis/build_evidence.py compare` | 5 |
| Table 5 (influencers vs ordinary agents) | `analysis/build_evidence.py influencer` | 5 |
| Figure 4 (fake-reach equivalence, TOST) | `analysis/tost_blocks.py`, rendered by `analysis/make_figures.py` | 1, 6 |
| Figure 5 (prevalence profile of the seeding effect) | `analysis/robustness/switchover_audit.py`, rendered by `analysis/make_figures.py` | 3, 6 |
| Ignition-versus-shape decomposition | `analysis/sv_decomposition.py` (ignition-rate Sobol indices: `sobol/analyze_svd.py`) | 2, 7 |
| Mean-field benchmark (Appendix A) | `analysis/mean_field.py` | 4b |
| Burn-in adequacy and replication counts | `run_experiment_mser_probe.jl` + `analysis/replication_justification.py` | 4c |

The descriptive figures (cascade sizes, verification behaviour, calibration) come from the
evidence JSONs built by `analysis/build_evidence.py` (step 5) and are rendered by
`analysis/make_figures.py` (step 6).

## Repository layout

| Path | Contents |
|---|---|
| `reproduce.sh` | Resource-adaptive entry point: suggests a reproduction tier from the detected hardware and runs it (see **Re-running the simulation itself**). |
| `reproduce_analysis.sh` | One-command reproduction of every figure, table, and quantitative claim from the bundled data. |
| `run_experiment_baseline.jl` | Baseline sweep: 7-factor Saltelli/Sobol design, uniform-random cascade seeding. |
| `run_experiment_influencers.jl` | Influencer sweep: the same 7-factor design, hub seeding. |
| `run_experiment_mser_probe.jl` | Burn-in adequacy probe (MSER-5): logs every cascade for a stratified subset of design points. |
| `run_experiment_scaled.jl` | Scaled reviewer sweep: the unchanged model on a prefix of the committed design with fewer replicates; driven by `reproduce.sh`. |
| `scaled_runs/` | Self-contained scaled-tier run directories (disposable output; not committed). |
| `run_snellius_influencers.sh`, `run_snellius_baseline.sh` | SLURM job scripts for the influencer and baseline sweeps on the Snellius HPC cluster. |
| `Project.toml`, `Manifest.toml` | Julia environment (pinned to Julia 1.12.6). |
| `sobol/` | Sobol design generation (`make_design.py`), the committed deterministic design (`design.csv`), analysis (`analyze.py`, `analyze_svd.py`), sweep augmentation (`augment_sv.py`, adds the `avg_structural_virality` output to fresh sweeps), and the regenerated `results/`. |
| `analysis/` | Python evidence and figure pipeline: pure-compute scripts (`build_evidence.py`, `sv_decomposition.py`, `tost_blocks.py`, `mean_field.py`, `replication_justification.py`) write evidence JSONs and `make_figures.py` renders every figure; `build_evidence.py`'s `stream` subcommands reduce the large Arrow tables to per-simulation caches for bounded-memory runs; shared libraries live in `lib/` (`utils.py`, `plotstyle.py`, `data_io.py`, `paired_stats.py`); robustness checks in `robustness/`; `influence_identification.jl` checks that the top-degree hub seed pool matches k-shell and Collective Influence node rankings. |
| `analysis/runs/` | Bundled frozen inputs: the node/cascade cache (`nodecache/`), the exact structural-virality and switchover accumulators, the MSER burn-in probe, and two frozen cross-check JSONs. Regenerated outputs written next to them stay gitignored. |
| `data/` | Bundled per-simulation tables under clean names: `baseline/` and `influencer/` (`simulations.arrow`) plus `baseline_svd/` and `influencer_svd/` (gzipped per-simulation CSVs including `avg_structural_virality`). Fresh runner sweeps land in `data/sweep_<timestamp>_*/` (not committed). |
| `requirements.txt` | Python dependencies. |

## Model factors

The seven swept inputs (`sobol/problem.json`):

| Factor | Paper symbol | Range | Meaning |
|---|---|---|---|
| `v_cost` | c | 0.01–1.0 | verification cost |
| `loss` | ℓ (P_f) | 1.0–3.0 | reputation loss from sharing fake news |
| `tpr` | TPR | 0.6–0.99 | detector true-positive rate |
| `fpr` | FPR | 0.01–0.4 | detector false-positive rate |
| `p_fake` | p_fake | 0.05–0.95 | prior probability a cascade is fake |
| `log10_lambda` | log10 λ | −2.0–2.0 | bounded-rationality softmax precision (back-transformed as 10^x) |
| `loss_aversion` | λ_LA | 1.0–3.0 | prospect-theory loss-aversion coefficient (1.0 = loss-neutral) |

Both seeding sweeps vary all seven factors over the same Saltelli design: 16,384 design points
with 30 replications each, or 491,520 simulations per seeding arm, paired one-to-one by
`global_sim_id`.

## Requirements

**Python.** Two dependency stacks are needed because they are mutually incompatible in one
environment (`SALib 1.4.8` predates NumPy 2); see `requirements.txt`.

- *Main* (evidence building + figures): create a fresh environment and install the pinned NumPy-2
  stack:

  ```bash
  pip install -r requirements.txt
  ```

- *Sobol sensitivity analysis* (optional): create a **separate** environment with `numpy < 2` and
  install the pinned set in the commented block of `requirements.txt`.

**Julia 1.12.6** (pinned in `Manifest.toml`; packages: Arrow, DataFrames, DataStructures, Graphs,
ProgressMeter, StatsBase). Only needed to re-run the simulations themselves; the bundled-data
reproduction is pure Python. Instantiate once:

```bash
julia --project=. -e 'using Pkg; Pkg.instantiate()'
```

## Data

The bundled seeding-comparison data live under `data/` with clean names: `baseline/` and
`influencer/` hold the small per-simulation `simulations.arrow` summaries, and `baseline_svd/`
and `influencer_svd/` hold the gzipped per-simulation CSVs that carry the
`avg_structural_virality` column. The two arms share the same Saltelli design and
per-simulation seeds, so simulations match one-to-one on `global_sim_id`. The raw per-node and
per-cascade Arrow tables (~174 GB) are not bundled; regenerating them (section A) writes fresh
sweeps into `data/sweep_<timestamp>_*/`, which stay untracked.

## Reproduction

`bash reproduce_analysis.sh` (see **Reproduce everything**) runs the full fast tier from the
bundled data. The subsections below describe the pieces and the optional full regeneration.

### A. Regenerate the simulation sweeps (Julia, optional)

```bash
# Baseline: 7-factor Saltelli design, uniform-random seeding -> data/sweep_<timestamp>_baseline/
WRITE_FULL=true OUT_TAG=baseline julia --project=. -t auto run_experiment_baseline.jl

# Influencer sweep: same design, hub seeding
WRITE_FULL=true OUT_TAG=influencer julia --project=. -t auto run_experiment_influencers.jl
```

Both runners read the committed `sobol/design.csv`. By default only the per-simulation table is
written; `WRITE_FULL=true` also writes the per-node, per-edge, and per-cascade Arrow tables that
the cache re-streaming (`STREAM_FROM_ARROW=1`) and `sobol/augment_sv.py` need. These are
HPC-scale (491,520 simulations per arm); see the Snellius section. The burn-in length is
validated separately by `run_experiment_mser_probe.jl` together with
`analysis/replication_justification.py` (step 4c).

### B. Sobol global sensitivity analysis (Sobol-SA environment)

The bundled route (what `reproduce_analysis.sh` step 7 runs for both arms):

```bash
python sobol/analyze.py --sweep data/baseline_svd \
    --welfare-npz analysis/runs/nodecache/acc_nodes_baseline.npz \
    --problem sobol/problem.json --out sobol/results/baseline_sv
python sobol/analyze_svd.py --sweep data/baseline_svd \
    --problem sobol/problem.json --out sobol/results/baseline_svd
```

`--welfare-npz` replaces the bundled CSVs' superseded `sen_welfare` column with the
renormalised-Gini values from the node cache. `analyze_svd.py` reports indices for the ignition
rate; its conditional-SV rows are empty by construction (16 design points have no spreading
cascade in any replicate, so their design-point means are undefined), matching the reference
results. From scratch instead: regenerate the design with
`python sobol/make_design.py --N 1024` (deterministic, seed 42; N(2k+2) = 16,384 points), run
the ABM with `WRITE_FULL` (step A), add `avg_structural_virality` with
`python sobol/augment_sv.py data/sweep_<timestamp>` (main environment; writes
`data/sweep_<timestamp>_sv/`), and point `analyze.py --sweep` at the result.

### C. Seeding comparison: evidence and figures (main environment)

```bash
# The three evidence JSONs, reduced from the bundled node/cascade cache (< 2 GB RAM)
python analysis/build_evidence.py eda        --baseline data/baseline                              --cache analysis/runs/nodecache
python analysis/build_evidence.py compare    --baseline data/baseline --influencer data/influencer --cache analysis/runs/nodecache
python analysis/build_evidence.py influencer                                                       --cache analysis/runs/nodecache

# Render every figure whose inputs are present into analysis/figures/
python analysis/make_figures.py
```

The remaining pure-compute analyses (`sv_decomposition.py`, `tost_blocks.py`, `mean_field.py`,
`replication_justification.py`, and the checks in `robustness/`) write their outputs under
`analysis/runs/`; `reproduce_analysis.sh` steps 1-4c run them in the right order (run
`sv_decomposition.py reduce` before `mean_field.py`, which reads its evidence JSON). Their
frozen inputs, including the two cross-check JSONs `tost_blocks.py` compares against, are
bundled under `analysis/runs/`. On fresh sweeps, rebuild the cache first with the
`build_evidence.py` `stream` subcommands (`stream`, `stream-cascades`, `stream-influencer`,
`sample-cascade-sizes`) and pass its directory via `--cache`.

### Snellius (HPC)

```bash
sbatch run_snellius_influencers.sh true    # full influencer sweep, one 192-core Genoa node
sbatch run_snellius_baseline.sh true       # full baseline sweep, same node type
```

The `true` argument enables `WRITE_FULL`, which the downstream analyses require. The sweep jobs
request a 6-hour cap; actual runtime is not yet calibrated, and Snellius bills the elapsed time,
not the cap. Evidence and figures are then built locally from the finished sweeps (section C).

## Outputs

- `analysis/eda_evidence.json`, `analysis/compare_evidence.json`,
  `analysis/influencer_node_evidence.json`: statistical evidence (committed; refreshed by
  step 5).
- `analysis/figures/`: all figures, rendered in one pass by `make_figures.py` (gitignored).
- `analysis/runs/`: per-run outputs of the pure-compute analyses (gitignored, next to the
  bundled frozen inputs).
- `sobol/results/`: `sa_indices.csv`, `sa_indices_S2.csv`, `sa_indices.tex`, and per-output
  plots (gitignored).

## Notes

- The two seeding-comparison arms share the same 7-factor Saltelli design and per-simulation
  seeds; they differ only in the cascade seed pool (uniform-random nodes versus the top-degree
  hubs), so simulations are matched by `global_sim_id`.
- `sobol/design.csv` is committed and deterministic (seed 42); the Julia runners read it and do
  not modify it.

## Acknowledgements

We thank SURF (https://www.surf.nl) for the support in using the National Supercomputer Snellius.
