# Pathwise Counterfactual Certification

Reproducible experiments for certifying that one independently retrained classifier
accepts every point in a prescribed finite path suffix. A last-failure statistic
turns this joint event into ordered Bernoulli outcomes. A shared mixture e-process
supports adaptive selection of a suffix and repeated inspection of the evidence.

## Contents

- `src/`: certificate implementation, model fitting, path construction, simulations,
  independent audits, sensitivity analysis, and reproduction runners.
- `configs/main.json`: dataset choices, training seeds, model budgets, path resolution,
  inspection times, and target/error probabilities.
- `data/`: eight numeric benchmark tables, original UCI downloads, and provenance.
- `results/banks/`: 144 paired path evaluation banks from 72 dataset–learner–split
  settings. Each bank contains 512 certification and 1,024 independent audit models.
- `results/simulation_trials.csv.gz`: 60,000 Monte Carlo experiments under exact
  probability laws, with records for eight certificate/comparison rules.
- `results/analysis/`: numerical summaries, query-level records, robustness profiles,
  dataset-cluster bootstrap intervals, and grid/target sensitivity results.
- `tests/`: checks of the mixture integral, null expectation, monotonicity, exact
  binomial levels, nominal eligibility, and last-failure query scan.
- `checksums.sha256`: integrity checks for the distributed files.
- `CITATION.cff`, `LICENSE`, and `DATA_LICENSE.md`: citation and licensing metadata.

## Environment

Use Python 3.12. The reference run used Python 3.12.1 on macOS with an Apple M3 Pro,
12 CPU cores, an 18-core integrated GPU, and 18 GB unified memory. The exact recorded
interpreter and platform details are in `results/environment.json`. Dependencies
are pinned in `requirements.txt`.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python src/verify_archive.py
python -m unittest discover -s tests -v
```

## Reproduce the numerical results

The archived prediction banks allow a complete deterministic replay of certificate
selection, audit estimates, Monte Carlo summaries, and sensitivity analyses.
The following command creates a fresh directory and compares every resulting CSV
against the reference results:

```bash
python src/reproduce.py --mode replay --workdir reruns/replay-1 --verify
```

A successful run prints the number of verified numerical files. Each invocation
uses a new work directory so the reference results remain available for comparison.
Replay takes several minutes on the reference laptop; it uses the archived fitted
model predictions and does not require GPU computation.

## Regenerate training and simulation experiments

```bash
python src/reproduce.py --mode full --workdir reruns/full-1 --device mps
```

This starts classical models with eight CPU workers and neural models in batches
of 32 on Apple's Metal backend, concurrently. It then evaluates the paired
coordinatewise paths, generates the 60,000 simulation experiments using four CPU
workers, and computes all numerical analyses. CPU and neural training logs are
saved in the chosen run directory. Use `--device cpu` on a computer without Metal.
All model draws are seeded; CPU libraries and GPU kernels can introduce small
floating-point differences across platforms. Exact result verification is provided
by replaying the archived prediction banks; a fresh fit quantifies training-level
reproducibility on the chosen platform. The reference computation uses modest
memory, so increasing parallelism beyond the default may give little benefit.

The original tables can also be downloaded and converted again:

```bash
python src/download_data.py
```

Source URLs and hashes are recorded in `data/provenance.json`.

## Experimental design

Eight datasets are crossed with logistic regression, histogram gradient boosting,
and a one-hidden-layer ReLU network, and three stratified train/test splits. Large
training sets are capped at 2,000 observations. A scaler is fitted to the retained
training table and frozen. The retraining law draws an ordinary bootstrap sample
from that table; neural draws also include independent initialization and minibatch
sampling. Hyperparameters are fixed before certification.

One nominal model and 32 search models determine each query's anchor. Certification
and independent audit draws use separate seeds. Up to 24 nominally negative test
queries per setting yield 1,725 query instances. Each has a 64-point linear path and
a coordinatewise path toward the same anchor, for 3,450 paired path–query cases.
Coordinates in the latter path change in descending absolute standardized distance.
The same retraining seed schedule is reused across datasets and both path types;
these paired records should not be treated as independent datasets.

The target is the probability that **one model accepts all points in a suffix**.
`tau=0.95` specifies that probability; `delta=0.05` controls erroneous certification
for each frozen query/path. A candidate must also have a fully accepted nominal
suffix. Path selection must use the search bank before certification outcomes are
inspected. The guarantee applies to the listed points and the declared retraining
law. Feasible and causal feature interventions require application-specific design.

### Methods

| Internal name | Rule |
|---|---|
| `persistent_anytime` | PCC: shared uniform-mixture e-process on whole-suffix outcomes |
| `persistent_bonferroni` | Same process with error budget divided by path length |
| `persistent_spending` | Exact binomial tests with summable error budgets across looks |
| `persistent_fixed` | Exact binomial suffix test at a prespecified model budget |
| `pointwise_fst` | Fixed-sequence marginal e-tests, starting at the endpoint |
| `pointwise_beta` | Fixed-budget Jeffreys-prior marginal posterior score |
| `persistent_repeated_cp` | Diagnostic uncorrected repeated fixed-sample tests |
| `persistent_naive` | Diagnostic empirical-success threshold |

Pointwise methods target marginal probabilities. Their path-error frequencies
measure a difference in target, rather than a violation of a joint guarantee.
Fixed-budget results at different budgets represent separate planned experiments.
Retained e-process crossings and alpha-spending crossings permit repeated looks.

### Datasets and labels

| File stem | Rows | Features | Positive class |
|---|---:|---:|---|
| `breast` | 569 | 30 | Benign |
| `ionosphere` | 351 | 34 | Good radar return |
| `sonar` | 208 | 60 | Rock |
| `banknote` | 1,372 | 4 | Original class 1 |
| `spambase` | 4,601 | 57 | Spam |
| `wine_red` | 1,599 | 11 | Quality at least 6 |
| `wine_white` | 4,898 | 11 | Quality at least 6 |
| `magic` | 19,020 | 10 | Gamma |

These labels define computational classification tasks. Dataset credits and
CC BY 4.0 terms appear in `DATA_LICENSE.md`.

## Result schema

Each compressed bank has `cert` and `evaluation` arrays of shape
`(models, queries, path_points)`, Boolean-valued predictions stored as unsigned
bytes. `nominal` has shape `(queries, path_points)`. `paths` stores standardized
feature vectors, and `cost` is mean absolute standardized displacement. Training,
test, query, and anchor indices refer to rows of the corresponding dataset table.
The `seeds` array orders the nominal, search, certification, and audit draws.
Adjacent JSON files report the configuration and recorded durations.

`benchmark_records.csv` contains each rule's selected index at each inspection,
its cost, and independent pointwise and suffix audit estimates. Index `-1` denotes
abstention; candidate indices are zero-based. `path_summary.csv` and
`method_summary.csv` aggregate those records. `accuracy_summary.csv` reports
independent-model predictive accuracy and variation across split means. `paired_costs.csv` compares costs on
jointly certified cases. `cluster_intervals.csv` uses 10,000 dataset-group resamples.
`robustness_profiles.csv.gz` records the entire independent-audit path profiles.
`sensitivity_records.csv.gz` and `sensitivity_summary.csv` repeat certification for
16, 32, and 64 retained points and target probabilities 0.90, 0.95, and 0.99.

`simulation_trials.csv.gz` records exact-law errors and final selected probabilities.
The `eta=-1` setting is the boundary-null design (shared success probability 0.949
before a deterministic endpoint); other `eta` values are notch probabilities.
`first_n=513` denotes no certificate by 512 draws. Simulation `index` values are
one-based, with `-1` for abstention. The configuration records every scenario seed.

## License and citation

Original code and documentation: MIT, copyright 2026 Cemil Emre Yavas.
Dataset license and creator attribution: `DATA_LICENSE.md`.
Use `CITATION.cff` for software citation. Contact: cemil@fairesresearch.org.
