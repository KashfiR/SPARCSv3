# SPARCS: Stable Permutative Adaptive Rank-Based Correlation Screening

High-dimensional gene signature discovery through permutation-based stable feature selection,
benchmarked against adaptive regularization, iterative screening, and stability selection.

## Overview

SPARCS is a feature selection method for high-dimensional regression and classification. It combines
rank-based Sure Independence Screening, nonlinear dependence detection through the Randomized
Dependence Coefficient, permutation calibration, batch Complementary Pairs Stability Selection, and
adaptive thresholding. It is compared here against Spearman-ISIS, CPSS-Spearman, and an Adaptive
Elastic Net trained by stochastic gradient descent (ASGDR/ASGDC).

This repository is the complete implementation behind the paper, including the manuscript source.

## Layout

| File | Contents | Paper section |
|---|---|---|
| `utils.py` | Spearman screening, RDC, elbow detection, residuals | 2.2.1 |
| `sparcs.py` | SPARCS, all nine algorithm steps | 2.2.1 |
| `isis_cpss.py` | Spearman-ISIS, CPSS-Spearman, Shah–Samworth error bound | 2.2.2, 2.2.3 |
| `asgd.py` | Adaptive Elastic Net (ASGDR, ASGDC) | 2.2.4 |
| `datasets.py` | Synthetic generators for datasets s1 and s2 | 2.3.1 |
| `preprocessing.py` | CuMiDa loading, cleaning, splitting | 2.3.2 |
| `evaluation.py` | LightGBM harness, recovery metrics, overlap metrics | 2.4 |
| `experiments.py` | Driver producing `results/` | 3 |
| `figures.py` | Rebuilds every figure and `table1.tex` from `results/` | 3 |
| `paper/SPARCS_paper.tex` | Manuscript source | — |

## Usage

```python
import pandas as pd, numpy as np
from sparcs import SPARCS
from isis_cpss import ISIS, CPSS
from asgd import ASGDR, ASGDC, select_by_importance

X = pd.DataFrame(...)          # samples x features
y = np.array(...)

selected = SPARCS(X, y, task='regression', max_features=100, verbose=True)
selected_isis = ISIS(X, y, task='regression', max_features=100)
selected_cpss = CPSS(X, y, task='regression', B=30, tau=0.6)

model = ASGDR(random_state=42).fit(X, y, feature_names=list(X.columns))
selected_asgd = select_by_importance(model, 100)
```

Full benchmark, one dataset per session:

```bash
pip install -r requirements.txt
python experiments.py --smoke                 # ~2 min, verifies the install

python experiments.py --dataset s1       --seeds 5
python experiments.py --dataset s2       --seeds 5
python experiments.py --dataset liver    --seeds 5 --liver    /path/to/Liver_GSE14520_U133A.csv
python experiments.py --dataset leukemia --seeds 5 --leukemia /path/to/Leukemia_GSE28497.csv

python experiments.py --status                # progress so far
python figures.py                             # figures + table1.tex
```

### Replication

Every configuration is repeated across seeds (5 by default). A seed controls the train/test split
and every selector's internal randomization; the underlying data is held fixed so that the same
feature universe is on offer in each replicate, which is what makes selection stability measurable.
Pass `--redraw-data` to also redraw the synthetic datasets per seed — that widens the error bars to
full Monte Carlo variability, but the synthetic feature sets are then incomparable across seeds and
stability is reported for the real datasets only.

Reported values are means with a *t*-based 95% confidence interval (`--confidence` to change the
level, `figures.py --errorbar std` to draw one standard deviation instead). `figures.py` also emits
`fig6_selection_stability.png`, the mean pairwise Jaccard between each method's own selections
across seeds.

### Interruptions and adding seeds

Every `(dataset, method, seed)` replicate is checkpointed to `results/checkpoints/` the moment it
finishes, so an interrupted run only repeats the replicate that was in flight, and re-running the
same command restores the rest from disk. Each checkpoint stores a fingerprint of the settings that
produced it, so a stale replicate from a different problem size or a changed method parameter is
recomputed rather than silently mixed into the aggregate.

Seeds are incremental: running `--seeds 3` and later `--seeds 5` computes only the two new
replicates. `results/results.json` is rewritten after every session with whatever is complete, and
`figures.py` works on partial results. Pass `--fresh` to ignore checkpoints entirely.

## Parameters

**SPARCS** — `task`, `initial_screening_size` (defaults to ⌈n/log n⌉), `M_prefilter` (Spearman
prefilter width), `M_rdc` (RDC pool cap), `n_perm`, `alpha`, `min_candidates`, `B_stability` (CPSS
pairs), `stability_tau` (threshold floor), `k_add`, `rdc_threshold`, `shrink_rate`, `min_pool`,
`max_features`.

**ASGDR/ASGDC** — `l1_ratio`, `alpha`, `gamma`, `ridge_alpha`.
**ISIS** — `task`, `max_features`.
**CPSS** — `B`, `tau`, `max_features`.

Benchmark settings live at the top of `experiments.py` and are copied into `results.json`.

Two constraints worth knowing. The permutation floor is `1/(1 + n_perm)`, so `n_perm` must be at
least `1/alpha - 1`; the default 30 supports `alpha = 0.05`. And the RDC bandwidth `s` (in
`utils.rdc`, default 1.0) trades sensitivity to high-frequency dependence against the level of the
null — raising it detects oscillatory relationships but lifts the independent-pair baseline, which
matters because `rdc_threshold` is an absolute cutoff.

## Datasets

- Leukemia GSE28497, multiclass, 7 subtypes
- Liver HCC GSE14520, binary
- Synthetic s1 (20% nonlinear) and s2 (80% nonlinear), regression, known ground truth

The two real datasets come from [CuMiDa](https://sbcb.inf.ufrgs.br/cumida) and are not redistributed
here. The synthetic datasets are seeded and reproduce exactly.

## Reproduction status

The manuscript's methods sections describe this code exactly. The results table and figures must be
regenerated: the numbers in the June 2026 draft were produced by an earlier implementation that
differed in ways that affect results. `paper/table1_previous_run.tex` preserves those numbers for
reference and is deliberately not input by the manuscript. Running `experiments.py` followed by
`figures.py` and copying `results/figures/` next to the `.tex` file fills in the table and all five
figures.

## License

MIT
