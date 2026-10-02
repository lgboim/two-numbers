# two_numbers

A small tool for one question: **when a target is grouped, how much of a headline score does the group alone give, and what does a representation add below the group?**

It reports two numbers next to any published headline *H*:

- **B, the group baseline**: a predictor that knows only each unit's group (the training-set mean of its group), scored with the same metric as *H* (R², Pearson, Spearman or AUROC). If *B* reaches *H*, the headline alone shows nothing beyond the group.
- **CPG, the conditional predictive gain**: `1 - SSE(full) / SSE(group mean)` on held-out rows, where the full predictor adds the representation to the group mean. It comes with a 95% row-bootstrap interval, a whole-group interval (when there are at least 20 groups), an exact split into a cell-mean part and a within-cell part, a within-group permutation test, a joint (group-centred) fit on the feature route, and an optional resolution profile across grouping levels.

An interval that includes zero is not evidence that information is absent.

## Install

```
pip install -r requirements.txt
```

## Use from Python

```python
import two_numbers as tn
# y: target (n,) or (n, k); g: group labels (n,); train: boolean mask (n,)
R = tn.report(y, g, train, X=features)            # fits the probes itself
R = tn.report(y, g, train, pred=predictions)      # or uses ready out-of-sample predictions
print(tn.format_report(R))
```

## Use from the command line

```
python two_numbers.py data.csv --y lat,lon --group country --test is_test --pred p_lat,p_lon --metric r2
```

## Example and tests

```
python examples.py     # three synthetic cases with known answers: knowledge within groups, group only, a finer category
python tests.py        # edge cases: train-only means, perfect baseline, missing labels, inner-fold isolation
```

## How it fits the probes

With features, the standard probe is a ridge regression on the target. The residual probe is a ridge regression on the target minus the group mean; its penalty is chosen by 5-fold cross-validation in which the group means (and, for the joint fit, the feature centring) are recomputed from each inner training fold only. Test rows of groups unseen in training use the global training mean, and are counted in the report.

## What it does and does not do

- `pred` must be aligned **total** predictions of the target for every row (only test rows are scored), not residuals.
- The joint (group-centred) fit is available only on the feature route (`X`), not with ready predictions.
- The **CPG is always squared-error based**, even when the headline metric is Spearman, Pearson or AUROC; only *B* and *H* use the headline metric.
- With a multivariate target, the primary CPG is the mean of the per-axis ratios; the pooled version is reported beside it.
- The resolution profile scores the **same fixed predictions** against the means of other grouping levels; it does not refit a probe per level.
- `report` handles **one outer train/test split**. It is not designed for pooled predictions from several cross-validation folds, where group means differ by fold.
- Intervals condition on the fitted predictions (no refitting). If some bootstrap draws are undefined (for example, a resample whose baseline has zero error), they are excluded and their share is shown; an undefined interval is reported as such.
- The feature route needs at least 10 training rows.

## Reference

Elboim, A. (2026). *What Aggregate Scores Hide: Group Baselines for Fine-Grained Claims.* Preprints.org (the DOI will be added here when the preprint is posted).

Software archive: Zenodo, doi: [10.5281/zenodo.23110289](https://doi.org/10.5281/zenodo.23110289). Version 1.0.0. See `CITATION.cff`.

License: MIT.
