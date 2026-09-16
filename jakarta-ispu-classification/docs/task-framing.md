# Task Framing: Target Leakage and Task Asymmetry

This document explains two properties of the experiment that determine how the reported
results should be interpreted. Both were identified during post-hoc review of the completed
experiment. They are documented here rather than omitted, because a reader who does not know
about them will draw the wrong conclusion from the results table.

---

## 1. The target is a deterministic function of the features

### The ISPU calculation

The Indeks Standar Pencemar Udara is not a direct measurement. It is an index computed from
measured pollutant concentrations in two stages:

1. Each pollutant concentration is converted into a **sub-index** on a common 0–500 scale
   using piecewise-linear breakpoints defined by regulation.
2. The **daily ISPU value is the maximum across those sub-indices**, and the pollutant that
   produced it is designated the *critical* pollutant.
3. The **category** is assigned by fixed thresholds on that maximum:

| ISPU value | Category |
|---|---|
| 0 – 50 | BAIK |
| 51 – 100 | SEDANG |
| 101 – 200 | TIDAK SEHAT |
| 201 – 300 | SANGAT TIDAK SEHAT |
| > 300 | BERBAHAYA |

### What this dataset actually contains

In `ispu_dki_all.csv`, the columns `pm25`, `pm10`, `so2`, `co`, `o3` and `no2` hold
**sub-index values, already converted** — not raw concentrations in µg/m³. The `max` column
holds their maximum, `critical` names the argmax, and `categori` holds the resulting label.

This is directly verifiable in the raw file. Four rows from the head of the dataset:

| pm25 | pm10 | so2 | co | o3 | no2 | max | critical | categori |
|---:|---:|---:|---:|---:|---:|---:|---|---|
| NaN | 60 | 4 | **73** | 27 | 14 | 73 | CO | SEDANG |
| NaN | 32 | 2 | 16 | **33** | 9 | 33 | O3 | BAIK |
| NaN | **27** | 2 | 19 | 20 | 9 | 27 | PM10 | BAIK |
| NaN | **64** | 8 | 51 | 19 | 15 | 64 | PM10 | SEDANG |

In every row, `max` equals the largest sub-index present, `critical` names the column that
produced it, and `categori` is the threshold bucket of `max`. The relationship is exact.

### Why excluding `max` and `critical` is not sufficient

The notebook excludes `max`, `critical`, `stasiun`, `categori` and both target columns from
the feature set, and asserts this programmatically before training. That exclusion is correct
and necessary — but it removes only the *explicit* statement of the answer.

The binary target can be written as:

```
target = 1  if  max(pm25, pm10, so2, co, o3, no2) > 100
target = 0  otherwise
```

Every argument of that expression is in the feature matrix. A decision tree splitting on
`pm25 > 100`, `pm10 > 100`, and so on can express the disjunction

```
(pm25 > 100) OR (pm10 > 100) OR ... OR (no2 > 100)
```

which is logically equivalent to the rule. Random Forest and XGBoost are ensembles of exactly
such trees. Recovering this rule is well within their capacity, and 1,460 training rows is
far more than needed to do so.

### The evidence in the results

Three observations are consistent with rule reconstruction and difficult to explain otherwise:

**a) Two different algorithms produce byte-identical predictions.** Random Forest and XGBoost
have different loss functions, different regularisation, different tree-growth strategies and
different class-balancing mechanisms. They nonetheless produce the same Precision (0.9767),
the same Recall (0.9438) and the same confusion matrix (TN=275, FP=2, FN=5, TP=84). Two
estimators converging on an identical decision boundary is expected when the boundary is a
deterministic threshold; it would be a remarkable coincidence if they were estimating a noisy
statistical relationship.

**b) SHAP attribution is concentrated exactly where the rule predicts.** PM2.5 alone carries
about 62% of total mean absolute SHAP, and the six sub-indices together about 96%. Temporal
and missingness features are near zero. PM2.5 is the most frequent critical pollutant in
Jakarta, so a model implementing `max(...) > 100` should place most of its weight there.

**c) The residual errors trace to imputation.** Seven days out of 366 are misclassified in
2024 — precisely the year in which 20 PM10 values, 1 PM2.5 value and a handful of others were
imputed. Where an imputed value replaces the original that actually determined the published
category, the reconstructed rule can disagree with the label. In the absence of missingness
the expected error rate would approach zero.

### What this means for the reported scores

An F1 of 0.9600 on this task is **not** a claim that the model can assess Jakarta's air
quality. It is a measurement of how completely a model recovers a documented arithmetic rule
from data in which that rule is embedded. As an engineering result it is unremarkable; as a
methodological demonstration it is informative, because it shows how a target derived from
its own features can produce headline metrics that look like strong predictive performance.

### What would remove the leakage

Two designs would make this a genuine prediction problem:

- **Next-day forecasting.** Predict day *t*'s category from features at *t-1*, *t-2*, … The
  target is then not a function of the available features. This is what the LSTM and GRU
  models in this notebook already do — see section 2 below.
- **Raw concentration inputs plus exogenous variables.** Use measured concentrations together
  with meteorological data (wind speed and direction, temperature, humidity, rainfall,
  boundary-layer height). Even then, same-day concentrations remain close to the target; the
  meaningful version of the task is forecasting.

---

## 2. The two model families are not solving the same task

### The sequence construction

The sequence builder in the notebook is:

```python
def create_sequences_with_dates(X_array, y_array, date_array, window_size=14):
    X_seq, y_seq, target_dates = [], [], []
    for i in range(window_size, len(X_array)):
        X_seq.append(X_array[i - window_size:i])   # days i-14 ... i-1
        y_seq.append(y_array[i])                   # day i
        target_dates.append(date_array[i])
    return np.asarray(X_seq), np.asarray(y_seq), pd.to_datetime(pd.Series(target_dates))
```

Python slice semantics matter here. `X_array[i - 14 : i]` is **exclusive of index `i`**. The
window covers days *i-14* through *i-1*. Day *i*'s own features are never in the input.

So the recurrent models receive a 14-day history and must predict the following day's
category without seeing that day's pollutant readings. That is genuine next-day forecasting.

The tree-based models receive `X_test`, which is the target day's own feature row.

### The consequence

| | Random Forest / XGBoost | LSTM / GRU |
|---|---|---|
| Input | Target day's own 17 features | 14 preceding days, standardised |
| Target day's pollutants visible? | **Yes** | **No** |
| Target is a function of the input? | **Yes** (deterministic) | **No** |
| Effective task | Rule reconstruction | Next-day forecasting |
| Reported F1 | 0.9600 | 0.4839 (LSTM), 0.4000 (GRU) |

Placing 0.9600 and 0.4839 in the same column invites a conclusion that the data do not
support. The correct statement is:

> Under this experimental design, the tree-based models were given the information required
> to determine the label exactly, while the recurrent models were required to forecast it.
> The performance gap is a consequence of the input design, not a demonstration that one
> model family is superior on this problem.

### Standardisation compounds the difference

The recurrent models additionally receive `StandardScaler`-transformed inputs. Standardisation
is necessary for neural-network training, but it maps the interpretable threshold at 100 onto
a dataset-dependent z-score. The sharp, human-defined boundary that the tree models exploit
directly becomes, for the recurrent models, an arbitrary point in a continuous distribution
that must be learned. Combined with the sequential aggregation performed by the recurrent
layers, the threshold structure is substantially obscured.

### Were the recurrent models given a fair chance?

Not entirely, and this should be stated plainly. Specifically:

- **No hyperparameter search.** Window length (14), hidden units (64), dropout rates,
  learning rate and batch size were all fixed a priori without tuning.
- **Decision threshold fixed at 0.50.** For a class-weighted model on an imbalanced target
  with a train-to-test shift in base rate (36.1% → 24.3%), 0.50 is unlikely to be
  F1-optimal. Threshold tuning on the validation split would probably raise both LSTM and GRU
  scores without retraining.
- **No forecasting baseline.** A persistence baseline — predict day *t*'s class as equal to
  day *t-1*'s — is the natural reference point for a next-day task. Air quality is strongly
  autocorrelated, so persistence would likely be competitive. Without it, there is no way to
  tell whether an LSTM F1 of 0.4839 is poor, adequate, or respectable *for this task*.

These are limitations of the experiment, not of the architectures.

---

## 3. How to extend this work honestly

Ordered by value relative to effort:

1. **Add a persistence baseline** for the forecasting task, and a majority-class baseline for
   the classification task. Neither requires training. Both establish whether any reported
   score is meaningful.
2. **Add the rule baseline explicitly.** Classify directly on `max(sub-indices) > 100` with no
   model at all. If it approaches F1 = 1.0, the leakage argument is demonstrated numerically
   rather than argued analytically. `src/verify_leakage.py` in this repository does this.
3. **Tune the decision threshold** for LSTM and GRU on the validation split rather than
   fixing it at 0.50.
4. **Restrict the tree models to lagged features** so that all four models solve the same
   next-day forecasting task. This is the single change that would make the comparison valid.
5. **Adopt rolling-origin validation** — train on 2020–2021 → test 2022, train on 2020–2022 →
   test 2023, and so on — to replace a single test year with a distribution of estimates.
6. **Join meteorological covariates**, which is what would make the forecasting task
   genuinely tractable rather than merely well-posed.

---

## Summary

The experiment as executed is internally consistent, correctly implemented, and free of
procedural errors: the chronological split is respected, imputation is fitted on training data
only, class imbalance is handled, and excluded columns are asserted programmatically. The
issues documented here are not bugs.

They are framing issues, and they are the kind that only become visible after the results are
in. Recording them is what allows the numbers in this repository to be read correctly.
