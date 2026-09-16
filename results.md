# Experimental Results

All figures are taken from the executed notebook,
`notebooks/Klasifikasi_ISPU_Jakarta.ipynb`. Test set: 366 days in 2024, of which 89 are
Tidak Sehat+ (base rate 24.3%). Positive class throughout: **Tidak Sehat+**.

Before reading these numbers, see [`task-framing.md`](task-framing.md), which explains why
the tree-based and recurrent models are not solving the same problem.

---

## 1. Headline comparison

| Model | Input representation | Precision | Recall | F1-Score | ROC-AUC |
|---|---|---:|---:|---:|---:|
| Random Forest | Target day, tabular | 0.9767 | 0.9438 | **0.9600** | **0.9822** |
| XGBoost | Target day, tabular | 0.9767 | 0.9438 | **0.9600** | 0.9705 |
| LSTM | 14 preceding days | 0.4639 | 0.5056 | 0.4839 | 0.6902 |
| GRU | 14 preceding days | 0.4507 | 0.3596 | 0.4000 | 0.7159 |

Machine-readable: [`../results/model_comparison.csv`](../results/model_comparison.csv).

---

## 2. Confusion matrices

Derived from the reported metrics and the test-set composition (89 positives, 277 negatives).

| Model | TN | FP | FN | TP | Total errors |
|---|---:|---:|---:|---:|---:|
| Random Forest | 275 | 2 | 5 | 84 | **7** |
| XGBoost | 275 | 2 | 5 | 84 | **7** |
| LSTM | 225 | 52 | 44 | 45 | 96 |
| GRU | 238 | 39 | 57 | 32 | 96 |

Machine-readable: [`../results/confusion_matrices.csv`](../results/confusion_matrices.csv).

---

## 3. Per-model reading

### Random Forest

| Metric | Value |
|---|---:|
| Precision | 0.9767 |
| Recall | 0.9438 |
| F1-Score | 0.9600 |
| ROC-AUC | 0.9822 |

Seven errors in 366 days: 5 unhealthy days missed and 2 false alarms. Recall (0.9438) sits
below Precision (0.9767), meaning the model errs toward under-warning. For a public-health
application that asymmetry runs the wrong way — a missed unhealthy day is worse than a
spurious advisory — and would be corrected by lowering the decision threshold below 0.50.

The errors are concentrated in 2024's imputed days. Twenty PM10 values and one PM2.5 value
were reconstructed in the test year; where the imputed value straddles the ISPU = 100
threshold differently from the original, the reconstructed rule disagrees with the published
label.

### XGBoost

| Metric | Value |
|---|---:|
| Precision | 0.9767 |
| Recall | 0.9438 |
| F1-Score | 0.9600 |
| ROC-AUC | 0.9705 |

Identical hard predictions to Random Forest — the same 7 misclassified days — despite a
different loss, a different regularisation scheme, shallower trees (`max_depth=4`) and a
different class-balancing mechanism.

The two models diverge only in ROC-AUC (0.9822 vs 0.9705), which is threshold-independent and
therefore sensitive to probability calibration rather than to the decision boundary. The
boundary is the same; the confidence assigned to points near it is not.

This convergence is itself a result. Two distinct algorithms reaching an identical decision
boundary is expected when that boundary is a deterministic threshold present in the data. It
would be a substantial coincidence if they were estimating a noisy statistical relationship.

### LSTM

| Metric | Value |
|---|---:|
| Precision | 0.4639 |
| Recall | 0.5056 |
| F1-Score | 0.4839 |
| ROC-AUC | 0.6902 |

Detects 45 of 89 unhealthy days at the cost of 52 false alarms. ROC-AUC of 0.6902 is
meaningfully above chance (0.50), so the model has learned genuine signal from the 14-day
history — but the signal is weak relative to what same-day sub-indices provide.

This is the expected outcome for a next-day forecast. The model must anticipate tomorrow's
air quality from the past fortnight without meteorological covariates, which are the dominant
driver of day-to-day variation in pollutant dispersion.

### GRU

| Metric | Value |
|---|---:|
| Precision | 0.4507 |
| Recall | 0.3596 |
| F1-Score | 0.4000 |
| ROC-AUC | 0.7159 |

An informative contrast with LSTM. GRU has the **higher ROC-AUC** (0.7159 vs 0.6902) but the
**lower F1** (0.4000 vs 0.4839). Its probability ranking is better; its performance at the
fixed 0.50 threshold is worse, because it is systematically less confident and therefore
predicts the positive class less often (32 positive predictions vs LSTM's 97).

The practical implication: GRU is likely the better underlying model of the two, and its low
F1 is largely an artefact of leaving the threshold at 0.50. Tuning the threshold on the
validation split would probably reverse the F1 ordering. This was not done — see
[`limitations.md`](limitations.md).

---

## 4. Model selection for interpretation

Random Forest, on equal F1 and higher ROC-AUC. The margin is thin and the choice is not
load-bearing: XGBoost produces the same predictions, so nothing in the SHAP analysis depends
on which was selected.

---

## 5. SHAP attribution, test set 2024

Mean absolute SHAP values for the positive class, Random Forest, over 366 test days.

| Rank | Feature | Mean \|SHAP\| | Share of total |
|---:|---|---:|---:|
| 1 | PM2.5 | 0.323157 | 61.8% |
| 2 | PM10 | 0.083515 | 16.0% |
| 3 | O3 | 0.047695 | 9.1% |
| 4 | CO | 0.026628 | 5.1% |
| 5 | NO2 | 0.023424 | 4.5% |
| 6 | SO2 | 0.013650 | 2.6% |
| 7 | PM2.5 missing indicator | 0.008758 | 1.7% |
| 8 | Bulan | 0.007862 | 1.5% |
| 9 | Tahun | 0.007825 | 1.5% |
| 10 | PM10 missing indicator | 0.005726 | 1.1% |
| 11 | Pasca-pandemi indicator | 0.002825 | 0.5% |
| 12 | Hari dalam minggu | 0.001205 | 0.2% |
| 13 | SO2 missing indicator | 0.001089 | 0.2% |
| 14 | Indikator akhir pekan | 0.000376 | 0.1% |
| 15 | CO missing indicator | 0.000095 | <0.1% |
| 16 | O3 missing indicator | 0.000030 | <0.1% |
| 17 | NO2 missing indicator | 0.000007 | <0.1% |

Full table: [`../results/shap_feature_importance_test2024.csv`](../results/shap_feature_importance_test2024.csv).

### Reading

**Attribution is overwhelmingly concentrated in the pollutant sub-indices** — about 96% of
total mean absolute SHAP across the six of them, with PM2.5 alone at roughly 62%. Temporal
features contribute about 3.7% combined, and missingness indicators about 3.0%.

This is precisely the profile the ISPU rule predicts. Since the category is the threshold
bucket of the maximum sub-index, and PM2.5 is the sub-index most frequently reaching that
maximum in Jakarta, a model implementing the rule should place most weight there. SHAP here
functions as a **verification** that the model learned the intended structure rather than
some incidental artefact of the split.

**The eight-to-one ratio between PM2.5 and PM2.5-missingness** (0.3232 vs 0.0088) indicates
the model relies far more on the value itself than on whether it was reconstructed. In 2024
only one PM2.5 value was imputed, so the indicator is almost constant and carries little
information — as expected.

**Temporal features contribute little.** `bulan` at 0.0079 and `tahun` at 0.0078 are near
noise. This is reassuring for `tahun` in particular, which is methodologically problematic:
every test row carries `tahun=2024`, a value never present during training. Its negligible
attribution means the model is not leaning on it, limiting the practical damage.

**What SHAP does not show here.** It does not establish that PM2.5 causes unhealthy air days.
In this design it could not: the target is derived from PM2.5 and its siblings by
construction, so the attribution reflects an arithmetic definition, not an atmospheric
mechanism.

---

## 6. Retrospective SHAP by year, 2020–2024

A diagnostic model refitted on the full period. **Fitted and explained on the same data** —
descriptive only, not a generalisation estimate.

| Year | PM2.5 | PM10 | SO2 | CO | O3 | NO2 |
|---|---:|---:|---:|---:|---:|---:|
| 2020 | 0.147223 | 0.060376 | 0.019672 | 0.023260 | **0.178858** | 0.052844 |
| 2021 | **0.332115** | 0.073627 | 0.010863 | 0.022255 | 0.042590 | 0.024469 |
| 2022 | **0.344051** | 0.071245 | 0.009655 | 0.017406 | 0.044692 | 0.016474 |
| 2023 | **0.327718** | 0.066298 | 0.008432 | 0.028473 | 0.036214 | 0.021980 |
| 2024 | **0.325607** | 0.076996 | 0.013993 | 0.025886 | 0.046096 | 0.023611 |

Dominant and second-ranked pollutant by year:

| Year | Dominant | Mean \|SHAP\| | Second | Mean \|SHAP\| |
|---|---|---:|---|---:|
| 2020 | O3 | 0.178858 | PM2.5 | 0.147223 |
| 2021 | PM2.5 | 0.332115 | PM10 | 0.073627 |
| 2022 | PM2.5 | 0.344051 | PM10 | 0.071245 |
| 2023 | PM2.5 | 0.327718 | PM10 | 0.066298 |
| 2024 | PM2.5 | 0.325607 | PM10 | 0.076996 |

Full table: [`../results/shap_annual_pollutant_importance.csv`](../results/shap_annual_pollutant_importance.csv).

### Reading

**2021–2024 are stable.** PM2.5 attribution sits in a narrow band, 0.3277 to 0.3441, with
PM10 consistently second at 0.066–0.077. The model's decision structure does not drift across
these four years.

**2020 is the exception, and it is a data artefact.** O3 displaces PM2.5 as the dominant
feature, and PM2.5 attribution falls to 0.1472 — less than half its value in any other year.

The cause is documented in the missingness table: **PM2.5 was 100% missing across all 366
days of 2020**. Every 2020 PM2.5 value is an imputed month-specific median derived from
2021–2023. A column of smooth seasonal constants carries almost no discriminative power, so
the model necessarily shifted weight onto the next most informative available sub-index,
which in that period was O3.

**This row therefore describes the imputation scheme, not Jakarta's 2020 atmosphere.** It
should not be reported as a finding about pandemic-period air quality. It is, however, a
clean illustration of how a data-quality decision made during preprocessing propagates all the
way through to interpretability output — which is a finding worth stating in its own right.

---

## 7. Summary

| Aspect | Result |
|---|---|
| Valid observations | 1,826 days |
| Tidak Sehat+ | 616 days (33.7%) |
| Bukan Tidak Sehat+ | 1,210 days (66.3%) |
| Best F1 on test | Random Forest and XGBoost, 0.9600 (tied) |
| Best ROC-AUC on test | Random Forest, 0.9822 |
| Model selected for SHAP | Random Forest |
| Dominant feature, 2021–2024 | PM2.5 |
| Dominant feature, 2020 | O3 — an imputation artefact |

The reported figures are specific to this dataset, this preprocessing pipeline, this feature
configuration, this chronological split and these input representations. See
[`limitations.md`](limitations.md) for the constraints on generalising any of them.
