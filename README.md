# Jakarta ISPU Classification

Binary classification of Jakarta Air Pollution Standard Index (ISPU) categories, comparing
two tree-based ensembles against two recurrent neural networks, with SHAP-based model
interpretation.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/erielsdewantoro/jakarta-ispu-classification/blob/main/notebooks/Klasifikasi_ISPU_Jakarta.ipynb)
[![nbviewer](https://img.shields.io/badge/render-nbviewer-orange)](https://nbviewer.org/github/erielsdewantoro/jakarta-ispu-classification/blob/main/notebooks/Klasifikasi_ISPU_Jakarta.ipynb)
![Python](https://img.shields.io/badge/python-3.10-blue)
![License](https://img.shields.io/badge/license-MIT-green)

This project was developed as an undergraduate thesis in Information Technology at
Universitas Bina Sarana Informatika (UBSI).

---

## Quick View

| Component | Details |
|---|---|
| Task | Binary classification of daily ISPU status |
| Positive class | **Tidak Sehat+** — 616 of 1,826 days (33.73%) |
| Negative class | **Bukan Tidak Sehat+** — 1,210 of 1,826 days (66.27%) |
| Unit of observation | One row per calendar day (daily DKI Jakarta ISPU series) |
| Research period | 2020-01-01 to 2024-12-31 |
| Split | Chronological — train 2020–2023 (n=1,460), test 2024 (n=366) |
| Features | 17 — 6 pollutant sub-indices, 5 temporal, 6 missingness indicators |
| Models | Random Forest, XGBoost, LSTM, GRU |
| Interpretation | SHAP (TreeExplainer) on Random Forest |
| Best F1 on test | 0.9600 — Random Forest and XGBoost (tied) |
| ⚠ Read first | [Two different tasks are being compared](#a-note-on-task-framing-read-this-before-the-results) |

---

## Overview

Air quality is a routinely monitored environmental indicator in Jakarta, published daily as
the Indeks Standar Pencemar Udara (ISPU). This project treats the daily ISPU status as a
binary classification target and compares how four modelling approaches behave on it.

The four original ISPU categories are collapsed into two classes:

| Target class | Original ISPU categories | Days (2020–2024) |
|---|---|---:|
| **Tidak Sehat+** (positive) | TIDAK SEHAT, SANGAT TIDAK SEHAT | 616 |
| **Bukan Tidak Sehat+** (negative) | BAIK, SEDANG | 1,210 |

The collapse is motivated by extreme sparsity at the tails: across the research period there
are only 7 SANGAT TIDAK SEHAT days and 50 BAIK days, which is too few to support a stable
four-class model under a chronological split.

> This is a classification study. It does not predict the numerical ISPU value.

---

## Research Objectives

1. Build and evaluate classification models that identify whether a given day falls in the
   **Tidak Sehat+** category.
2. Compare a tabular, tree-based modelling approach (Random Forest, XGBoost) against a
   sequential, deep-learning approach (LSTM, GRU).
3. Apply SHAP to the selected ensemble model to describe which features drive its decisions,
   and to check whether the learned decision structure is consistent with the documented
   ISPU rule.

---

## A note on task framing — read this before the results

Two things about this experiment materially change how the numbers below should be read.
Both are documented in full in [`docs/task-framing.md`](docs/task-framing.md).

### 1. The pollutant columns are ISPU sub-indices, not raw concentrations

In this dataset, `pm25`, `pm10`, `so2`, `co`, `o3` and `no2` hold **already-converted ISPU
sub-index values**. The published daily category is derived from the highest sub-index via
fixed thresholds (0–50 BAIK, 51–100 SEDANG, 101–200 TIDAK SEHAT, 201–300 SANGAT TIDAK SEHAT),
and the `max` and `critical` columns record that maximum and its source pollutant.

This identity is directly visible in the raw data. For example, in the first row of the file:

```
pm10=60  so2=4  co=73  o3=27  no2=14   →  max=73, critical=CO, categori=SEDANG
pm10=32  so2=2  co=16  o3=33  no2=9    →  max=33, critical=O3, categori=BAIK
```

Dropping `max` and `critical` from the feature set removes the explicit answer, but not the
underlying dependency: a tree-based model that receives all six sub-indices can reconstruct
`max(...) > 100` from splits on its own inputs.

**Therefore the Random Forest and XGBoost scores should not be read as air-quality
forecasting performance.** They measure how well a model recovers a known deterministic rule
from data in which that rule is present. The residual 1.9% error rate is attributable mainly
to imputed values replacing the originals that actually determined the published category.

### 2. The tree models and the recurrent models solve different problems

The sequence builder maps `X[i-14 : i] → y[i]`. The window ends the day *before* the target
day, so **LSTM and GRU never see the target day's own pollutant values**. They perform
genuine next-day forecasting from a 14-day history.

Random Forest and XGBoost, by contrast, receive the target day's own sub-indices.

| | Random Forest / XGBoost | LSTM / GRU |
|---|---|---|
| Input | Target day's own 17 features | 14 preceding days, standardised |
| Sees target day's pollutants? | **Yes** | **No** |
| Effective task | Recover the ISPU rule | Forecast tomorrow's status |
| Difficulty | Near-deterministic | Genuinely hard |

The performance gap in the results table is therefore **not** evidence that ensembles beat
recurrent networks on this problem. It reflects the fact that one group was given the answer
key in a scrambled form and the other was not. An LSTM F1 of 0.4839 on next-day forecasting
and a Random Forest F1 of 0.9600 on same-day rule reconstruction are not comparable
quantities.

Reporting this openly is deliberate. It is the most useful methodological finding the
experiment produced.

---

## Dataset

Daily ISPU records for DKI Jakarta published by **Dinas Lingkungan Hidup Provinsi DKI
Jakarta / Satu Data Jakarta**, accessed via Kaggle as `ispu_dki_all.csv`.

| Property | Value |
|---|---|
| Raw rows × columns | 5,538 × 11 |
| Raw date range | 2010-01-01 to 2025-02-28 |
| Distinct station labels | 5 |
| Rows in research period (2020–2024) | 1,827 |
| Rows after removing `TIDAK ADA DATA` | **1,826** |

The raw span of 2010-01-01 to 2025-02-28 is exactly 5,538 days, so the file contains exactly
one row per calendar day — it is a single daily series, not a station-by-day panel. The
`stasiun` column is therefore excluded from the feature set: the study treats the data as one
city-level daily series rather than a comparison between monitoring stations.

The raw CSV is **not committed** to this repository. See [`data/README.md`](data/README.md)
for how to obtain it and where to place it.

### Original ISPU category distribution, 2020–2024

| Category | Days |
|---|---:|
| SEDANG | 1,160 |
| TIDAK SEHAT | 609 |
| BAIK | 50 |
| SANGAT TIDAK SEHAT | 7 |
| TIDAK ADA DATA (excluded) | 1 |

---

## Features

Seventeen features in three groups. `max`, `critical`, `stasiun`, `categori` and the target
columns are excluded, and the notebook asserts this programmatically before training.

| Group | Features | Count |
|---|---|---:|
| Pollutant sub-indices | `pm25`, `pm10`, `so2`, `co`, `o3`, `no2` | 6 |
| Temporal | `tahun`, `bulan`, `hari_dalam_minggu`, `is_weekend`, `is_pasca_pandemi` | 5 |
| Missingness indicators | `{pollutant}_was_missing` for each of the six pollutants | 6 |

Missingness indicators are recorded **before** imputation, so the model can distinguish an
observed value from a reconstructed one.

---

## Missing Data

Missingness is substantial and highly structured, which constrains interpretation.

| Column | Missing | % of 1,826 |
|---|---:|---:|
| `pm25` | 369 | 20.21 |
| `pm10` | 200 | 10.95 |
| `so2` | 46 | 2.52 |
| `co` | 27 | 1.48 |
| `o3` | 23 | 1.26 |
| `no2` | 14 | 0.77 |

By year and pollutant:

| Year | PM2.5 | PM10 | SO2 | CO | O3 | NO2 |
|---|---:|---:|---:|---:|---:|---:|
| 2020 | **366** | 1 | 29 | 7 | 2 | 4 |
| 2021 | 2 | 20 | 11 | 12 | 15 | 2 |
| 2022 | 0 | 0 | 0 | 0 | 0 | 0 |
| 2023 | 0 | **159** | 5 | 6 | 4 | 7 |
| 2024 | 1 | 20 | 1 | 2 | 2 | 1 |

**PM2.5 is 100% missing throughout 2020** (all 366 days), and PM10 is missing for 159 days in
2023. Every PM2.5 value used for 2020 is imputed. Any statement about PM2.5's contribution in
2020 describes the imputation, not the atmosphere.

### Imputation strategy

Applied **after** the chronological split, with all parameters fitted on training data only:

1. Time-based linear interpolation, `limit=7` days in both directions.
2. Remaining gaps filled with the **month-specific median** from the training period.
3. Any residual gaps filled with the **global training median**.

Test-period values are imputed using the medians learned from 2020–2023, never from 2024.
The notebook asserts zero remaining nulls in both subsets afterwards.

---

## Train–Test Split

Chronological, never random, because the data is a daily time series.

| Subset | Period | Observations | Positive rate |
|---|---|---:|---:|
| Train | 2020-01-01 → 2023-12-31 | 1,460 | 0.361 (527 days) |
| Test | 2024-01-01 → 2024-12-31 | 366 | 0.243 (89 days) |

The positive rate drops from 36.1% in training to 24.3% in testing. The test year is
therefore not just later — it is materially cleaner. This distribution shift is part of what
the models are being asked to absorb.

### Sequence construction for LSTM and GRU

| Component | Value |
|---|---|
| Window length | 14 days |
| Mapping | `X[i-14 : i] → y[i]` (target day excluded from window) |
| Training sequences | 1,156 |
| Validation sequences | 290 (last 20% of training sequences, chronological) |
| Test sequences | 366 |

The final 14 days of the training period seed the first 2024 prediction, so the recurrent
models are evaluated on exactly the same 366 target days as the tree models — no test day is
lost to the window.

---

## Model Configuration

| Model | Hyperparameters |
|---|---|
| **Random Forest** | `n_estimators=400`, `max_depth=None`, `min_samples_leaf=3`, `class_weight="balanced"`, `random_state=42` |
| **XGBoost** | `n_estimators=400`, `learning_rate=0.03`, `max_depth=4`, `subsample=0.9`, `colsample_bytree=0.9`, `objective="binary:logistic"`, `eval_metric="logloss"`, `scale_pos_weight=1.7704` (= 933 ÷ 527), `random_state=42` |
| **LSTM** | `LSTM(64)` → `Dropout(0.25)` → `Dense(32, relu)` → `Dropout(0.15)` → `Dense(1, sigmoid)`; Adam `lr=0.001`, binary cross-entropy, `epochs=60`, `batch_size=32`, `EarlyStopping(monitor="val_loss", patience=8, restore_best_weights=True)`, balanced `class_weight`, `shuffle=False` |
| **GRU** | Identical to LSTM with `GRU(64)` in place of `LSTM(64)` |

Features are standardised with `StandardScaler` (fitted on training data only) for the
recurrent models. Tree models use unscaled features. Classification threshold is 0.50 for all
four models.

---

## Results

Evaluated on the 2024 test set: 366 days, 89 of them Tidak Sehat+.
**Positive class = Tidak Sehat+.**

| Model | Input | Precision | Recall | F1-Score | ROC-AUC |
|---|---|---:|---:|---:|---:|
| **Random Forest** | Same-day tabular | 0.9767 | 0.9438 | **0.9600** | **0.9822** |
| **XGBoost** | Same-day tabular | 0.9767 | 0.9438 | **0.9600** | 0.9705 |
| **LSTM** | 14-day history | 0.4639 | 0.5056 | 0.4839 | 0.6902 |
| **GRU** | 14-day history | 0.4507 | 0.3596 | 0.4000 | 0.7159 |

Machine-readable copy: [`results/model_comparison.csv`](results/model_comparison.csv).

Accuracy is deliberately not the headline metric. A majority-class classifier would reach
75.7% accuracy on this test set while identifying zero Tidak Sehat+ days.

### Confusion matrices (test 2024)

| Model | TN | FP | FN | TP |
|---|---:|---:|---:|---:|
| Random Forest | 275 | 2 | 5 | 84 |
| XGBoost | 275 | 2 | 5 | 84 |
| LSTM | 225 | 52 | 44 | 45 |
| GRU | 238 | 39 | 57 | 32 |

Random Forest and XGBoost produce identical Precision, Recall and F1 because they produce
identical hard predictions — the same 7 misclassified days out of 366. They differ only in
predicted probabilities, which is why their ROC-AUC values diverge (0.9822 vs 0.9705). This
convergence is itself evidence for the rule-reconstruction reading: two different algorithms
independently arrived at the same decision boundary because the boundary is a deterministic
threshold, not a statistical estimate.

### Model selected for interpretation

Random Forest, on the basis of equal F1 and higher ROC-AUC. The choice is marginal — XGBoost
is not meaningfully worse, and nothing in the SHAP analysis depends on which of the two was
picked.

---

## SHAP Interpretation

SHAP (TreeExplainer) was applied to the Random Forest model over the 366 test-set days.

### Mean absolute SHAP, test set 2024

| Rank | Feature | Mean \|SHAP\| |
|---:|---|---:|
| 1 | PM2.5 | 0.323157 |
| 2 | PM10 | 0.083515 |
| 3 | O3 | 0.047695 |
| 4 | CO | 0.026628 |
| 5 | NO2 | 0.023424 |
| 6 | SO2 | 0.013650 |
| 7 | PM2.5 missing indicator | 0.008758 |
| 8 | Bulan | 0.007862 |
| 9 | Tahun | 0.007825 |
| 10 | PM10 missing indicator | 0.005726 |
| 11 | Pasca-pandemi indicator | 0.002825 |
| 12 | Hari dalam minggu | 0.001205 |

Full table: [`results/shap_feature_importance_test2024.csv`](results/shap_feature_importance_test2024.csv).

**How to read this.** PM2.5 accounts for roughly 62% of total mean absolute SHAP across all
17 features, and the six pollutant sub-indices together account for about 96%. Temporal and
missingness features contribute almost nothing.

This is exactly the profile the ISPU rule predicts. PM2.5 is the sub-index that most often
reaches the daily maximum in Jakarta, so a model reconstructing `max(...) > 100` should
concentrate almost all of its decision weight there. SHAP here functions as a **consistency
check**: it confirms the model learned the intended rule rather than an incidental artefact
of the split. It is not evidence that PM2.5 *causes* unhealthy air days — the causal claim
would require concentration data and a design this study does not have.

### Retrospective SHAP by year, 2020–2024

A diagnostic model was refitted on the full 2020–2024 period to read year-by-year feature
contribution. This is a retrospective diagnostic, **not** a generalisation estimate: it is
fitted and explained on the same data.

| Year | PM2.5 | PM10 | SO2 | CO | O3 | NO2 |
|---|---:|---:|---:|---:|---:|---:|
| 2020 | 0.147223 | 0.060376 | 0.019672 | 0.023260 | **0.178858** | 0.052844 |
| 2021 | **0.332115** | 0.073627 | 0.010863 | 0.022255 | 0.042590 | 0.024469 |
| 2022 | **0.344051** | 0.071245 | 0.009655 | 0.017406 | 0.044692 | 0.016474 |
| 2023 | **0.327718** | 0.066298 | 0.008432 | 0.028473 | 0.036214 | 0.021980 |
| 2024 | **0.325607** | 0.076996 | 0.013993 | 0.025886 | 0.046096 | 0.023611 |

2020 is the sole year in which O3 rather than PM2.5 dominates. This should be read as a
**data artefact, not an atmospheric finding**: PM2.5 was 100% missing in 2020, so every 2020
PM2.5 value is an imputed monthly or global median. Imputed constants carry little
discriminative signal, so the model necessarily shifted weight onto the next most informative
available sub-index. The 2020 row describes the imputation scheme.

Full table: [`results/shap_annual_pollutant_importance.csv`](results/shap_annual_pollutant_importance.csv).

---

## Key Findings

1. **Random Forest and XGBoost achieve identical hard predictions** (F1 0.9600, 7 errors in
   366 days) and differ only in probability calibration (ROC-AUC 0.9822 vs 0.9705).
2. **The high ensemble scores reflect rule reconstruction, not forecasting.** The target is a
   deterministic threshold function of the input sub-indices; the models recovered it.
3. **LSTM and GRU were solving a genuinely harder problem** — next-day forecasting without
   access to the target day's pollutant values. Their scores are not directly comparable to
   the ensembles'.
4. **SHAP confirms the learned rule matches the documented one**: PM2.5 carries ~62% of total
   attribution and pollutant sub-indices ~96%, consistent with PM2.5 being Jakarta's most
   frequent critical pollutant.
5. **The 2020 SHAP anomaly is traceable to complete PM2.5 missingness**, illustrating how
   imputation propagates directly into interpretability output.

---

## Limitations

A fuller treatment is in [`docs/limitations.md`](docs/limitations.md).

1. **Target leakage is structural, not incidental.** Excluding `max` and `critical` does not
   remove the dependency between target and features. See
   [`docs/task-framing.md`](docs/task-framing.md).
2. **Cross-family comparison is not like-for-like.** The two model families receive different
   inputs and effectively solve different tasks.
3. **PM2.5 is entirely imputed for 2020** and PM10 for 159 days of 2023. Interpretability
   output for those periods reflects the imputation scheme.
4. **The `tahun` feature cannot generalise.** Training covers 2020–2023; every test row has
   `tahun=2024`, a value never observed during training. Tree models handle it as an
   out-of-range split. Its SHAP contribution is negligible (0.0078), so the practical impact
   is small, but the feature is methodologically unsound for deployment.
5. **No hyperparameter search was performed.** All configurations are fixed a priori. The
   recurrent models in particular were not tuned for window length, capacity, or decision
   threshold, so their scores are a lower bound rather than a fair ceiling.
6. **Single train–test split.** No walk-forward or rolling-origin validation, so the reported
   figures rest on one test year and carry no confidence interval.
7. **Retrospective SHAP is fitted and explained on the same data** and is a diagnostic only.
8. **Results are specific to this city, period, and design** and should not be generalised
   without further validation.

---

## Reproducibility

```bash
git clone https://github.com/erielsdewantoro/jakarta-ispu-classification.git
cd jakarta-ispu-classification
pip install -r requirements.txt
```

1. Obtain `ispu_dki_all.csv` from its original source — see [`data/README.md`](data/README.md).
2. Place it at the repository root or in `data/`.
3. Open `notebooks/Klasifikasi_ISPU_Jakarta.ipynb` and run all cells in order.

The notebook auto-detects Google Colab, installs dependencies, and prompts for a file upload
if the dataset is not found locally.

**On determinism.** `random_state=42` is fixed for NumPy, Python's `random`, scikit-learn,
XGBoost and TensorFlow's global seed. Tree-based results reproduce exactly. TensorFlow
results may vary by a few points across hardware, driver and library versions because some
GPU kernels are non-deterministic. If your LSTM or GRU figures differ slightly from those
above, that is expected; the tree-based figures should not move.

---

## Project Structure

```text
.
├── README.md
├── LICENSE
├── CITATION.cff
├── requirements.txt
├── .gitignore
│
├── data/
│   └── README.md                     # how to obtain the dataset + data dictionary
│
├── notebooks/
│   └── Klasifikasi_ISPU_Jakarta.ipynb
│
├── src/
│   ├── README.md
│   └── verify_leakage.py             # reproduces the target-leakage evidence
│
├── results/
│   ├── README.md
│   ├── model_comparison.csv
│   ├── confusion_matrices.csv
│   ├── class_distribution.csv
│   ├── shap_feature_importance_test2024.csv
│   ├── shap_annual_pollutant_importance.csv
│   └── figures/
│
└── docs/
    ├── task-framing.md               # target leakage and task asymmetry
    ├── methodology.md
    ├── results.md
    └── limitations.md
```

---

## Technologies

Python · pandas · NumPy · scikit-learn · XGBoost · TensorFlow/Keras · SHAP · Matplotlib ·
Seaborn · Jupyter · Google Colab

---

## Academic Context

Undergraduate thesis, Program Studi Teknologi Informasi, Universitas Bina Sarana Informatika.

**Title:** *Analisis Perbandingan Kinerja Model Ensemble dan Deep Learning untuk Klasifikasi
ISPU Jakarta Menggunakan Explainable AI (SHAP)*

This repository is a technical portfolio version of the research. It contains the analysis,
code and results; it does not reproduce the thesis manuscript or any administrative documents.

---

## Author

**Eriel Setiawan Dewantoro**
Information Technology, Universitas Bina Sarana Informatika

[GitHub](https://github.com/erielsdewantoro) · [LinkedIn](https://www.linkedin.com/in/eriel-setiawan-dewantoro/)

---

## License

Code in this repository is released under the [MIT License](LICENSE).

The MIT License applies to the code only. It confers no rights over the underlying ISPU
dataset, which remains subject to the terms of its original publisher. Consult those terms
before redistributing the data or any derivative of it.
