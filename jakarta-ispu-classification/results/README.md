# Results

Curated numerical outputs from `notebooks/Klasifikasi_ISPU_Jakarta.ipynb`. These CSVs are the
machine-readable source for every table in the README and in `docs/`.

Interpretation lives in [`../docs/results.md`](../docs/results.md). Before reading any number
here, see [`../docs/task-framing.md`](../docs/task-framing.md), which explains why the
tree-based and recurrent models are not solving the same task.

---

## Files

| File | Contents |
|---|---|
| `model_comparison.csv` | Precision, Recall, F1-Score and ROC-AUC for all four models on the 2024 test set |
| `confusion_matrices.csv` | TN / FP / FN / TP counts per model |
| `class_distribution.csv` | Target distribution for the full period and each split |
| `missing_values_by_year.csv` | Missing-value counts by year and pollutant, before imputation |
| `shap_feature_importance_test2024.csv` | Mean absolute SHAP for all 17 features, Random Forest, test set |
| `shap_annual_pollutant_importance.csv` | Retrospective mean absolute SHAP by year and pollutant |
| `figures/` | Selected PNG figures exported from the notebook |

---

## Headline results

Test set: 366 days in 2024, 89 of them Tidak Sehat+. Positive class: **Tidak Sehat+**.

| Model | Input | Precision | Recall | F1-Score | ROC-AUC |
|---|---|---:|---:|---:|---:|
| Random Forest | Same-day tabular | 0.9767 | 0.9438 | **0.9600** | **0.9822** |
| XGBoost | Same-day tabular | 0.9767 | 0.9438 | **0.9600** | 0.9705 |
| LSTM | 14-day sequence | 0.4639 | 0.5056 | 0.4839 | 0.6902 |
| GRU | 14-day sequence | 0.4507 | 0.3596 | 0.4000 | 0.7159 |

Random Forest and XGBoost produce identical hard predictions — the same 7 errors in 366 days —
and differ only in probability calibration.

---

## Reproducing the full output set

Running the notebook end to end writes 18 CSV tables and 11 PNG figures to `output/`, plus an
archive at `output/output_eksperimen_ispu.zip`. That directory is gitignored; the files here
are the curated subset used for reporting.

Tree-based figures reproduce exactly under `random_state=42`. LSTM and GRU figures may vary by
a few points across hardware and library versions because some TensorFlow GPU kernels are
non-deterministic.
