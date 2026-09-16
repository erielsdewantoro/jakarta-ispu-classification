# Source Code

The end-to-end implementation lives in the notebook:

```text
notebooks/Klasifikasi_ISPU_Jakarta.ipynb
```

It covers data loading, inspection, cleaning, target construction, feature engineering,
leakage guards, the chronological split, imputation, scaling, sequence construction, training
for all four models, evaluation, and SHAP interpretation.

This directory holds standalone scripts that support the analysis rather than duplicate it.

---

## `verify_leakage.py`

Reproduces the target-leakage evidence in [`../docs/task-framing.md`](../docs/task-framing.md)
and computes the three baselines the main experiment omits.

```bash
python src/verify_leakage.py                          # searches nearby for the CSV
python src/verify_leakage.py path/to/ispu_dki_all.csv  # explicit path
```

It performs three checks:

1. **Identity check** — verifies row by row that `max` equals the maximum of the six sub-index
   columns, that `critical` names the column producing it, and that `categori` is the
   threshold bucket of `max`. If these hold, the target is a deterministic function of the
   model's own features.
2. **Rule baseline** — classifies the 2024 test set using `max(sub-indices) > 100` alone, with
   no model. Compare the resulting F1 against the reported Random Forest F1 of 0.9600.
3. **Trivial baselines** — majority-class (the floor for the tabular task) and persistence
   (the fair reference point for the next-day forecasting task that LSTM and GRU perform).

The script requires `ispu_dki_all.csv`, which is not committed — see
[`../data/README.md`](../data/README.md). It is read-only and writes nothing.

---

## Refactoring

The research code is deliberately kept in a single notebook so that the narrative, the code
and the outputs stay in one readable sequence. If the project is later refactored into modules
(`preprocessing.py`, `models.py`, `evaluation.py`, `interpretation.py`), they belong here.
