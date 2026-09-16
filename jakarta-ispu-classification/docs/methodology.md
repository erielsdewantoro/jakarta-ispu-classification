# Methodology

Complete description of the experimental procedure. Every figure quoted here is taken from
the executed notebook, `notebooks/Klasifikasi_ISPU_Jakarta.ipynb`.

---

## 1. Research design

A supervised binary classification problem over a daily time series, evaluated with a single
chronological hold-out year.

| Element | Specification |
|---|---|
| Unit of observation | One calendar day |
| Research period | 2020-01-01 → 2024-12-31 |
| Target | Binary — Tidak Sehat+ vs Bukan Tidak Sehat+ |
| Positive class | Tidak Sehat+ |
| Validation scheme | Chronological hold-out (train 2020–2023, test 2024) |
| Primary metrics | Precision, Recall, F1-Score, ROC-AUC |
| Interpretability | SHAP TreeExplainer |
| Global seed | 42 (Python `random`, NumPy, scikit-learn, XGBoost, TensorFlow) |

---

## 2. Data loading and structural inspection

The raw file `ispu_dki_all.csv` is loaded and `tanggal` is parsed to datetime with
`errors="coerce"`. Column presence is validated against the expected set — `tanggal`,
`stasiun`, `pm25`, `pm10`, `so2`, `co`, `o3`, `no2`, `max`, `critical`, `categori` — and the
notebook raises immediately if any is absent.

| Property | Value |
|---|---|
| Rows | 5,538 |
| Columns | 11 |
| Date range | 2010-01-01 → 2025-02-28 |
| Distinct station labels | 5 |

The date span is exactly 5,538 days, so the file holds exactly one row per calendar day. It is
a single daily series rather than a station-by-day panel.

---

## 3. Period selection and cleaning

Rows are restricted to 2020-01-01 → 2024-12-31, yielding 1,827 observations. One row carrying
the category `TIDAK ADA DATA` is removed, leaving **1,826** valid observations.

Category distribution before binarisation:

| Category | Days |
|---|---:|
| SEDANG | 1,160 |
| TIDAK SEHAT | 609 |
| BAIK | 50 |
| SANGAT TIDAK SEHAT | 7 |
| TIDAK ADA DATA (removed) | 1 |

The notebook asserts the resulting date boundaries and the absence of `TIDAK ADA DATA` before
proceeding.

---

## 4. Target construction

```python
analysis_df["target_tidak_sehat"] = (
    analysis_df["categori"].isin(["TIDAK SEHAT", "SANGAT TIDAK SEHAT"]).astype(int)
)
```

| Target class | Source categories | Days | Share |
|---|---|---:|---:|
| Tidak Sehat+ (1) | TIDAK SEHAT, SANGAT TIDAK SEHAT | 616 | 33.73% |
| Bukan Tidak Sehat+ (0) | BAIK, SEDANG | 1,210 | 66.27% |

**Rationale for binarisation.** The four-class problem is unusable under a chronological
split: SANGAT TIDAK SEHAT occurs on 7 days and BAIK on 50 across five years. Collapsing to a
binary target places the decision boundary at the ISPU = 100 threshold, which is also the
point at which public-health advisories change, making the split substantively meaningful
rather than merely convenient.

---

## 5. Feature engineering

Seventeen features in three groups.

**Pollutant sub-indices (6).** `pm25`, `pm10`, `so2`, `co`, `o3`, `no2`, coerced to numeric
with `errors="coerce"`.

**Temporal (5).**

| Feature | Definition |
|---|---|
| `tahun` | Calendar year |
| `bulan` | Month 1–12, a proxy for seasonality |
| `hari_dalam_minggu` | Day of week, 0 = Monday … 6 = Sunday |
| `is_weekend` | 1 if Saturday or Sunday |
| `is_pasca_pandemi` | 1 if year ≥ 2022, distinguishing the 2020–2021 restricted-mobility period from 2022–2024 |

**Missingness indicators (6).** `{pollutant}_was_missing` = 1 where the value was null
**before** imputation. Recording missingness as an explicit feature allows the model to
distinguish an observed value from a reconstructed one, and lets SHAP quantify how much the
model relies on reconstructed data.

### Excluded columns

| Column | Reason |
|---|---|
| `max` | The ISPU value itself; direct leakage |
| `critical` | Identifies the pollutant producing `max`; direct leakage |
| `categori` | Source of the target |
| `stasiun` | The study treats the data as one city-level daily series, not a between-station comparison |
| `target_tidak_sehat`, `target_label` | Target columns |

The notebook enforces these exclusions programmatically and raises if any appears in
`feature_cols`. Note that this removes the explicit leakage but not the structural dependency
between target and sub-indices — see [`task-framing.md`](task-framing.md).

---

## 6. Chronological split

```python
test_start_date = pd.Timestamp("2024-01-01")
train_raw = feature_df_raw.loc[feature_df_raw["tanggal"] < test_start_date]
test_raw  = feature_df_raw.loc[feature_df_raw["tanggal"] >= test_start_date]
```

| Subset | Period | n | Positive days | Positive rate |
|---|---|---:|---:|---:|
| Train | 2020-01-01 → 2023-12-31 | 1,460 | 527 | 0.361 |
| Test | 2024-01-01 → 2024-12-31 | 366 | 89 | 0.243 |

Random splitting is inappropriate for a daily series: neighbouring days are strongly
autocorrelated, so random assignment would place near-duplicate days on both sides of the
split and inflate every metric. The chronological split also reproduces the realistic setting
in which a model trained on history is applied to a later period.

The positive rate falls from 36.1% to 24.3%. The test year is both later and materially
cleaner than the training years, so the models face a base-rate shift as well as a temporal
one.

---

## 7. Missing-value handling

Imputation is performed **after** the split, with all parameters fitted on training data only.

### Observed missingness

| Column | Missing | % |
|---|---:|---:|
| `pm25` | 369 | 20.21 |
| `pm10` | 200 | 10.95 |
| `so2` | 46 | 2.52 |
| `co` | 27 | 1.48 |
| `o3` | 23 | 1.26 |
| `no2` | 14 | 0.77 |

By year:

| Year | PM2.5 | PM10 | SO2 | CO | O3 | NO2 |
|---|---:|---:|---:|---:|---:|---:|
| 2020 | 366 | 1 | 29 | 7 | 2 | 4 |
| 2021 | 2 | 20 | 11 | 12 | 15 | 2 |
| 2022 | 0 | 0 | 0 | 0 | 0 | 0 |
| 2023 | 0 | 159 | 5 | 6 | 4 | 7 |
| 2024 | 1 | 20 | 1 | 2 | 2 | 1 |

PM2.5 is missing on all 366 days of 2020 — a complete absence, not scattered gaps. PM10 is
missing on 159 days of 2023.

### Three-stage procedure

```python
def impute_pollutants(part_df, monthly_medians=None, global_medians=None, fit=False):
    work = part_df.copy().sort_values("tanggal").set_index("tanggal")
    for col in pollutant_cols:
        work[col] = pd.to_numeric(work[col], errors="coerce")
        work[col] = work[col].interpolate(method="time", limit=7, limit_direction="both")
    if fit:
        monthly_medians = {c: work.groupby(work.index.month)[c].median() for c in pollutant_cols}
        global_medians  = {c: work[c].median() for c in pollutant_cols}
    for col in pollutant_cols:
        month_values = work.index.month.map(monthly_medians[col])
        work[col] = work[col].fillna(pd.Series(month_values, index=work.index))
        work[col] = work[col].fillna(global_medians[col])
    return work.reset_index(), monthly_medians, global_medians
```

1. **Time-based linear interpolation**, `limit=7` days in both directions. Short gaps are
   bridged from their temporal neighbours; the 7-day cap prevents interpolation across
   month-long absences.
2. **Month-specific median**, learned from training data. Preserves seasonal level, which a
   single global constant would flatten.
3. **Global training median** for any residual gap.

The `fit=True` path is used only on the training subset. Test-period values are filled using
medians learned from 2020–2023, so no test-period information enters the imputation. The
notebook asserts zero remaining nulls in both subsets.

**Effect on 2020.** With PM2.5 wholly absent, stages 1 and 2 cannot recover anything from
within 2020; every value comes from the month-specific training median, which is dominated by
2021–2023. The 2020 PM2.5 column is therefore a smooth seasonal curve, not a measurement.
This is the direct cause of the 2020 SHAP anomaly described in
[`results.md`](results.md).

---

## 8. Scaling and model-input preparation

`StandardScaler` is fitted on the training feature matrix and applied to both subsets. Scaled
features are used **only** by LSTM and GRU; tree-based models operate on unscaled values,
which is appropriate because tree splits are invariant to monotone rescaling.

---

## 9. Sequence construction for the recurrent models

```python
for i in range(window_size, len(X_array)):
    X_seq.append(X_array[i - window_size:i])   # days i-14 ... i-1
    y_seq.append(y_array[i])                   # day i
```

| Component | Value |
|---|---|
| Window | 14 days |
| Shape | `(n, 14, 17)` |
| Training sequences | 1,156 |
| Validation sequences | 290 |
| Test sequences | 366 |

**Window length.** Fourteen days spans two full weekly cycles, capturing weekday/weekend
structure and short-term meteorological persistence without stretching so far back that early
context becomes uninformative. The value was fixed a priori and not tuned.

**Validation split.** The first 80% of training sequences are used for fitting and the final
20% for validation, preserving chronological order. Random validation splitting would leak
future information into early-stopping decisions.

**Test alignment.** The last 14 training days seed the first 2024 prediction:

```python
X_context_for_test = np.vstack([X_train_scaled_df.tail(WINDOW_SIZE).values,
                                X_test_scaled_df.values])
```

This keeps all 366 test days evaluable. The notebook asserts that the resulting target dates
match the tabular test dates exactly, so all four models are scored on an identical set of
days.

**Important.** The window ends at day *i-1*. The recurrent models never see the target day's
own features and are therefore performing next-day forecasting, unlike the tree models. See
[`task-framing.md`](task-framing.md).

---

## 10. Model specifications

### Random Forest

```python
RandomForestClassifier(
    n_estimators=400, max_depth=None, min_samples_leaf=3,
    class_weight="balanced", random_state=42, n_jobs=-1,
)
```

`min_samples_leaf=3` restrains growth on 1,460 rows; `class_weight="balanced"` reweights
inversely to class frequency, compensating for the 66/34 imbalance; unbounded depth is
tolerable given the leaf-size floor.

### XGBoost

```python
XGBClassifier(
    n_estimators=400, learning_rate=0.03, max_depth=4,
    subsample=0.9, colsample_bytree=0.9,
    objective="binary:logistic", eval_metric="logloss",
    scale_pos_weight=933/527,      # = 1.7704
    random_state=42, n_jobs=-1,
)
```

A low learning rate paired with 400 shallow trees favours gradual, well-regularised fitting.
`scale_pos_weight` is computed from the training class ratio, the boosting analogue of
balanced class weights.

### LSTM and GRU

```python
Sequential([
    Input(shape=(14, 17)),
    LSTM(64, return_sequences=False),   # GRU(64) for the GRU variant
    Dropout(0.25),
    Dense(32, activation="relu"),
    Dropout(0.15),
    Dense(1, activation="sigmoid"),
])
model.compile(optimizer=Adam(learning_rate=0.001),
              loss="binary_crossentropy", metrics=["accuracy"])
```

Training configuration:

| Setting | Value |
|---|---|
| Epochs | 60 (maximum) |
| Batch size | 32 |
| Early stopping | `monitor="val_loss"`, `patience=8`, `restore_best_weights=True` |
| Class weight | Balanced, computed from the training sequence targets |
| `shuffle` | `False` — preserves temporal ordering within epochs |
| Decision threshold | 0.50 |

The two architectures are identical apart from the recurrent cell, isolating the LSTM-vs-GRU
comparison to the gating mechanism alone.

No hyperparameter search was performed for any of the four models.

---

## 11. Evaluation

Positive class: Tidak Sehat+.

| Metric | Definition | Why it is used here |
|---|---|---|
| Precision | TP / (TP + FP) | Cost of false alarms on an unhealthy-air warning |
| Recall | TP / (TP + FN) | Cost of missing a genuinely unhealthy day — the more consequential error |
| F1-Score | Harmonic mean | Single summary balancing the two under class imbalance |
| ROC-AUC | Area under the ROC curve | Threshold-independent separability; distinguishes models with identical hard predictions |

Accuracy is reported as secondary only. On this test set a majority-class classifier reaches
75.7% accuracy while detecting no unhealthy days at all, so accuracy carries little
information.

Supporting visualisations: confusion matrix, ROC curves for all four models, and
precision–recall curves with the positive-class base rate drawn as a reference line.

---

## 12. Interpretation with SHAP

**Test-set SHAP.** `shap.TreeExplainer` is applied to the selected ensemble over the 366
test-set days. The 2024 period is used because it was not seen during training, so the
attributions describe behaviour on genuinely unseen data. Values for the positive class are
extracted, and their shape is asserted against the test matrix.

**Retrospective SHAP.** A separate diagnostic model is refitted on the full imputed 2020–2024
period, and mean absolute SHAP is averaged by year to trace how attribution shifts over time.
This model is fitted and explained on the same data; it is a descriptive diagnostic, not a
generalisation estimate, and is labelled as such throughout.

SHAP describes how the trained model uses its inputs. It does not establish causation between
pollutant levels and the ISPU category — and in this design it could not, since the target is
derived from those same inputs by construction.

---

## 13. Outputs

The notebook writes 18 CSV tables and 11 PNG figures to `output/`, then archives them as
`output/output_eksperimen_ispu.zip`. The curated subset used in this repository's reporting
is committed under `results/`.
