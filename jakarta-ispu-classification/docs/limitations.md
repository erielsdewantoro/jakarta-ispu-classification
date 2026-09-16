# Limitations

Constraints that bound what can legitimately be claimed from this work, ordered from most to
least consequential.

---

## 1. Target leakage is structural

The most important limitation. The pollutant columns are ISPU sub-indices, and the ISPU
category is the threshold bucket of their maximum. The target is therefore a deterministic
function of the feature set.

Excluding `max` and `critical` removes the explicit answer but not the dependency. A
tree-based model can express `max(sub-indices) > 100` as a disjunction of splits on its own
inputs.

**Consequence.** Random Forest's F1 of 0.9600 and ROC-AUC of 0.9822 measure how completely a
model recovers a known arithmetic rule from data in which the rule is embedded. They are not
an estimate of air-quality prediction performance and must not be quoted as such.

Full argument and supporting evidence: [`task-framing.md`](task-framing.md).

---

## 2. The two model families solve different tasks

The sequence builder maps `X[i-14 : i] → y[i]`, and Python slicing excludes index `i`. LSTM
and GRU therefore never observe the target day's own pollutant values; they forecast the next
day from a 14-day history. Random Forest and XGBoost receive the target day's own features.

**Consequence.** The performance gap in the results table reflects an asymmetry in input
design, not the relative merit of the architectures. Statements of the form "ensemble methods
outperform recurrent networks on ISPU classification" are not supported by this experiment.

Making the comparison valid would require restricting all four models to lagged features so
that each solves the same next-day forecasting problem.

---

## 3. No baseline was established

Neither a trivial baseline nor a domain baseline was computed. This is the most easily
remedied gap.

| Missing baseline | What it would establish |
|---|---|
| Majority-class classifier | The floor. Would score 75.7% accuracy and F1 = 0 on the 2024 test set, showing directly why accuracy is uninformative here |
| Direct threshold rule — `max(sub-indices) > 100` | The ceiling for the tabular task. If it approaches F1 = 1.0, the leakage argument is demonstrated numerically rather than argued |
| Persistence — day *t* class equals day *t-1* class | The reference point for the forecasting task. Air quality is strongly autocorrelated, so persistence would likely be competitive with LSTM and GRU |

Without these, there is no way to tell whether an LSTM F1 of 0.4839 is poor, adequate, or
respectable **for the task it was actually given**.

`src/verify_leakage.py` in this repository computes the first two.

---

## 4. PM2.5 is entirely imputed for 2020

PM2.5 is missing on all 366 days of 2020 — 20.2% of the full research period. Every value
used for that year is an imputed month-specific median derived from 2021–2023.

PM10 is additionally missing on 159 days of 2023.

**Consequences.**
- No statement about PM2.5's contribution in 2020 describes the atmosphere; it describes the
  imputation scheme.
- The retrospective SHAP anomaly for 2020, where O3 displaces PM2.5 as the dominant feature,
  is traceable entirely to this and must not be reported as an air-quality finding.
- 366 of the 1,460 training rows (25%) contain a fully synthetic PM2.5 column, which is the
  single most influential feature in the model.

The missingness indicators partially mitigate this by letting the model distinguish observed
from reconstructed values, but they do not restore the lost information.

---

## 5. No hyperparameter search

All configurations were fixed a priori. No grid search, random search, or Bayesian
optimisation was performed for any of the four models.

The recurrent models are most affected: window length (14), hidden units (64), dropout rates
(0.25 / 0.15), learning rate (0.001) and batch size (32) were all set without tuning. Their
reported scores are a lower bound on what the architectures can achieve, not a fair ceiling.

---

## 6. Decision threshold fixed at 0.50

All four models classify at a probability of 0.50. For class-weighted models on an imbalanced
target with a base-rate shift between training (36.1%) and test (24.3%), 0.50 is unlikely to
be F1-optimal.

The effect is visible in the results. GRU achieves a **higher** ROC-AUC than LSTM (0.7159 vs
0.6902) but a **lower** F1 (0.4000 vs 0.4839), because it is systematically less confident and
predicts positive on only 32 days against LSTM's 97. Threshold tuning on the validation split
would likely reverse the F1 ordering without retraining either model.

Random Forest shows the mirror image: Precision (0.9767) exceeds Recall (0.9438), so the model
errs toward under-warning. For a public-health application that asymmetry is backwards — a
missed unhealthy day costs more than a spurious advisory — and a lower threshold would be the
appropriate correction.

---

## 7. A single train–test split

One training window (2020–2023) and one test year (2024). No walk-forward validation, no
rolling origin, no cross-validation adapted to time series.

**Consequences.**
- Every figure is a point estimate with no confidence interval.
- The results are conditioned on 2024 happening to be a relatively clean year (24.3% positive
  rate against 36.1% in training). A test year with a different base rate could produce
  materially different figures.
- Model ranking rests on a single comparison, which is fragile — particularly for the LSTM/GRU
  ordering, where the gap is small and threshold-dependent.

Rolling-origin validation (train 2020–2021 → test 2022; train 2020–2022 → test 2023; and so
on) would replace each point estimate with a distribution.

---

## 8. The `tahun` feature cannot generalise

Training covers 2020–2023; every test row carries `tahun=2024`, a value never observed during
training. Tree models handle this as an out-of-range split, and a model deployed on 2025 data
would face the same problem again, permanently.

Its SHAP contribution is negligible (0.0078, rank 9 of 17), so the practical impact on these
results is small. But the feature is methodologically unsound for any deployment scenario and
should be removed or replaced with a relative time index in future work.

---

## 9. Retrospective SHAP is fitted and explained on the same data

The diagnostic model in section 19.5 of the notebook is refitted on the complete imputed
2020–2024 period, and SHAP is computed over that same period. Imputation for that model is
also fitted on the full period rather than on training data only.

This is appropriate for a descriptive diagnostic and is labelled as such throughout, but it is
**not** a generalisation estimate and the year-by-year attribution values must not be quoted
as out-of-sample results.

---

## 10. Data-source constraints

The study uses secondary data and inherits whatever collection, calibration and publication
decisions were made upstream.

- The dataset was accessed via Kaggle rather than directly from Satu Data Jakarta, adding an
  intermediary step between source and analysis.
- The `stasiun` column carries five distinct labels across a file with exactly one row per
  day. The station attribution for any given day was not independently verified, and the study
  treats the series as a single city-level record.
- Sub-index conversion was performed upstream by the publisher using breakpoints and averaging
  windows that this analysis did not verify.
- No meteorological covariates are available — wind, temperature, humidity, rainfall and
  boundary-layer height are the dominant drivers of pollutant dispersion, and their absence is
  the main reason the forecasting task is hard.

---

## 11. Class collapse discards information

Four ISPU categories were collapsed into two. The binarisation was necessary — SANGAT TIDAK
SEHAT occurs on 7 days and BAIK on 50 across five years, too few for a stable four-class model
under a chronological split — but it has costs.

The model cannot distinguish a marginally unhealthy day from a severe one, and the 7 SANGAT
TIDAK SEHAT days are absorbed into a class dominated by the 609 TIDAK SEHAT days, so they
exert almost no influence on training. For operational use, severity gradation is precisely
what matters most.

---

## 12. Scope of generalisation

The reported performance reflects DKI Jakarta, the years 2020–2024, this specific feature
configuration and this train–test boundary. It should not be extended to other cities, other
periods, other monitoring networks, or other ISPU implementations without revalidation.

The 2020–2021 portion of the training data additionally covers pandemic-period mobility
restrictions, a regime that is unlikely to recur and that the `is_pasca_pandemi` indicator
only crudely captures.

---

## Future Work

Ordered by value relative to effort:

1. **Add the three baselines** in section 3. No training required; immediately establishes
   whether any reported score is meaningful.
2. **Tune the decision threshold** on the validation split for all four models.
3. **Reframe as next-day forecasting for every model**, restricting the tree-based models to
   lagged features so the comparison becomes like-for-like. This is the single change that
   would most improve the study.
4. **Adopt rolling-origin validation** to replace point estimates with distributions.
5. **Join meteorological covariates** from BMKG, which is what would make forecasting
   genuinely tractable.
6. **Compare imputation strategies** — forward fill, KNN, seasonal decomposition, or excluding
   2020 entirely — and report the sensitivity of results to that choice.
7. **Run a hyperparameter search**, particularly over window length and recurrent capacity.
8. **Revisit the four-class problem** with ordinal-aware methods that can exploit the natural
   ordering of the categories.
9. **Model the station dimension explicitly** if a true station-by-day panel can be obtained.
