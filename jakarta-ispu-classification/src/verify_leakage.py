"""
verify_leakage.py
=================

Reproduces the target-leakage evidence described in `docs/task-framing.md`, and computes the
baselines missing from the main experiment.

The claim under test is that the ISPU category is a deterministic threshold function of the
pollutant sub-indices that were used as model features, so the reported Random Forest and
XGBoost scores measure rule reconstruction rather than air-quality prediction.

Three checks are performed.

1. IDENTITY CHECK
   Verify, row by row, that `max` equals the maximum of the six sub-index columns, that
   `critical` names the column producing it, and that `categori` is the threshold bucket of
   `max`. If the identity holds, the target is fully determined by the features.

2. RULE BASELINE
   Classify the 2024 test set using `max(sub-indices) > 100` alone, with no model. An F1 near
   1.0 demonstrates the leakage numerically. Compare against the reported Random Forest F1 of
   0.9600.

3. TRIVIAL BASELINES
   Majority-class and persistence baselines, which establish the floor for the tabular task
   and the reference point for the next-day forecasting task respectively.

Usage
-----
    python src/verify_leakage.py                       # looks for ispu_dki_all.csv nearby
    python src/verify_leakage.py path/to/ispu_dki_all.csv

The script is read-only and writes nothing.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score

START_DATE = "2020-01-01"
END_DATE = "2024-12-31"
TEST_START = "2024-01-01"
POLLUTANTS = ["pm25", "pm10", "so2", "co", "o3", "no2"]
POSITIVE_CATEGORIES = ["TIDAK SEHAT", "SANGAT TIDAK SEHAT"]
UNHEALTHY_THRESHOLD = 100

CANDIDATES = [
    Path("ispu_dki_all.csv"),
    Path("data/ispu_dki_all.csv"),
    Path("../ispu_dki_all.csv"),
    Path("/content/ispu_dki_all.csv"),
]


def locate_dataset(argv: list[str]) -> Path:
    if len(argv) > 1:
        path = Path(argv[1])
        if not path.exists():
            raise FileNotFoundError(f"Dataset not found at {path}")
        return path
    for candidate in CANDIDATES:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        "ispu_dki_all.csv not found. Pass the path explicitly:\n"
        "    python src/verify_leakage.py path/to/ispu_dki_all.csv"
    )


def rule(text: str) -> None:
    print(f"\n{'=' * 78}\n{text}\n{'=' * 78}")


def load(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["tanggal"] = pd.to_datetime(df["tanggal"], errors="coerce")
    df = df.loc[(df["tanggal"] >= START_DATE) & (df["tanggal"] <= END_DATE)]
    df = df.loc[df["categori"] != "TIDAK ADA DATA"]
    df = df.sort_values("tanggal").reset_index(drop=True)
    for col in POLLUTANTS + ["max"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["target"] = df["categori"].isin(POSITIVE_CATEGORIES).astype(int)
    return df


# --------------------------------------------------------------------------------------
# Check 1 — the deterministic identity
# --------------------------------------------------------------------------------------
def check_identity(df: pd.DataFrame) -> None:
    rule("CHECK 1  Is `max` the maximum of the six sub-indices?")

    computed_max = df[POLLUTANTS].max(axis=1, skipna=True)
    comparable = df["max"].notna() & computed_max.notna()
    agree = np.isclose(df.loc[comparable, "max"], computed_max[comparable])

    n = int(comparable.sum())
    n_agree = int(agree.sum())
    print(f"  Rows compared      : {n:,}")
    print(f"  max == max(sub-idx): {n_agree:,}  ({n_agree / n:.2%})")

    if n_agree < n:
        print(f"\n  {n - n_agree} mismatching rows (first 5):")
        mismatch = df.loc[comparable].loc[~agree.values]
        print(mismatch[["tanggal"] + POLLUTANTS + ["max", "critical", "categori"]].head())

    rule("CHECK 1b  Does `critical` name the argmax column?")
    argmax_col = df[POLLUTANTS].idxmax(axis=1, skipna=True).str.upper()
    critical_norm = df["critical"].astype(str).str.upper().str.replace(".", "", regex=False)
    argmax_norm = argmax_col.str.replace("PM25", "PM25", regex=False)
    match = (critical_norm.str.replace("PM2.5", "PM25", regex=False) == argmax_norm)
    print(f"  critical == argmax : {int(match.sum()):,} / {len(df):,}  ({match.mean():.2%})")
    print("  (Small deviations are expected where several sub-indices tie at the maximum.)")

    rule("CHECK 1c  Is `categori` the threshold bucket of `max`?")
    bucket = pd.cut(
        df["max"],
        bins=[-np.inf, 50, 100, 200, 300, np.inf],
        labels=["BAIK", "SEDANG", "TIDAK SEHAT", "SANGAT TIDAK SEHAT", "BERBAHAYA"],
    ).astype(str)
    agree_cat = bucket == df["categori"].astype(str)
    print(f"  bucket == categori : {int(agree_cat.sum()):,} / {len(df):,}  ({agree_cat.mean():.2%})")

    print(
        "\n  CONCLUSION: if the three checks above agree at or near 100%, the target is a\n"
        "  deterministic function of the pollutant sub-indices used as model features.\n"
        "  Dropping `max` and `critical` removes the explicit answer but not the dependency."
    )


# --------------------------------------------------------------------------------------
# Checks 2 and 3 — baselines
# --------------------------------------------------------------------------------------
def scores(name: str, y_true, y_pred) -> dict:
    return {
        "baseline": name,
        "precision": round(precision_score(y_true, y_pred, zero_division=0), 4),
        "recall": round(recall_score(y_true, y_pred, zero_division=0), 4),
        "f1_score": round(f1_score(y_true, y_pred, zero_division=0), 4),
        "accuracy": round(float((np.asarray(y_true) == np.asarray(y_pred)).mean()), 4),
    }


def check_baselines(df: pd.DataFrame) -> None:
    rule("CHECKS 2 & 3  Baselines on the 2024 test set")

    test = df.loc[df["tanggal"] >= TEST_START].reset_index(drop=True)
    y_true = test["target"].values
    print(f"  Test days: {len(test):,}   Tidak Sehat+: {int(y_true.sum()):,} "
          f"({y_true.mean():.2%})")

    rows = []

    # Rule baseline, no model at all.
    rule_pred = (test[POLLUTANTS].max(axis=1, skipna=True) > UNHEALTHY_THRESHOLD).astype(int)
    rows.append(scores("Threshold rule: max(sub-indices) > 100", y_true, rule_pred))

    # Majority class.
    rows.append(scores("Majority class (always negative)", y_true, np.zeros_like(y_true)))

    # Persistence: today's class equals yesterday's. Seeded from the last training day.
    prior = df.loc[df["tanggal"] < TEST_START, "target"].iloc[-1]
    persistence = np.concatenate([[prior], y_true[:-1]])
    rows.append(scores("Persistence (yesterday's class)", y_true, persistence))

    baselines = pd.DataFrame(rows)
    print()
    print(baselines.to_string(index=False))

    print("\n  Reported models, for comparison:")
    reported = pd.DataFrame([
        {"baseline": "Random Forest (reported)", "precision": 0.9767, "recall": 0.9438,
         "f1_score": 0.9600, "accuracy": round(359 / 366, 4)},
        {"baseline": "LSTM (reported, next-day)", "precision": 0.4639, "recall": 0.5056,
         "f1_score": 0.4839, "accuracy": round(270 / 366, 4)},
        {"baseline": "GRU (reported, next-day)", "precision": 0.4507, "recall": 0.3596,
         "f1_score": 0.4000, "accuracy": round(270 / 366, 4)},
    ])
    print()
    print(reported.to_string(index=False))

    print(
        "\n  HOW TO READ THIS\n"
        "  - If the threshold rule reaches F1 near 1.00 with no model, the tabular task is\n"
        "    rule recovery, not prediction, and Random Forest's 0.9600 is an upper-bound\n"
        "    approximation of an arithmetic identity.\n"
        "  - Compare persistence against LSTM and GRU. Both solve next-day forecasting, so\n"
        "    persistence is their fair reference point, not Random Forest."
    )


def main() -> None:
    path = locate_dataset(sys.argv)
    print(f"Dataset: {path.resolve()}")
    df = load(path)
    print(f"Research-period rows after cleaning: {len(df):,}")
    check_identity(df)
    check_baselines(df)
    rule("Done. See docs/task-framing.md for the full argument.")


if __name__ == "__main__":
    main()
