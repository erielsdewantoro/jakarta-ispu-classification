# Dataset

This directory documents the dataset. **The raw CSV is not committed to this repository.**

---

## File

`ispu_dki_all.csv`

## Source

Daily Indeks Standar Pencemar Udara (ISPU) records for DKI Jakarta, published by
**Dinas Lingkungan Hidup Provinsi DKI Jakarta** via **Satu Data Jakarta**. The version used in
this research was accessed through Kaggle.

## How to obtain it

1. Search Kaggle for the DKI Jakarta ISPU dataset, or retrieve the source data directly from
   Satu Data Jakarta.
2. Download the CSV and rename it to `ispu_dki_all.csv` if necessary.
3. Place it at the repository root or in this `data/` directory. The notebook searches, in
   order: `./ispu_dki_all.csv`, `./dataset/ispu_dki_all.csv`, `/content/ispu_dki_all.csv`,
   `/content/dataset/ispu_dki_all.csv`.
4. On Google Colab, if the file is not found the notebook opens an upload prompt automatically.

The dataset is excluded from version control via `.gitignore`. Check the terms of the original
publisher before redistributing the data or any derivative of it.

---

## Raw file characteristics

| Property | Value |
|---|---|
| Rows | 5,538 |
| Columns | 11 |
| Date range | 2010-01-01 → 2025-02-28 |
| Distinct `stasiun` labels | 5 |
| Granularity | One row per calendar day |

The span 2010-01-01 → 2025-02-28 is exactly 5,538 days, confirming one row per day. The file
is a single daily series, not a station-by-day panel, which is why `stasiun` is excluded from
the feature set.

---

## Data dictionary

| Column | Type | Unit | Description |
|---|---|---|---|
| `tanggal` | date | — | Observation date, one row per day |
| `stasiun` | string | — | Monitoring station label (DKI1 Bunderan HI, DKI2 Kelapa Gading, DKI3 Jagakarsa, DKI4 Lubang Buaya, DKI5 Kebon Jeruk). Excluded from features |
| `pm25` | numeric | **ISPU sub-index** | Fine particulate matter sub-index, 0–500 scale |
| `pm10` | numeric | **ISPU sub-index** | Coarse particulate matter sub-index |
| `so2` | numeric | **ISPU sub-index** | Sulphur dioxide sub-index |
| `co` | numeric | **ISPU sub-index** | Carbon monoxide sub-index |
| `o3` | numeric | **ISPU sub-index** | Ground-level ozone sub-index |
| `no2` | numeric | **ISPU sub-index** | Nitrogen dioxide sub-index |
| `max` | numeric | ISPU | Highest sub-index for the day — the published ISPU value. **Excluded: direct leakage** |
| `critical` | string | — | Pollutant producing `max`. **Excluded: direct leakage** |
| `categori` | string | — | ISPU category label. Source of the target |

> **These are sub-indices, not concentrations.** The pollutant columns hold values already
> converted onto the common ISPU scale, not measurements in µg/m³ or ppm. This distinction is
> central to how the results should be read — see [`../docs/task-framing.md`](../docs/task-framing.md).

---

## ISPU category thresholds

The daily category is assigned from `max` by fixed thresholds:

| ISPU value | Category | In this study |
|---|---|---|
| 0 – 50 | BAIK | Bukan Tidak Sehat+ (0) |
| 51 – 100 | SEDANG | Bukan Tidak Sehat+ (0) |
| 101 – 200 | TIDAK SEHAT | **Tidak Sehat+ (1)** |
| 201 – 300 | SANGAT TIDAK SEHAT | **Tidak Sehat+ (1)** |
| > 300 | BERBAHAYA | Not observed in the research period |

The binary boundary therefore sits at ISPU = 100, which is also the point at which
public-health advisories change.

---

## Research subset

| Step | Rows |
|---|---:|
| Raw file | 5,538 |
| Filtered to 2020-01-01 → 2024-12-31 | 1,827 |
| After removing one `TIDAK ADA DATA` row | **1,826** |

### Category distribution, 2020–2024

| Category | Days |
|---|---:|
| SEDANG | 1,160 |
| TIDAK SEHAT | 609 |
| BAIK | 50 |
| SANGAT TIDAK SEHAT | 7 |
| TIDAK ADA DATA (removed) | 1 |

### Binary target distribution

| Target | Days | Share |
|---|---:|---:|
| Bukan Tidak Sehat+ | 1,210 | 66.27% |
| Tidak Sehat+ | 616 | 33.73% |

### Chronological split

| Subset | Period | Days | Tidak Sehat+ | Positive rate |
|---|---|---:|---:|---:|
| Train | 2020-01-01 → 2023-12-31 | 1,460 | 527 | 0.361 |
| Test | 2024-01-01 → 2024-12-31 | 366 | 89 | 0.243 |

---

## Missing values in the research period

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

PM2.5 is missing on **all 366 days of 2020**. Every 2020 PM2.5 value used in the analysis is
imputed. See [`../docs/limitations.md`](../docs/limitations.md) for what this rules out.

---

## Engineered features

Seventeen features, all derived in the notebook.

| Feature | Group | Definition |
|---|---|---|
| `pm25`, `pm10`, `so2`, `co`, `o3`, `no2` | Pollutant | Sub-index values, imputed |
| `tahun` | Temporal | Calendar year |
| `bulan` | Temporal | Month 1–12, seasonality proxy |
| `hari_dalam_minggu` | Temporal | Day of week, 0 = Monday … 6 = Sunday |
| `is_weekend` | Temporal | 1 if Saturday or Sunday |
| `is_pasca_pandemi` | Temporal | 1 if year ≥ 2022 |
| `{pollutant}_was_missing` | Missingness | 1 if the value was null before imputation (six features) |

| Target column | Definition |
|---|---|
| `target_tidak_sehat` | 1 for TIDAK SEHAT or SANGAT TIDAK SEHAT, else 0 |
| `target_label` | Text label — "Tidak Sehat+" or "Bukan Tidak Sehat+" |
