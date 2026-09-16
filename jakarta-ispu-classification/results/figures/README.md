# Figures

Selected figures exported from the notebook. Copy them here from `output/figures/` after
running the analysis, then reference them from the README.

Recommended set, in the order they are produced:

| File | Content |
|---|---|
| `01_distribusi_kategori_ispu_asli.png` | Original ISPU category distribution, 2020–2024 |
| `02_distribusi_target_biner.png` | Binary target distribution |
| `03_missing_values_per_kolom.png` | Missing values per pollutant |
| `04_heatmap_missing_tahun_polutan.png` | Missingness heatmap by year and pollutant |
| `05_perbandingan_performa_empat_model.png` | Four-model metric comparison |
| `06_confusion_matrix_random_forest.png` | Random Forest confusion matrix |
| `07_roc_curve_empat_model.png` | ROC curves, all four models |
| `08_precision_recall_curve_empat_model.png` | Precision–recall curves |
| `09_shap_summary_data_uji.png` | SHAP summary plot, test set 2024 |
| `10_shap_feature_importance_data_uji.png` | SHAP feature importance bar chart |
| `11_heatmap_shap_historis_tahun_polutan.png` | Retrospective SHAP heatmap by year |

The highest-value figures for a README are `05`, `06`, `09` and `11`. Keep the set small —
the complete visual output remains available in the notebook.

To embed one:

```markdown
![Model comparison](results/figures/05_perbandingan_performa_empat_model.png)
```
