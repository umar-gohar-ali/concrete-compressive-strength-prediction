# concrete-compressive-strength-prediction
An end-to-end machine learning pipeline for predicting concrete compressive strength using the [UCI Concrete Compressive Strength dataset](https://doi.org/10.24432/C5PK67). Eleven baseline models were benchmarked, the top five were tuned with Optuna, and the final model was interpreted using SHAP.

## Highlights

- **Best model:** CatBoost (baseline) — **R² = 0.9324**, **RMSE = 4.01 MPa**, generalisation gap of only 0.0016
- Engineered a **water-cement ratio** feature, identified via SHAP as one of the top predictors
- Hyperparameter tuning with **Optuna** (70 trials/model, TPE sampler) on CatBoost, LightGBM, XGBoost, Random Forest, and Gradient Boosting
- Model interpretability via **SHAP TreeExplainer** (beeswarm + dependence plots)
- Key drivers of strength: **age**, **water-cement ratio**, and **blast furnace slag**

## Repository Structure

```
├── 01_Dataset
│   ├── Processed              # Cleaned, split, and scaled data
│   └── Raw                    # Original dataset
├── 02_Codes
│   ├── exploatory data analysis   # eda.ipynb
│   ├── modelling               # modelling.ipynb
│   └── pre processing          # preprocessing.ipynb
├── 04_Models
│   ├── baseline                # 11 baseline models (.pkl)
│   └── tuned                   # 5 Optuna-tuned models (.pkl)
├── 05_Results
│   ├── metrics                 # Performance CSVs, best model info
│   ├── optuna_studies           # Saved Optuna study objects
│   └── plots
│       ├── model_performance    # Optuna tuning visualizations (HTML)
│       ├── pre-processing        # EDA plots
│       └── shap                  # SHAP summary & dependence plots
├── 06_Report
│   └── Concrete_Strength_Prediction_Report.pdf
└── Profiling_Report_EDA.html.              # Exploatory Data ANALYSIS report, all stats and plots included
```

## Methodology

1. **Preprocessing** — Removed duplicates, engineered water-cement ratio, split data (80/20, seed=2006), standardised features for distance-based models.
2. **Baseline Modelling** — Trained 11 models (linear, distance-based, tree-based) with default hyperparameters.
3. **Hyperparameter Tuning** — Optimised top 5 models using Optuna (70 trials each).
4. **Explainability** — Interpreted the final CatBoost model with SHAP.

## Results

| Model | Type | R² | RMSE (MPa) | Gap |
|---|---|---|---|---|
| CatBoost | Baseline | 0.9324 | 4.0119 | 0.00164 |
| GradientBoosting | Tuned | 0.9295 | 4.0976 | 0.00536 |
| XGBoost | Tuned | 0.9203 | 4.3578 | 0.0169 |
| LightGBM | Baseline | 0.9193 | 4.3853 | 0.00534 |
| RandomForest | Tuned | 0.8988 | 4.9114 | 0.00589 |

## Tools & Libraries

Python · Scikit-learn · XGBoost · LightGBM · CatBoost · Optuna · SHAP

## Report

Full write-up available in [`06_Report/Concrete_Strength_Prediction_Report.pdf`](./06_Report/Concrete_Strength_Prediction_Report.pdf).

## Author

Umar Gohar Ali
