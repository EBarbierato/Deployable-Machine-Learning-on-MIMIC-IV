# Deployable-Machine-Learning-on-MIMIC-IV
Deployable Machine Learning on MIMIC-IV_Leakage-Safe Prediction and Calibration for Incident Diabetes

.
├── scripts/
│   ├── 00_... (optional upstream steps if provided)
│   ├── 06_train_baselines.py
│   ├── 09_calibrate_and_thresholds.py
│   ├── 10_explainability.py
│   └── utils/... (helpers if present)
├── config/
│   ├── project.yaml             # global paths and switches
│   └── tasks.yaml               # task definitions (e.g., diab_incident_365d)
├── tools/
│   └── make_paper_figs_zip.sh   # helper to bundle figures
├── README.md
└── LICENSE
$OUT/
├── baselines/
│   └── diab_incident_365d/
│       ├── logistic/
│       │   ├── model.joblib
│       │   ├── metrics.json
│       │   ├── val_preds.parquet
│       │   └── test_preds.parquet
│       └── random_forest/
│           ├── model.joblib
│           ├── metrics.json
│           ├── val_preds.parquet
│           └── test_preds.parquet
├── calibration/
│   ├── logistic/
│   │   ├── val_reliability_raw.png
│   │   └── test_reliability_cal.png
│   └── random_forest/
│       └── test_reliability_cal.png
├── dca/
│   ├── diab_incident_365d_logistic_cal/decision_curve.png
│   └── diab_incident_365d_random_forest_cal/decision_curve.png
├── explain/
│   ├── diab_incident_365d_logistic/
│   │   ├── shap_bar_logistic.png
│   │   ├── shap_summary_logistic.png
│   │   ├── top30_permutation_importance.png
│   │   └── shap_global.csv / explain_global.csv
│   └── diab_incident_365d_random_forest/
│       └── top30_perm_importance_random_forest.png
├── pr_overlay_calibrated_models.png
└── roc_overlay_calibrated_models.png


Requirements

Python ≥ 3.10

Recommended package versions:

numpy ≥ 1.23

pandas ≥ 1.5

pyarrow ≥ 10

scikit-learn 1.7.2 (match the model’s saved version to avoid unpickle warnings)

matplotlib ≥ 3.7

joblib ≥ 1.3

shap ≥ 0.44

NOTE If you see InconsistentVersionWarning when loading a model (e.g., trained on sklearn 1.7.2 but you have 1.7.0), install the same sklearn version used at training time to avoid surprises.

3) Data Inputs

You need pre-extracted Parquet files for:

Features (train/val/test) with admission-time features only (temporal guard applied)

Labels dataframe with the target column (e.g., diab_incident_365d)

Typical locations (adjust to your environment):

$OUT/imputed/X_train.parquet
$OUT/imputed/X_val.parquet
$OUT/imputed/X_test.parquet
$OUT/labels/labels.parquet


The labels parquet must include a boolean/0-1 column named by --label-col (e.g., diab_incident_365d).
Splits must be patient-level disjoint.

4) Configuration

Edit config/project.yaml and config/tasks.yaml as needed. Minimal parameters are passed via CLI flags (shown below). If your scripts also read YAML, keep paths consistent.

A simple mental model:

project.yaml → where $OUT lives and global toggles

tasks.yaml → which target(s) to run (e.g., diab_incident_365d), model grids, etc.

5) Environment Variable

Set your artifacts directory once:

export OUT=/absolute/path/to/comorbidity_artifacts
mkdir -p "$OUT"

6) Train Baselines (Nested CV selection on training only)

Trains Logistic Regression and Random Forest with model-specific preprocessing and writes predictions on val/test.

python scripts/06_train_baselines.py \
  --x-train "$OUT/imputed/X_train.parquet" \
  --x-val   "$OUT/imputed/X_val.parquet" \
  --x-test  "$OUT/imputed/X_test.parquet" \
  --labels  "$OUT/labels/labels.parquet" \
  --label-col diab_incident_365d \
  --models "logistic,random_forest" \
  --select-metric auroc \
  --outdir "$OUT/baselines"


Outputs per model (under $OUT/baselines/diab_incident_365d/<model>/):

model.joblib

metrics.json (CV selection summary)

val_preds.parquet, test_preds.parquet (ids, raw scores, labels)

7) Calibration & Thresholding (Validation-selected calibrator, frozen on test)

Calibrate probabilities (Platt & isotonic), select by Brier on validation, then apply to test. Also produces reliability diagrams and overlay ROC/PR for calibrated models.

python scripts/09_calibrate_and_thresholds.py \
  --val-preds "$OUT/baselines/diab_incident_365d/logistic/val_preds.parquet,$OUT/baselines/diab_incident_365d/random_forest/val_preds.parquet" \
  --test-preds "$OUT/baselines/diab_incident_365d/logistic/test_preds.parquet,$OUT/baselines/diab_incident_365d/random_forest/test_preds.parquet" \
  --task-col task \
  --methods "platt,isotonic" \
  --ece-bins 10 \
  --outdir "$OUT"


Key outputs

$OUT/calibration/logistic/val_reliability_raw.png

$OUT/calibration/logistic/test_reliability_cal.png

$OUT/calibration/random_forest/test_reliability_cal.png

$OUT/roc_overlay_calibrated_models.png

$OUT/pr_overlay_calibrated_models.png

8) Explainability (Permutation importance & SHAP)

Generates permutation importance for both models and SHAP for logistic regression. Make sure your numeric columns are true numeric (not pandas Int64Dtype etc.).

Logistic Regression
python scripts/10_explainability.py \
  --x-val "$OUT/imputed/X_val.parquet" \
  --labels "$OUT/labels/labels.parquet" \
  --label-col diab_incident_365d \
  --model-type logistic \
  --model-path "$OUT/baselines/diab_incident_365d/logistic/model.joblib" \
  --metric auprc \
  --top-k 30 \
  --shap True --shap-n-sample 2000 --shap-bg 200 \
  --outdir "$OUT/explain/diab_incident_365d_logistic"

Random Forest (permutation only)
python scripts/10_explainability.py \
  --x-val "$OUT/imputed/X_val.parquet" \
  --labels "$OUT/labels/labels.parquet" \
  --label-col diab_incident_365d \
  --model-type random_forest \
  --model-path "$OUT/baselines/diab_incident_365d/random_forest/model.joblib" \
  --metric auprc \
  --top-k 30 \
  --shap False \
  --outdir "$OUT/explain/diab_incident_365d_random_forest"


Key outputs

Logistic: shap_bar_logistic.png, shap_summary_logistic.png, top30_permutation_importance.png, shap_global.csv, explain_global.csv

RF: top30_perm_importance_random_forest.png

9) Decision-Curve Analysis (Net Benefit)

If DCA is integrated into 09_calibrate_and_thresholds.py, the following PNGs are already produced:

$OUT/dca/diab_incident_365d_logistic_cal/decision_curve.png
$OUT/dca/diab_incident_365d_random_forest_cal/decision_curve.png


If your DCA step is separate, refer to scripts/ for a dca_* script and pass the calibrated test predictions.

10) Collect Figures for the Paper

We provide a helper shell script. Adjust the internal list if your figure paths differ.

bash tools/make_paper_figs_zip.sh "$OUT"  # writes comorbidity_figures_YYYYMMDD_HHMM.zip next to $OUT


The default list (preserve/extend as needed):

pr_overlay_calibrated_models.png

roc_overlay_calibrated_models.png

dca/diab_incident_365d_logistic_cal/decision_curve.png

dca/diab_incident_365d_random_forest_cal/decision_curve.png

calibration/logistic/val_reliability_raw.png

calibration/logistic/test_reliability_cal.png

calibration/random_forest/test_reliability_cal.png

explain/diab_incident_365d_logistic/shap_bar_logistic.png

explain/diab_incident_365d_logistic/shap_summary_logistic.png

explain/diab_incident_365d_logistic/top30_permutation_importance.png

explain/diab_incident_365d_random_forest/top30_perm_importance_random_forest.png

11) Reproducibility & Leakage Safety

Temporal guard: features use only data with timestamps ≤ index admission time.

Patient-level splits: no subject appears in more than one partition.

Model-specific preprocessing: scaling for LR; no scaling for RF.

Selection protocol: nested CV on training only; validation is used solely to choose calibrator; test is untouched until the end.

Calibration: Platt vs isotonic picked by Brier on validation, then frozen and applied to test.

12) Troubleshooting

A) sklearn warning on unpickle version

InconsistentVersionWarning: Trying to unpickle estimator LogisticRegression from version 1.7.2 ...


Install the exact sklearn version used to train:
pip install scikit-learn==1.7.2

B) SHAP errors (e.g., dtype / additivity / deprecations)

Ensure all columns passed to SHAP are true numeric (float64/int64). Cast if needed:

df = df.apply(pd.to_numeric, errors='coerce').astype('float64')


Avoid passing pandas Int64Dtype() extension dtypes.

If LinearExplainer.explain_row(..., check_additivity=...) errors, remove the check_additivity kwarg for recent SHAP versions.

The deprecation warning about feature_perturbation is harmless; recent versions use maskers.

C) Permutation importance complaining about scorer args
If you see errors like:

average_precision_score() got an unexpected keyword argument 'needs_threshold'


make sure you build scorers using sklearn.metrics.make_scorer(average_precision_score, needs_threshold=True) only where appropriate, or use the provided script which maps auprc to the correct scorer signature.

D) No DCA CSVs found
Some pipelines output only PNGs. That’s expected if you didn’t enable CSV export. The figures are sufficient for the paper.
