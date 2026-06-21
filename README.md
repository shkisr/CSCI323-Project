# B2B Lead Score Model for Bizex
**CSCI323 Modern Artificial Intelligence — Spring 2026**
University of Wollongong in Dubai (UOWD)

--

---

## Project Overview
Bizex is a UAE-based B2B consultancy that receives inbound service enquiries through an online lead capture form. This project builds a machine learning pipeline that automatically scores and ranks incoming leads by their likelihood to convert, replacing manual follow-up decisions with a data-driven priority queue.

The pipeline runs in two stages:
1. **Unsupervised clustering** (K-Means + DBSCAN) — segments leads by activity type, location, and enquiry detail
2. **Supervised classification** (5 models compared) — predicts whether each lead is Interested, Warm Prospect, or Hot Prospect

---

## Results Summary
| Model | CV Accuracy | Weighted F1 | Mean AUC |
|---|---|---|---|
| SVM | 0.6399 | 0.49 | 0.551 |
| Logistic Regression | 0.6052 | 0.51 | 0.530 |
| KNN | 0.5843 | 0.49 | 0.561 |
| Decision Tree | 0.4438 | 0.35 | 0.587 |
| Random Forest | 0.4667 | 0.42 | 0.592 |

**Winner:** SVM with linear kernel, C=0.1 (selected via GridSearchCV)

**K-Means Clustering:** K=2, Silhouette=0.7552, Davies-Bouldin=0.1676

> Note: All models predicted only the Interested class on the 22-sample test set due to severe class imbalance (4 Warm, 4 Hot). ROC-AUC scores remain above 0.50 baseline across all classes, confirming meaningful signal was learned.

---

## Repository Structure
```
├── README.md
├── company.csv                  
└── company_5.ipynb
```

---

## Setup and Installation

### Requirements
- Python 3.8+
- Jupyter Notebook or JupyterLab

### Install dependencies
```bash
pip install pandas numpy scikit-learn matplotlib seaborn
```

### Run the notebook
```bash
jupyter notebook company_5.ipynb
```
Run all cells in order (Kernel → Restart & Run All).

All random seeds are fixed at `42` throughout — results are fully reproducible.

---

## Pipeline Steps

The notebook runs the following steps in sequence:

**Step 1 — Data Loading and Cleaning**
Loads `company.csv`, drops unstructured text columns (`full_name`, `write_about_your_business_in_short`), standardises column names, normalises lead status labels, removes duplicates, and applies One-Hot Encoding to `activity_type` and `location`.

**Step 2 — Train/Test Split**
80/20 stratified split with `random_state=42`.

**Step 3 — Unsupervised Clustering**
- K-Means: Elbow + Silhouette analysis to find optimal K, k-means++ initialisation, cluster enrichment analysis
- DBSCAN: k-distance graph for epsilon selection, noise point identification, sub-clustering of the "others" activity category
- Cluster labels added as one-hot features to the supervised feature set

**Step 4 — Baseline Model Comparison**
5-fold stratified cross-validation across all five classifiers inside sklearn Pipelines with ColumnTransformer preprocessor.

**Step 5 — Hyperparameter Tuning**
GridSearchCV on the winning baseline (SVM), outputs a 3×5 evaluation dashboard (confusion matrices, classification report heatmaps, ROC-AUC curves).

**Step 6 — Final Deployed Model**
Tuned SVM (linear, C=0.1) evaluated on held-out test set with full metrics dashboard.

---

## Data
`company.csv` contains inbound lead enquiries from Bizex's service form with the following columns:

| Column | Description |
|---|---|
| `full_name` | Lead name (dropped in preprocessing) |
| `select_the_type_of_activity` | Service/activity type → `activity_type` |
| `choose_your_preferred_location?` | Location preference → `location` |
| `write_about_your_business_in_short` | Free-text description (dropped; length used as feature) |
| `Lead Status` | Target variable: Interested / Warm Prospect / Hot Prospect |

**Dataset size after cleaning:** 108 records (70 Interested, 19 Warm, 19 Hot)

---

## Key Design Decisions
- `StandardScaler` applied only in the clustering stage (for distance-based algorithms); omitted in the supervised pipeline to preserve binary OHE structure
- Cluster assignments from K-Means encoded as one-hot features and passed into supervised models, connecting the two pipeline stages
- Ordinal encoding of target: Interested=0, Warm=1, Hot=2 (natural order preserved)
- All preprocessing inside sklearn `Pipeline` + `ColumnTransformer` to prevent data leakage across CV folds

---

## Limitations
- 108 records is too small for reliable minority class prediction — Warm and Hot F1 = 0.00 on test set
- K-Means Euclidean distance is suboptimal for one-hot encoded data (Hamming/Gower would be better)
- No SMOTE or class weighting applied — future work should address imbalance
- Batch pipeline only — no real-time scoring endpoint yet

---

## License
Academic project — not for commercial use.
