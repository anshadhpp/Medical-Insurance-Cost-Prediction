# Medical Insurance Cost Prediction

Predicting individual medical insurance charges using a Random Forest Regressor, with a full exploratory data analysis (EDA) uncovering the key cost drivers behind health insurance premiums.

## Dataset

The dataset contains 1,338 records of individual medical insurance beneficiaries. After removing 1 duplicate row, 1,337 clean records were used for analysis.

| Attribute | Type | Description |
|---|---|---|
| `age` | Integer | Age of the primary beneficiary (18–64 years) |
| `sex` | Categorical | Gender: `male` or `female` |
| `bmi` | Float | Body Mass Index (15.96–53.13) |
| `children` | Integer | Number of children/dependents covered (0–5) |
| `smoker` | Categorical | Smoking status: `yes` or `no` |
| `region` | Categorical | Residential area: `northeast`, `northwest`, `southeast`, `southwest` |
| `charges` | Float | **Target variable** — medical costs billed by insurance ($1,121.87 – $63,770.43) |

## Exploratory Data Analysis

Key insights uncovered during EDA:

- **Charges are right-skewed** with a bimodal-like pattern, hinting at a hidden categorical driver.
- **Smoking status is the dominant cost separator** — smokers pay 3–5× more than non-smokers on average (median ~$34,000 vs ~$7,500).
- **Smoking × BMI interaction** — high BMI barely affects charges for non-smokers, but for smokers, BMI above the obesity threshold (30) leads to a sharp, near-exponential jump in cost.
- **Region has only a modest effect** (~$2,400 spread between highest and lowest average charges), making it a weak predictor compared to health-related features.

## Modeling

Based on the EDA, `sex`, `region`, and `children` were dropped as weak predictors. The final feature set (`age`, `bmi`, `smoker`) was fed into a `scikit-learn` pipeline:

- **Preprocessing:** numerical features passed through as-is; `smoker` one-hot encoded.
- **Model:** `RandomForestRegressor` (100 estimators, `random_state=42`).
- **Split:** 80/20 train-test split.

### Performance

| Metric | Value | Description |
|---|---|---|
| **R² Score** | **0.8787** | The model explains ~87.9% of the variance in insurance charges. |
| **MAE** | **$2,620.82** | On average, predictions are off by ~$2,620.82. |
| **RMSE** | **$4,721.32** | Penalizes larger errors more heavily; reasonable given charges up to $63,770. |

### Feature Importance

![Random Forest Feature Importances](https://drive.google.com/uc?export=view&id=1SSJSb_cwNXh6MkbnZdFAZJqGwQgm3SLw)

- **Smoking status (~60%)** is the single biggest driver of the model's predictions.
- **BMI (~26%)** matters, but its effect is concentrated among smokers.
- **Age (~14%)** contributes a steady, predictable increase in cost.

## Key Findings & Business Implications

1. **Smoking is the #1 cost driver**, creating a near-binary split between low-cost and high-cost customers.
2. **BMI matters most in combination with smoking** — a "smoking × obesity" interaction effect drives the highest charges.
3. **Age adds a modest, consistent increase**, most visible among non-smokers.
4. Risk-based pricing should weight smoking status and BMI heavily; wellness programs targeting smoking cessation and weight management would likely offer the best ROI in reducing claim costs. Regional pricing adjustments appear unnecessary given the weak regional signal.

## Project Structure

```
.
├── Health_Insurance.ipynb   # Full analysis: EDA, modeling, evaluation
├── insurance.csv            # Dataset
├── images/
│   └── feature_importance.png
├── requirements.txt
└── README.md
```

## Getting Started

```bash
git clone https://github.com/<your-username>/medical-insurance-cost-prediction.git
cd medical-insurance-cost-prediction
pip install -r requirements.txt
jupyter notebook Health_Insurance.ipynb
```

## Tech Stack

- Python, pandas, NumPy
- scikit-learn (RandomForestRegressor, Pipeline, ColumnTransformer)
- Matplotlib, Seaborn

## Author

Anshad
