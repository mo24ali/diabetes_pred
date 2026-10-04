# diabetes_pred

A small scikit-learn project that predicts diabetes from the Pima Indians Diabetes dataset.

## Data

`data/diabetes_dataset.csv` — 768 samples, 8 features, and a binary `Outcome` target
(`0` = non-diabetic, `1` = diabetic). Imbalanced: 500 negative, 268 positive.

## Workflow

Everything runs top to bottom in `diabetes_prediction.ipynb`:

1. Load the CSV and inspect it with `describe()` and `value_counts()`
2. Split into features `X` and target `Y`
3. Standardize features with `StandardScaler` (fit once, reused for the prediction input)
4. Train/test split: 80/20 with `random_state=2` (614 / 154 samples)
5. Train `sklearn.svm.SVC(kernel='linear')`
6. Score with `accuracy_score`, then predict on a single record

## Results

| Split | Accuracy |
| --- | --- |
| Training | 0.772 |
| Testing | 0.766 |

## Usage

```bash
python -m venv .venv
source .venv/bin/activate
pip install pandas numpy scikit-learn jupyter
jupyter notebook
```

The final cell takes a tuple in the order `Pregnancies, Glucose, BloodPressure,
SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age` and prints
`DIABETIC` or `Not DIABETIC`.