# Task 2: Missing Values and Outlier Removal

## Objective
Use Pandas only to detect and handle missing values, detect outliers statistically, and save a cleaned dataset.

## Dataset summary
- Original dataset: 45,015 rows × 12 columns
- Missing cells in original dataset: 5
- Rows after handling missing target labels and imputing features: 45,014
- Rows in cleaned dataset: 38,040
- Rows removed by IQR outlier filtering: 6,974
- Missing cells in final dataset: 0

## Files
- `loan.csv`: original dataset.
- `preprocess.py`: Pandas-only cleaning script.
- `cleaned_dataset.csv`: final cleaned dataset.
- `missing_values_report.csv`: missing counts and percentages by original column.
- `outlier_report.csv`: IQR bounds and outlier counts for each selected numeric feature.
- `requirements.txt`: dependency list.
- `README.md`: documentation.

## 1. Missing values
- Rows with missing `loan_status` are dropped because a target label should not be guessed.
- Missing numeric values are filled with their column median.
- Missing categorical values are filled with their most frequent value (mode).

## 2. Outlier detection and removal
The Interquartile Range (IQR) method is applied to these continuous numeric features:
`credit_score`, `loan_amnt`, `loan_percent_income`, `person_age`, `person_emp_exp`, and `person_income`.

For each feature:
- IQR = Q3 − Q1
- Lower bound = Q1 − 1.5 × IQR
- Upper bound = Q3 + 1.5 × IQR

A value outside the bounds is considered an outlier. `ID` is excluded because it is an identifier; `loan_status` is excluded because it is the target label. A row is removed if any selected feature is an outlier.

## Run the project
Install Pandas:
```bash
python -m pip install pandas
```
Keep `loan.csv` in the same folder as `preprocess.py`, then run:
```bash
python preprocess.py
```

The script creates `cleaned_dataset.csv`, `missing_values_report.csv`, and `outlier_report.csv`.

## Pandas-only requirement
All data loading, missing-value handling, IQR calculations, filtering, and CSV output use Pandas. No scikit-learn or other data-cleaning library is used.
