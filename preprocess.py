import pandas as pd

INPUT_FILE = "loan.csv"
OUTPUT_FILE = "cleaned_dataset.csv"

df = pd.read_csv(INPUT_FILE)
print("Original shape:", df.shape)

# Record missing-value counts before cleaning.
missing_report = df.isna().sum().rename("missing_count").to_frame()
missing_report["missing_percent"] = (missing_report["missing_count"] / len(df) * 100).round(3)
missing_report.to_csv("missing_values_report.csv")

# Drop rows with missing target labels instead of guessing the outcome.
if "loan_status" in df.columns:
    df = df.dropna(subset=["loan_status"])

# Fill missing numeric values with the column median.
for col in df.select_dtypes(include="number").columns:
    if df[col].isna().any():
        df[col] = df[col].fillna(df[col].median())

# Fill missing categorical values with the mode.
for col in df.select_dtypes(include=["object", "category", "bool"]).columns:
    if df[col].isna().any():
        modes = df[col].mode(dropna=True)
        df[col] = df[col].fillna(modes.iloc[0] if not modes.empty else "Unknown")

# Detect outliers with IQR on continuous numeric features only.
# ID is an identifier and loan_status is the target, so neither is filtered.
outlier_cols = [
    c for c in [
        "credit_score", "loan_amnt", "loan_percent_income",
        "person_age", "person_emp_exp", "person_income"
    ] if c in df.columns
]

keep_mask = pd.Series(True, index=df.index)
report = []
for col in outlier_cols:
    q1 = df[col].quantile(0.25)
    q3 = df[col].quantile(0.75)
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    is_outlier = (df[col] < lower) | (df[col] > upper)
    report.append({
        "column": col, "Q1": q1, "Q3": q3, "IQR": iqr,
        "lower_bound": lower, "upper_bound": upper,
        "outlier_count": int(is_outlier.sum())
    })
    keep_mask &= ~is_outlier

pd.DataFrame(report).to_csv("outlier_report.csv", index=False)

# Remove any row with an outlier in one or more selected features.
cleaned = df.loc[keep_mask].copy()
cleaned.to_csv(OUTPUT_FILE, index=False)

print("\nIQR outlier report:\n", pd.DataFrame(report))
print("\nRows after missing-value handling:", len(df))
print("Rows removed by outlier filtering:", len(df) - len(cleaned))
print("Final shape:", cleaned.shape)
print("Remaining missing cells:", int(cleaned.isna().sum().sum()))
print("Saved:", OUTPUT_FILE)
