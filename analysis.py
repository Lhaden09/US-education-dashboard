"""Reusable data loading, cleaning checks and analysis for the U.S. K-12 Education Trends project."""
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats

DATA_FILE = "states_all_clean.csv"
SCORE_COLUMNS = {
    "Grade 4 Mathematics": "AVG_MATH_4_SCORE",
    "Grade 8 Mathematics": "AVG_MATH_8_SCORE",
    "Grade 4 Reading": "AVG_READING_4_SCORE",
    "Grade 8 Reading": "AVG_READING_8_SCORE",
}
FINANCIAL_COLUMNS = [
    "TOTAL_REVENUE", "TOTAL_EXPENDITURE", "FEDERAL_REVENUE",
    "STATE_REVENUE", "LOCAL_REVENUE", "INSTRUCTION_EXPENDITURE",
]

def load_data(path=None):
    """Load the prepared state-year dataset and enforce core data types."""
    path = Path(path) if path else Path(__file__).parent / DATA_FILE
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()
    df["STATE"] = df["STATE"].astype("string").str.strip().str.upper()
    df["YEAR"] = pd.to_numeric(df["YEAR"], errors="coerce")
    df = df.dropna(subset=["STATE", "YEAR"]).copy()
    df["YEAR"] = df["YEAR"].astype(int)
    return df.sort_values(["STATE", "YEAR"]).reset_index(drop=True)

def data_quality_summary(df):
    return {
        "rows": int(len(df)),
        "columns": int(df.shape[1]),
        "states": int(df["STATE"].nunique()),
        "first_year": int(df["YEAR"].min()),
        "last_year": int(df["YEAR"].max()),
        "duplicate_state_year": int(df.duplicated(["STATE", "YEAR"]).sum()),
        "missing_cells": int(df.isna().sum().sum()),
        "missing_percent": float(df.isna().sum().sum() / (df.shape[0] * df.shape[1]) * 100),
    }

def missing_table(df):
    return (df.isna().sum().rename("Missing count").to_frame()
            .assign(**{"Missing (%)": lambda x: x["Missing count"] / len(df) * 100})
            .sort_values("Missing (%)", ascending=False))

def national_trends(df):
    """Aggregate state totals by year; achievement values are unweighted state means."""
    cols = {
        "Enrollment": ("ENROLL_CLEAN", "sum"),
        "Total revenue": ("TOTAL_REVENUE", "sum"),
        "Total expenditure": ("TOTAL_EXPENDITURE", "sum"),
        "Grade 4 math": ("AVG_MATH_4_SCORE", "mean"),
        "Grade 8 math": ("AVG_MATH_8_SCORE", "mean"),
        "Grade 4 reading": ("AVG_READING_4_SCORE", "mean"),
        "Grade 8 reading": ("AVG_READING_8_SCORE", "mean"),
        "Spending per student": ("EXP_PER_STUDENT", "mean"),
    }
    result = df.groupby("YEAR").agg(**{
        name: pd.NamedAgg(column=column, aggfunc=agg)
        for name, (column, agg) in cols.items()
    }).reset_index()
    return result

def state_comparison(df, year, metric):
    if metric not in df.columns:
        raise ValueError(f"Unknown metric: {metric}")
    return (df.loc[df["YEAR"] == year, ["STATE", metric]]
            .dropna().sort_values(metric, ascending=False))

def descriptive_statistics(df):
    cols = ["ENROLL_CLEAN", "TOTAL_REVENUE", "TOTAL_EXPENDITURE",
            "EXP_PER_STUDENT", *SCORE_COLUMNS.values()]
    return df[cols].describe().T

def outlier_summary(df, columns=None):
    """IQR flags are reported, not automatically deleted: large states can be genuine outliers."""
    columns = columns or ["ENROLL_CLEAN", "TOTAL_REVENUE", "TOTAL_EXPENDITURE", "EXP_PER_STUDENT"]
    rows = []
    for col in columns:
        s = df[col].dropna()
        q1, q3 = s.quantile([.25, .75])
        iqr = q3 - q1
        lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        rows.append({"Variable": col, "Valid values": len(s),
                     "Potential outliers": int(((s < lower) | (s > upper)).sum()),
                     "Lower IQR limit": lower, "Upper IQR limit": upper})
    return pd.DataFrame(rows)

def paired_change_test(df, column):
    """Paired t-test for states with values in both their first and last available year."""
    sub = df[["STATE", "YEAR", column]].dropna()
    first_year, last_year = int(sub.YEAR.min()), int(sub.YEAR.max())
    first = sub[sub.YEAR == first_year].set_index("STATE")[column]
    last = sub[sub.YEAR == last_year].set_index("STATE")[column]
    paired = pd.concat([first.rename("first"), last.rename("last")], axis=1).dropna()
    if len(paired) < 2:
        return {"first_year": first_year, "last_year": last_year, "n_states": len(paired),
                "mean_first": np.nan, "mean_last": np.nan, "t": np.nan, "p": np.nan}
    test = stats.ttest_rel(paired["first"], paired["last"])
    return {"first_year": first_year, "last_year": last_year, "n_states": len(paired),
            "mean_first": paired["first"].mean(), "mean_last": paired["last"].mean(),
            "t": float(test.statistic), "p": float(test.pvalue)}

def correlation_summary(df):
    rows = []
    for label, score_col in SCORE_COLUMNS.items():
        pair = df[["EXP_PER_STUDENT", score_col]].dropna()
        if len(pair) >= 3 and pair["EXP_PER_STUDENT"].nunique() > 1 and pair[score_col].nunique() > 1:
            r, p = stats.pearsonr(pair["EXP_PER_STUDENT"], pair[score_col])
        else:
            r, p = np.nan, np.nan
        rows.append({"Achievement measure": label, "N": len(pair), "Pearson r": r, "p-value": p})
    return pd.DataFrame(rows)
