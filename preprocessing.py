"""Preprocessing helpers extracted from the Streamlit app.

Contains: normalize_columns, find_column
"""
from typing import List, Optional
import pandas as pd


def find_column(df: pd.DataFrame, variants: List[str]) -> Optional[str]:
    """Find the first column name in df that contains any of the given variants (case-insensitive)."""
    for variant in variants:
        for col in df.columns:
            if variant.lower() in col.lower():
                return col
    return None


def normalize_columns(df: pd.DataFrame, for_prediction: bool = False) -> pd.DataFrame:
    """Normalize commonly named columns to canonical names used by the app.

    - Status -> 'Status' (0/1)
    - HasSideJob -> 'HasSideJob' (0/1)
    - MonthlyFamilyIncome -> scaled 0..1 or categorical codes if non-numeric

    The function tries to be tolerant of alternative names.
    """
    col_map = {}
    lower_cols = {c.lower(): c for c in df.columns}

    def find_variant(possible):
        for p in possible:
            if p.lower() in lower_cols:
                return lower_cols[p.lower()]
        for col in df.columns:
            lower_col = col.lower()
            for p in possible:
                if p.lower() in lower_col:
                    return col
        return None

    # common variants
    status_col = find_variant(['Status', 'studentstatus', 'student_status', 'enrollment_status', 'state'])
    hasjob_col = find_variant(['hasjob', 'HasSideJob', 'has_side_jobs', 'hasjobs', 'hassidejobs', 'employed', 'HasSideJobs'])
    income_col = find_variant(['MonthlyFamilyIncome', 'income', 'pay', 'wage', 'compensation'])

    if status_col:
        col_map[status_col] = 'Status'
    if hasjob_col:
        col_map[hasjob_col] = 'HasSideJob'
    if income_col:
        col_map[income_col] = 'MonthlyFamilyIncome'

    if col_map:
        df = df.rename(columns=col_map)

    # Only normalize Status if NOT doing prediction (to avoid data leakage)
    if not for_prediction and 'Status' in df.columns:
        if pd.api.types.is_numeric_dtype(df['Status']):
            df['Status'] = df['Status'].fillna(0).astype(int).clip(0, 1)
        else:
            df['Status'] = df['Status'].astype(str).str.lower().str.strip().apply(
                lambda x: 1 if 'enrolled' in x else 0
            ).astype(int)

    # Normalize HasSideJob
    if 'HasSideJob' in df.columns or 'hasjob' in df.columns:
        try:
            if 'hasjob' in df.columns and 'HasSideJob' not in df.columns:
                df = df.rename(columns={'hasjob': 'HasSideJob'})

            df['HasSideJob'] = df['HasSideJob'].map(
                lambda v: 1 if str(v).strip().lower() in ('1', 'true', 'yes', 'y', 't')
                else (0 if str(v).strip().lower() in ('0', 'false', 'no', 'n', 'f') else v)
            )
            df['HasSideJob'] = pd.to_numeric(df['HasSideJob'], errors='coerce').fillna(0).astype(int)
        except Exception:
            pass

    # Normalize MonthlyFamilyIncome
    if 'MonthlyFamilyIncome' in df.columns:
        if for_prediction:
            pass
        else:
            if not pd.api.types.is_numeric_dtype(df['MonthlyFamilyIncome']):
                df['MonthlyFamilyIncome'] = df['MonthlyFamilyIncome'].astype('category').cat.codes
            df['MonthlyFamilyIncome'] = (df['MonthlyFamilyIncome'] - df['MonthlyFamilyIncome'].min()) / (df['MonthlyFamilyIncome'].max() - df['MonthlyFamilyIncome'].min())

    return df
