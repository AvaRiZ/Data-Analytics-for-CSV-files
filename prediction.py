"""Prediction utilities for student enrollment"""
from typing import Dict, Any, Optional
import traceback

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight

from preprocessing import find_column, normalize_columns


def run_predictions(df: pd.DataFrame) -> Optional[Dict[str, Any]]:
    """Run student enrollment prediction and return a results dict or None on error."""
    try:
        if df is None or df.empty:
            return None

        original_size = len(df)
        data = df.copy()

        # normalize columns (avoid Status transform for prediction stage here)
        data = normalize_columns(data, for_prediction=True)

        # Handle Status numeric normalization if present
        if 'Status' in data.columns:
            if pd.api.types.is_numeric_dtype(data['Status']):
                data['Status'] = data['Status'].fillna(0).astype(int).clip(0, 1)

        # HasSideJob
        if 'HasSideJob' in data.columns:
            data['HasSideJob'] = data['HasSideJob'].map(
                lambda v: 1 if str(v).strip().lower() in ('1', 'true', 'yes', 'y', 't')
                else (0 if str(v).strip().lower() in ('0', 'false', 'no', 'n', 'f') else v)
            )
            data['HasSideJob'] = pd.to_numeric(data['HasSideJob'], errors='coerce')

        # MonthlyFamilyIncome
        if 'MonthlyFamilyIncome' in data.columns:
            if not pd.api.types.is_numeric_dtype(data['MonthlyFamilyIncome']):
                data['MonthlyFamilyIncome'] = data['MonthlyFamilyIncome'].astype('category').cat.codes
            data['MonthlyFamilyIncome'] = (data['MonthlyFamilyIncome'] - data['MonthlyFamilyIncome'].min()) / (data['MonthlyFamilyIncome'].max() - data['MonthlyFamilyIncome'].min())

        # Required columns
        required = ['HasSideJob', 'MonthlyFamilyIncome', 'Status']
        missing = [c for c in required if c not in data.columns]
        if missing:
            return None

        # Remove rows with missing required values
        data_clean = data[required].dropna()
        if data_clean.empty:
            return None

        X = data_clean[['HasSideJob', 'MonthlyFamilyIncome']].copy()
        y = data_clean['Status'].copy()

        enrolled_count = y.sum()
        dropped_count = len(y) - enrolled_count
        if enrolled_count == 0 or dropped_count == 0:
            return None

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        try:
            class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
            class_weight_dict = {0: class_weights[0], 1: class_weights[1]}
        except Exception:
            class_weight_dict = 'balanced'

        model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight=class_weight_dict, max_depth=5)
        model.fit(X_train_scaled, y_train)

        cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring='accuracy')

        y_pred = model.predict(X_test_scaled)
        X_full_scaled = scaler.transform(X)
        y_pred_full = model.predict(X_full_scaled)

        accuracy = accuracy_score(y_test, y_pred)
        full_accuracy = accuracy_score(y, y_pred_full)

        cm = confusion_matrix(y, y_pred_full)

        results = {
            'original_size': original_size,
            'clean_size': len(data_clean),
            'rows_removed': original_size - len(data_clean),
            'X_train_size': len(X_train),
            'X_test_size': len(X_test),
            'enrolled_count': enrolled_count,
            'dropped_count': dropped_count,
            'enrolled_percentage': enrolled_count / len(y) * 100,
            'dropped_percentage': dropped_count / len(y) * 100,
            'y_train_enrolled': y_train.sum(),
            'y_train_dropped': len(y_train) - y_train.sum(),
            'accuracy': accuracy,
            'cv_scores': cv_scores,
            'full_accuracy': full_accuracy,
            'feature_importance': model.feature_importances_,
            'y_pred_full': y_pred_full,
            'confusion_matrix': cm,
            'model': model,
            'X': X,
            'y': y,
            'y_pred': y_pred,
            'X_train': X_train,
            'X_test': X_test,
            'y_train': y_train,
            'y_test': y_test,
            'data_clean': data_clean
        }

        return results
    except Exception:
        return None
