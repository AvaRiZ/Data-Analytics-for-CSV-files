"""Clustering routines for students and faculty.

Extracted from the Streamlit app; functions return dicts with results so the UI can display them.
"""
from typing import Dict, Any, Optional
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

from preprocessing import find_column


def cluster_students(df: pd.DataFrame) -> Optional[Dict[str, Any]]:
    """Cluster students based on Age, GWA, Status and Year-like columns.

    Returns a results dict or None on error/missing columns.
    """
    try:
        if df is None or df.empty:
            return None

        data = df.copy()

        # Find columns
        age_col = find_column(data, ['Age', 'age'])
        gwa_col = find_column(data, ['GWA', 'gwa', 'Grade', 'grade', 'Score', 'score'])
        status_col = find_column(data, ['Status', 'status', 'StudentStatus'])
        year_col = find_column(data, ['Year', 'year', 'YearLevel', 'yearlevel', 'Level', 'level'])

        required_cols = [age_col, gwa_col, status_col, year_col]
        missing_cols = [c for c in required_cols if c is None]
        if missing_cols:
            return None

        cluster_data = data[[age_col, gwa_col, status_col, year_col]].copy()

        # Convert categorical to numeric
        if status_col and not pd.api.types.is_numeric_dtype(cluster_data[status_col]):
            cluster_data[status_col] = cluster_data[status_col].astype('category').cat.codes

        if year_col and not pd.api.types.is_numeric_dtype(cluster_data[year_col]):
            cluster_data[year_col] = cluster_data[year_col].astype('category').cat.codes

        # Handle missing values
        if cluster_data.isnull().any().any():
            cluster_data = cluster_data.fillna(cluster_data.mean())

        # Scale data
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(cluster_data)

        # Elbow method for optimal k
        inertias = []
        k_range = range(2, 8)

        for k in k_range:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            kmeans.fit(scaled_data)
            inertias.append(kmeans.inertia_)

        diff = np.diff(inertias)
        diff_ratio = diff[1:] / diff[:-1]
        optimal_k = 3 if len(diff_ratio) == 0 else np.argmin(diff_ratio) + 3
        optimal_k = min(max(optimal_k, 2), 7)

        # Perform clustering
        kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
        clusters = kmeans.fit_predict(scaled_data)

        data['Student_Cluster'] = clusters

        results = {
            'data': data,
            'clusters': clusters,
            'optimal_k': optimal_k,
            'age_col': age_col,
            'gwa_col': gwa_col,
            'status_col': status_col,
            'year_col': year_col,
            'scaled_data': scaled_data,
            'kmeans': kmeans,
            'inertias': inertias,
            'k_range': list(k_range)
        }

        return results
    except Exception:
        return None


def cluster_faculty(df: pd.DataFrame) -> Optional[Dict[str, Any]]:
    try:
        if df is None or df.empty:
            return None

        data = df.copy()

        exp_col = find_column(data, ['Experience', 'experience', 'Exp', 'exp', 'Years', 'years'])
        load_col = find_column(data, ['TeachingLoad', 'teachingload', 'Load', 'load', 'Teaching', 'teaching'])
        age_col = find_column(data, ['Age', 'age'])

        required_cols = [exp_col, load_col, age_col]
        if any(c is None for c in required_cols):
            return None

        cluster_data = data[[exp_col, load_col, age_col]].copy()

        if cluster_data.isnull().any().any():
            cluster_data = cluster_data.fillna(cluster_data.mean())

        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(cluster_data)

        optimal_k = 3
        kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
        clusters = kmeans.fit_predict(scaled_data)

        cluster_names = {0: 'Low Load', 1: 'Medium Load', 2: 'High Load'}
        data['Faculty_Cluster'] = [cluster_names.get(cluster, f'Cluster {cluster}') for cluster in clusters]

        results = {
            'data': data,
            'clusters': clusters,
            'optimal_k': optimal_k,
            'exp_col': exp_col,
            'load_col': load_col,
            'age_col': age_col,
            'cluster_names': cluster_names,
            'scaled_data': scaled_data,
            'kmeans': kmeans
        }

        return results
    except Exception:
        return None
