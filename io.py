"""I/O helpers for the Streamlit app."""
from typing import Optional
import pandas as pd
import streamlit as st


def upload_file() -> Optional[pd.DataFrame]:
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.session_state.df = df
        st.session_state.processed_df = df.copy()
        st.success(f"File uploaded successfully! Shape: {df.shape}")
        return df
    return None
