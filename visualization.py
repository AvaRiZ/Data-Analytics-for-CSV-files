"""Visualization helpers (plotting)"""
import plotly.express as px
import pandas as pd
import numpy as np


def create_plot(df: pd.DataFrame, x_column: str, y_column: str, chart_type: str):
    fig = None
    if chart_type == "Scatter":
        fig = px.scatter(df, x=x_column, y=y_column, title=f"{x_column} vs {y_column}")
    elif chart_type == "Line":
        fig = px.line(df, x=x_column, y=y_column, title=f"{x_column} vs {y_column}")
    elif chart_type == "Bar":
        fig = px.bar(df, x=x_column, y=y_column, title=f"{x_column} vs {y_column}")
    elif chart_type == "Histogram":
        fig = px.histogram(df, x=x_column, title=f"Distribution of {x_column}")
    elif chart_type == "Box":
        fig = px.box(df, x=x_column, y=y_column, title=f"Box plot of {y_column} by {x_column}")
    elif chart_type == "Heatmap":
        numeric_df = df.select_dtypes(include=[np.number])
        if not numeric_df.empty:
            corr = numeric_df.corr()
            fig = px.imshow(corr, title="Correlation Heatmap")
    return fig
