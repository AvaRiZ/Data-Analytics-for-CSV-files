import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import io

# Page configuration
st.set_page_config(
    page_title="Data Analysis Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #6272A4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #282A36;
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    .stButton>button {
        background-color: #6272A4;
        color: white;
        border: none;
        padding: 0.5rem 2rem;
        border-radius: 5px;
    }
    .stButton>button:hover {
        background-color: #4ECDC4;
    }
</style>
""", unsafe_allow_html=True)

# Title
st.markdown('<h1 class="main-header">📊 Data Analysis Dashboard</h1>', unsafe_allow_html=True)

# Initialize session state
if 'df' not in st.session_state:
    st.session_state.df = None
if 'processed_df' not in st.session_state:
    st.session_state.processed_df = None

# Sidebar navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Data Upload", "Data Exploration", "Visualization", 
                                  "Clustering", "Predictions", "Statistics"])

# File upload function
def upload_file():
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.session_state.df = df
        st.session_state.processed_df = df.copy()
        st.success(f"File uploaded successfully! Shape: {df.shape}")
        return df
    return None

# Data normalization function (same as your original)
def normalize_columns(df, for_prediction=False):
    """Normalize common column name variants to standard names."""
    # ... (copy the exact same normalize_columns function from your code)
    return df

# Page 1: Data Upload
if page == "Data Upload":
    st.header("📂 Upload Data")
    
    df = upload_file()
    
    if st.session_state.df is not None:
        st.subheader("Data Preview")
        st.dataframe(st.session_state.df.head())
        
        st.subheader("Data Information")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Rows", st.session_state.df.shape[0])
        with col2:
            st.metric("Columns", st.session_state.df.shape[1])
        with col3:
            st.metric("Missing Values", st.session_state.df.isnull().sum().sum())
        
        st.subheader("Column Details")
        st.write(st.session_state.df.dtypes)

# Page 2: Data Exploration
elif page == "Data Exploration" and st.session_state.df is not None:
    st.header("🔍 Data Exploration")
    
    df = st.session_state.df
    
    # Search functionality
    search_query = st.text_input("Search in data")
    if search_query:
        mask = df.apply(lambda row: row.astype(str).str.contains(search_query, case=False).any(), axis=1)
        filtered_df = df[mask]
        st.write(f"Found {len(filtered_df)} matching records")
        st.dataframe(filtered_df)
    else:
        st.dataframe(df)
    
    # Quick filters
    st.subheader("Quick Filters")
    if not df.empty:
        col1, col2 = st.columns(2)
        with col1:
            column = st.selectbox("Select column", df.columns)
        with col2:
            n = st.number_input("Number of values", min_value=1, max_value=100, value=10)
        
        if st.button("Show Top Values"):
            st.write(df.nlargest(n, column))
        
        if st.button("Show Bottom Values"):
            st.write(df.nsmallest(n, column))
    
    # Data cleaning
    st.subheader("Data Cleaning")
    if st.button("Remove Missing Values"):
        df_clean = df.dropna()
        st.session_state.df = df_clean
        st.success(f"Removed {len(df) - len(df_clean)} rows with missing values")

# Page 3: Visualization
elif page == "Visualization" and st.session_state.df is not None:
    st.header("📈 Data Visualization")
    
    df = st.session_state.df
    
    col1, col2, col3 = st.columns(3)
    with col1:
        x_column = st.selectbox("X-axis column", df.columns)
    with col2:
        y_column = st.selectbox("Y-axis column", df.columns)
    with col3:
        chart_type = st.selectbox("Chart type", 
                                 ["Scatter", "Line", "Bar", "Histogram", "Box", "Heatmap"])
    
    # Plot
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
    
    if fig:
        st.plotly_chart(fig, use_container_width=True)
        
        # Download button for plot
        buf = io.BytesIO()
        fig.write_image(buf, format="png")
        st.download_button(
            label="Download Plot",
            data=buf.getvalue(),
            file_name="plot.png",
            mime="image/png"
        )

# Page 4: Clustering
elif page == "Clustering" and st.session_state.df is not None:
    st.header("🎯 Clustering Analysis")
    
    df = st.session_state.df
    
    clustering_type = st.radio("Select clustering type", 
                              ["Student Clustering", "Faculty Clustering"])
    
    if clustering_type == "Student Clustering":
        # Student clustering implementation
        # ... (adapt your cluster_students function)
        pass
    
    elif clustering_type == "Faculty Clustering":
        # Faculty clustering implementation
        # ... (adapt your cluster_faculty function)
        pass

# Page 5: Predictions
elif page == "Predictions" and st.session_state.df is not None:
    st.header("🔮 Predictions")
    
    # Your prediction logic here
    # ... (adapt your Predictions function)
    st.info("Prediction functionality will be implemented here")

# Page 6: Statistics
elif page == "Statistics" and st.session_state.df is not None:
    st.header("📊 Statistics")
    
    df = st.session_state.df
    
    # Descriptive statistics
    st.subheader("Descriptive Statistics")
    st.write(df.describe())
    
    # Aggregations
    st.subheader("Custom Aggregations")
    
    col1, col2 = st.columns(2)
    with col1:
        agg_column = st.selectbox("Select column for aggregation", 
                                 df.select_dtypes(include=[np.number]).columns)
    with col2:
        agg_func = st.selectbox("Aggregation function",
                               ["Mean", "Sum", "Count", "Min", "Max", "Std"])
    
    if agg_column and agg_func:
        if agg_func == "Mean":
            result = df[agg_column].mean()
        elif agg_func == "Sum":
            result = df[agg_column].sum()
        elif agg_func == "Count":
            result = df[agg_column].count()
        elif agg_func == "Min":
            result = df[agg_column].min()
        elif agg_func == "Max":
            result = df[agg_column].max()
        elif agg_func == "Std":
            result = df[agg_column].std()
        
        st.metric(f"{agg_func} of {agg_column}", f"{result:.2f}")

# No data message
elif st.session_state.df is None:
    st.warning("Please upload a CSV file from the Data Upload page to continue.")

# Footer
st.sidebar.markdown("---")
st.sidebar.info("""
### Instructions:
1. Upload your CSV file
2. Explore data and clean if needed
3. Create visualizations
4. Run clustering or predictions
""")

# Run with: streamlit run app.py
