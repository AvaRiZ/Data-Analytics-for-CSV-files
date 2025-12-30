import streamlit as st
import sys
import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import io


# Try to import the ner module
try:
    import ner
    NER_AVAILABLE = True
except ImportError:
    NER_AVAILABLE = False
    st.warning("NER module not found. Recommendation feature will be limited.")

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
    .recommendation-card {
        background-color: #1E1E2F;
        padding: 1rem;
        border-radius: 8px;
        margin: 0.5rem 0;
        border-left: 4px solid #6272A4;
    }
    .highlight {
        background-color: #FFD166;
        padding: 0.2rem 0.5rem;
        border-radius: 4px;
        font-weight: bold;
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
if 'clustering_results' not in st.session_state:
    st.session_state.clustering_results = None
if 'prediction_results' not in st.session_state:
    st.session_state.prediction_results = None
if 'recommendation_results' not in st.session_state:
    st.session_state.recommendation_results = None

# Data normalization function (same as your original)
def normalize_columns(df, for_prediction=False):
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

def find_column(df, variants):
    """Helper function to find columns by variants"""
    for variant in variants:
        for col in df.columns:
            if variant.lower() in col.lower():
                return col
    return None

# Clustering functions adapted for Streamlit
def cluster_students_streamlit(df):
    """Student clustering for Streamlit"""
    try:
        if df is None or df.empty:
            st.error("No data available for clustering.")
            return None
        
        data = df.copy()
        
        # Find columns
        age_col = find_column(data, ['Age', 'age'])
        gwa_col = find_column(data, ['GWA', 'gwa', 'Grade', 'grade', 'Score', 'score'])
        status_col = find_column(data, ['Status', 'status', 'StudentStatus'])
        year_col = find_column(data, ['Year', 'year', 'YearLevel', 'yearlevel', 'Level', 'level'])

        required_cols = [age_col, gwa_col, status_col, year_col]
        missing_cols = [f"'{variants[0]}'" for col, variants in 
                       zip(required_cols, [['Age'], ['GWA'], ['Status'], ['Year']]) 
                       if col is None]
        
        if missing_cols:
            st.error(f"Missing required columns: {', '.join(missing_cols)}")
            return None

        # Prepare data
        cluster_data = data[[age_col, gwa_col, status_col, year_col]].copy()
        
        # Convert categorical to numeric
        if status_col and not pd.api.types.is_numeric_dtype(cluster_data[status_col]):
            cluster_data[status_col] = cluster_data[status_col].astype('category').cat.codes
        
        if year_col and not pd.api.types.is_numeric_dtype(cluster_data[year_col]):
            cluster_data[year_col] = cluster_data[year_col].astype('category').cat.codes

        # Handle missing values
        if cluster_data.isnull().any().any():
            cluster_data = cluster_data.fillna(cluster_data.mean())
            st.info("Missing values filled with column means")

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

        # Find optimal k (using simple elbow detection)
        diff = np.diff(inertias)
        diff_ratio = diff[1:] / diff[:-1]
        optimal_k = 3 if len(diff_ratio) == 0 else np.argmin(diff_ratio) + 3
        
        # Ensure optimal_k is within range
        optimal_k = min(max(optimal_k, 2), 7)

        # Perform clustering
        kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
        clusters = kmeans.fit_predict(scaled_data)
        
        # Add cluster labels
        data['Student_Cluster'] = clusters
        
        # Create results dictionary
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
        
    except Exception as e:
        st.error(f"An error occurred during student clustering: {e}")
        return None

def cluster_faculty_streamlit(df):
    """Faculty clustering for Streamlit"""
    try:
        if df is None or df.empty:
            st.error("No data available for clustering.")
            return None
        
        data = df.copy()
        
        # Find columns
        exp_col = find_column(data, ['Experience', 'experience', 'Exp', 'exp', 'Years', 'years'])
        load_col = find_column(data, ['TeachingLoad', 'teachingload', 'Load', 'load', 'Teaching', 'teaching'])
        age_col = find_column(data, ['Age', 'age'])

        required_cols = [exp_col, load_col, age_col]
        missing_cols = [f"'{variants[0]}'" for col, variants in 
                       zip(required_cols, [['Experience'], ['TeachingLoad'], ['Age']]) 
                       if col is None]
        
        if missing_cols:
            st.error(f"Missing required columns: {', '.join(missing_cols)}")
            return None

        # Prepare data
        cluster_data = data[[exp_col, load_col, age_col]].copy()

        # Handle missing values
        if cluster_data.isnull().any().any():
            cluster_data = cluster_data.fillna(cluster_data.mean())
            st.info("Missing values filled with column means")

        # Scale data
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(cluster_data)

        # Use 3 clusters for faculty
        optimal_k = 3
        
        # Perform clustering
        kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
        clusters = kmeans.fit_predict(scaled_data)
        
        # Add cluster labels with meaningful names
        cluster_names = {0: 'Low Load', 1: 'Medium Load', 2: 'High Load'}
        data['Faculty_Cluster'] = [cluster_names.get(cluster, f'Cluster {cluster}') for cluster in clusters]
        
        # Create results dictionary
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
        
    except Exception as e:
        st.error(f"An error occurred during faculty clustering: {e}")
        return None

# Prediction function adapted for Streamlit
def run_predictions_streamlit(df):
    """Run student enrollment prediction for Streamlit"""
    try:
        if df is None or df.empty:
            st.error("No data available for predictions.")
            return None
        
        # Show data size info
        original_size = len(df)
        
        # Work on a copy
        data = df.copy()
        
        # Manual normalization
        if 'Status' in data.columns:
            if pd.api.types.is_numeric_dtype(data['Status']):
                data['Status'] = data['Status'].fillna(0).astype(int).clip(0, 1)
            else:
                data['Status'] = data['Status'].astype(str).str.lower().str.strip().apply(
                    lambda x: 1 if 'enrolled' in x else 0
                ).astype(int)

        # Handle HasSideJob column
        if 'HasSideJob' in data.columns:
            data['HasSideJob'] = data['HasSideJob'].map(
                lambda v: 1 if str(v).strip().lower() in ('1', 'true', 'yes', 'y', 't')
                else (0 if str(v).strip().lower() in ('0', 'false', 'no', 'n', 'f') else v)
            )
            data['HasSideJob'] = pd.to_numeric(data['HasSideJob'], errors='coerce')

        # Handle MonthlyFamilyIncome
        if 'MonthlyFamilyIncome' in data.columns:
            if not pd.api.types.is_numeric_dtype(data['MonthlyFamilyIncome']):
                data['MonthlyFamilyIncome'] = data['MonthlyFamilyIncome'].astype('category').cat.codes
            # Scale to 0-1 range
            data['MonthlyFamilyIncome'] = (data['MonthlyFamilyIncome'] - data['MonthlyFamilyIncome'].min()) / (data['MonthlyFamilyIncome'].max() - data['MonthlyFamilyIncome'].min())

        # Check required columns
        required = ['HasSideJob', 'MonthlyFamilyIncome', 'Status']
        missing = [c for c in required if c not in data.columns]
        
        if missing:
            st.error(f"Missing required columns for prediction: {', '.join(missing)}")
            st.info("Looking for alternative column names...")
            
            # Try to find alternative columns
            for col in missing:
                if col == 'HasSideJob':
                    alt_col = find_column(data, ['hasjob', 'employed', 'job', 'sidejob'])
                    if alt_col:
                        data['HasSideJob'] = data[alt_col]
                        st.info(f"Using '{alt_col}' as HasSideJob")
                elif col == 'MonthlyFamilyIncome':
                    alt_col = find_column(data, ['income', 'pay', 'wage', 'salary', 'compensation'])
                    if alt_col:
                        data['MonthlyFamilyIncome'] = data[alt_col]
                        st.info(f"Using '{alt_col}' as MonthlyFamilyIncome")
                elif col == 'Status':
                    alt_col = find_column(data, ['studentstatus', 'enrollment', 'enrolled'])
                    if alt_col:
                        data['Status'] = data[alt_col]
                        st.info(f"Using '{alt_col}' as Status")
        
        # Re-check after alternatives
        missing = [c for c in required if c not in data.columns]
        if missing:
            st.error(f"Still missing required columns: {', '.join(missing)}")
            return None

        # Check for missing values
        missing_info = data[required].isnull().sum()
        if missing_info.sum() > 0:
            st.warning(f"Missing values found:\n{missing_info}\n\nThese rows will be removed.")
        
        # Remove rows with missing values
        data_clean = data[required].dropna()
        rows_removed = len(data) - len(data_clean)
        
        if rows_removed > 0:
            st.info(f"Removed {rows_removed} rows with missing values")
        
        if len(data_clean) == 0:
            st.error("No data remaining after cleaning!")
            return None

        # Prepare features and target
        X = data_clean[['HasSideJob', 'MonthlyFamilyIncome']].copy()
        y = data_clean['Status'].copy()

        # Check class distribution
        enrolled_count = y.sum()
        dropped_count = len(y) - enrolled_count
        
        if enrolled_count == 0 or dropped_count == 0:
            st.error("Need both enrolled and dropped students for prediction.")
            return None

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Compute class weights
        try:
            class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
            class_weight_dict = {0: class_weights[0], 1: class_weights[1]}
        except Exception:
            class_weight_dict = 'balanced'

        # Train model
        model = RandomForestClassifier(
            n_estimators=100,
            random_state=42,
            class_weight=class_weight_dict,
            max_depth=5
        )
        
        model.fit(X_train_scaled, y_train)

        # Cross-validation
        cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring='accuracy')
        
        # Predictions
        y_pred = model.predict(X_test_scaled)
        X_full_scaled = scaler.transform(X)
        y_pred_full = model.predict(X_full_scaled)

        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        full_accuracy = accuracy_score(y, y_pred_full)

        # Confusion matrix
        cm = confusion_matrix(y, y_pred_full)
        
        # Create results dictionary
        results = {
            'original_size': original_size,
            'clean_size': len(data_clean),
            'rows_removed': rows_removed,
            'X_train_size': len(X_train),
            'X_test_size': len(X_test),
            'enrolled_count': enrolled_count,
            'dropped_count': dropped_count,
            'enrolled_percentage': enrolled_count/len(y)*100,
            'dropped_percentage': dropped_count/len(y)*100,
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
        
    except Exception as e:
        st.error(f"An error occurred during prediction: {e}")
        import traceback
        st.error(traceback.format_exc())
        return None

# Recommendation function for Streamlit
def run_recommendations_streamlit(df, subject, top_n=10):
    """Run faculty recommendation for Streamlit"""
    try:
        if df is None or df.empty:
            st.error("No data available for recommendations.")
            return None
        
        if not NER_AVAILABLE:
            st.error("NER module not available. Please ensure ner.py is in the same directory.")
            return None
        
        # Run recommendation using ner module
        recommendations = ner.rank_faculty(
            df=df,
            subject=subject,
            top_n=top_n
        )
        
        return recommendations
        
    except Exception as e:
        st.error(f"An error occurred during recommendation: {e}")
        import traceback
        st.error(traceback.format_exc())
        return None

# Alternative simple recommendation function (if NER module is not available)
def simple_recommendation(df, subject, top_n=10):
    """Simple text-based recommendation as fallback"""
    try:
        if df is None or df.empty:
            return None
        
        # Find text column
        text_column = None
        cols_lower = [c.lower() for c in df.columns]
        
        # Look for text columns
        for cand in ("bio", "description", "profile", "details", "about", "summary", 
                    "expertise", "field", "specialization", "skills"):
            for i, c in enumerate(cols_lower):
                if cand in c:
                    text_column = df.columns[i]
                    break
            if text_column:
                break
        
        # Find name column
        name_column = None
        for cand in ("name", "faculty", "professor", "instructor", "staff"):
            for i, c in enumerate(cols_lower):
                if cand in c:
                    name_column = df.columns[i]
                    break
            if name_column:
                break
        
        if not text_column:
            # Fallback to first column
            text_column = df.columns[0]
        
        if not name_column:
            # Fallback to second column or first if only one column
            name_column = df.columns[1] if len(df.columns) > 1 else df.columns[0]
        
        subject_lower = subject.lower()
        results = []
        
        for idx, row in df.iterrows():
            text = str(row.get(text_column, ""))
            name = str(row.get(name_column, f"Faculty {idx}"))
            text_lower = text.lower()
            
            # Simple scoring based on keyword matches
            score = text_lower.count(subject_lower) * 3
            
            # Count individual word matches
            subject_words = subject_lower.split()
            for word in subject_words:
                if len(word) > 3:  # Ignore short words
                    score += text_lower.count(word)
            
            # Add extra information if available
            age = row.get('Age', 'N/A')
            gender = row.get('Gender', 'N/A')
            experience = row.get('YearsExperience', row.get('Experience', 'N/A'))
            field = row.get('FieldOfExpertise', text[:100] + "...")
            
            results.append({
                "name": name,
                "score": float(score),
                "text": text,
                "index": idx,
                "age": age,
                "gender": gender,
                "years_experience": experience,
                "field_of_expertise": field
            })
        
        # Sort by score
        results = sorted(results, key=lambda x: x["score"], reverse=True)
        return results[:top_n]
        
    except Exception as e:
        st.error(f"Error in simple recommendation: {e}")
        return None

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

# Sidebar navigation - ADDED RECOMMENDATIONS
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Data Upload", "Data Exploration", "Visualization", 
                                  "Clustering", "Predictions", "Recommendations", "Statistics"])

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
        
        col3, col4 = st.columns(2)
        with col3:
            if st.button("Show Top Values"):
                st.write(df.nlargest(n, column))
        with col4:
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
        y_column = st.selectbox("Y-axis column", df.columns, index=1 if len(df.columns) > 1 else 0)
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
    
    st.markdown("---")
    
    if clustering_type == "Student Clustering":
        st.subheader("Student Clustering")
        
        if st.button("Run Student Clustering", type="primary"):
            with st.spinner("Clustering students..."):
                results = cluster_students_streamlit(df)
                
                if results:
                    st.session_state.clustering_results = results
                    
                    # Display results
                    st.success(f"Student clustering completed! {results['optimal_k']} clusters identified.")
                    
                    # Create two columns for plots
                    col1, col2 = st.columns(2)
                    
                    # Plot 1: Scatter plot
                    with col1:
                        fig1, ax1 = plt.subplots(figsize=(8, 6))
                        scatter = ax1.scatter(
                            results['data'][results['age_col']], 
                            results['data'][results['gwa_col']], 
                            c=results['clusters'], 
                            cmap='viridis', 
                            alpha=0.7
                        )
                        ax1.set_xlabel(results['age_col'])
                        ax1.set_ylabel(results['gwa_col'])
                        ax1.set_title(f"Student Clusters: {results['age_col']} vs {results['gwa_col']}")
                        ax1.grid(True, alpha=0.3)
                        plt.colorbar(scatter, ax=ax1)
                        st.pyplot(fig1)
                    
                    # Plot 2: Cluster distribution
                    with col2:
                        fig2, ax2 = plt.subplots(figsize=(8, 6))
                        cluster_counts = results['data']['Student_Cluster'].value_counts().sort_index()
                        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFD166', '#06D6A0'][:len(cluster_counts)]
                        bars = ax2.bar(cluster_counts.index, cluster_counts.values, color=colors)
                        ax2.set_xlabel('Cluster')
                        ax2.set_ylabel('Number of Students')
                        ax2.set_title('Student Distribution Across Clusters')
                        
                        for bar in bars:
                            height = bar.get_height()
                            ax2.text(
                                bar.get_x() + bar.get_width()/2., 
                                height,
                                f'{int(height)}', 
                                ha='center', 
                                va='bottom'
                            )
                        st.pyplot(fig2)
                    
                    # Elbow plot
                    st.subheader("Elbow Method for Optimal K")
                    fig3, ax3 = plt.subplots(figsize=(10, 4))
                    ax3.plot(results['k_range'], results['inertias'], marker='o')
                    ax3.set_xlabel('Number of Clusters (k)')
                    ax3.set_ylabel('Inertia')
                    ax3.set_title('Elbow Method for Optimal K')
                    ax3.grid(True, alpha=0.3)
                    ax3.axvline(x=results['optimal_k'], color='red', linestyle='--', alpha=0.5, label=f'Optimal k={results["optimal_k"]}')
                    ax3.legend()
                    st.pyplot(fig3)
                    
                    # Cluster summary
                    st.subheader("Cluster Profiles")
                    
                    summary_data = results['data'].groupby('Student_Cluster').agg({
                        results['age_col']: 'mean',
                        results['gwa_col']: 'mean',
                        results['status_col']: lambda x: x.mode()[0] if len(x.mode()) > 0 else 'Unknown',
                        results['year_col']: 'mean'
                    }).round(2)
                    
                    st.dataframe(summary_data)
                    
                    # Detailed cluster information
                    for cluster_id in range(results['optimal_k']):
                        with st.expander(f"Cluster {cluster_id} Details"):
                            cluster_data = results['data'][results['data']['Student_Cluster'] == cluster_id]
                            st.write(f"**Number of students:** {len(cluster_data)}")
                            st.write(f"**Average {results['age_col']}:** {cluster_data[results['age_col']].mean():.1f}")
                            st.write(f"**Average {results['gwa_col']}:** {cluster_data[results['gwa_col']].mean():.2f}")
                            st.write(f"**Average {results['year_col']}:** {cluster_data[results['year_col']].mean():.1f}")
                            
                            if len(cluster_data) > 0:
                                st.dataframe(cluster_data.head())
                    
                    # Download clustered data
                    csv = results['data'].to_csv(index=False)
                    st.download_button(
                        label="Download Clustered Data",
                        data=csv,
                        file_name="student_clusters.csv",
                        mime="text/csv"
                    )
    
    elif clustering_type == "Faculty Clustering":
        st.subheader("Faculty Clustering")
        
        if st.button("Run Faculty Clustering", type="primary"):
            with st.spinner("Clustering faculty..."):
                results = cluster_faculty_streamlit(df)
                
                if results:
                    st.session_state.clustering_results = results
                    
                    # Display results
                    st.success(f"Faculty clustering completed! {results['optimal_k']} faculty groups identified.")
                    
                    # Create two columns for plots
                    col1, col2 = st.columns(2)
                    
                    # Plot 1: Scatter plot
                    with col1:
                        fig1, ax1 = plt.subplots(figsize=(8, 6))
                        scatter = ax1.scatter(
                            results['data'][results['exp_col']], 
                            results['data'][results['load_col']], 
                            c=results['clusters'], 
                            cmap='plasma', 
                            alpha=0.7
                        )
                        ax1.set_xlabel(results['exp_col'])
                        ax1.set_ylabel(results['load_col'])
                        ax1.set_title(f"Faculty Clusters: {results['exp_col']} vs {results['load_col']}")
                        ax1.grid(True, alpha=0.3)
                        plt.colorbar(scatter, ax=ax1)
                        st.pyplot(fig1)
                    
                    # Plot 2: Cluster distribution
                    with col2:
                        fig2, ax2 = plt.subplots(figsize=(8, 6))
                        cluster_counts = results['data']['Faculty_Cluster'].value_counts()
                        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1'][:len(cluster_counts)]
                        bars = ax2.bar(range(len(cluster_counts)), cluster_counts.values, color=colors)
                        ax2.set_xlabel('Faculty Group')
                        ax2.set_ylabel('Number of Faculty')
                        ax2.set_title('Faculty Distribution Across Groups')
                        ax2.set_xticks(range(len(cluster_counts)))
                        ax2.set_xticklabels(cluster_counts.index, rotation=45)
                        
                        for bar in bars:
                            height = bar.get_height()
                            ax2.text(
                                bar.get_x() + bar.get_width()/2., 
                                height,
                                f'{int(height)}', 
                                ha='center', 
                                va='bottom'
                            )
                        st.pyplot(fig2)
                    
                    # Cluster summary
                    st.subheader("Faculty Group Profiles")
                    
                    for cluster_name in results['cluster_names'].values():
                        with st.expander(f"{cluster_name} Group Details"):
                            cluster_data = results['data'][results['data']['Faculty_Cluster'] == cluster_name]
                            if len(cluster_data) > 0:
                                st.write(f"**Number of faculty:** {len(cluster_data)}")
                                st.write(f"**Average {results['exp_col']}:** {cluster_data[results['exp_col']].mean():.1f} years")
                                st.write(f"**Average {results['load_col']}:** {cluster_data[results['load_col']].mean():.1f}")
                                st.write(f"**Average {results['age_col']}:** {cluster_data[results['age_col']].mean():.1f} years")
                                
                                if len(cluster_data) > 0:
                                    st.dataframe(cluster_data.head())
                    
                    # Summary table
                    st.subheader("Summary Statistics by Faculty Group")
                    summary_stats = results['data'].groupby('Faculty_Cluster').agg({
                        results['exp_col']: ['mean', 'std', 'min', 'max'],
                        results['load_col']: ['mean', 'std', 'min', 'max'],
                        results['age_col']: ['mean', 'std', 'min', 'max']
                    }).round(2)
                    
                    st.dataframe(summary_stats)
                    
                    # Download clustered data
                    csv = results['data'].to_csv(index=False)
                    st.download_button(
                        label="Download Clustered Data",
                        data=csv,
                        file_name="faculty_clusters.csv",
                        mime="text/csv"
                    )
    
    # Show note if no data
    elif st.session_state.df is None:
        st.info("Please upload data first from the Data Upload page.")

# Page 5: Predictions
elif page == "Predictions" and st.session_state.df is not None:
    st.header("🔮 Student Enrollment Predictions")
    
    st.info("This model predicts student enrollment status based on side job and family income.")
    
    if st.button("Run Enrollment Predictions", type="primary"):
        with st.spinner("Training model and making predictions..."):
            results = run_predictions_streamlit(st.session_state.df)
            
            if results:
                st.session_state.prediction_results = results
                
                # Display results
                st.success("Prediction completed successfully!")
                
                # Data Processing Summary
                st.subheader("📋 Data Processing Summary")
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Original Dataset", f"{results['original_size']} students")
                with col2:
                    st.metric("After Cleaning", f"{results['clean_size']} students")
                with col3:
                    st.metric("Rows Removed", f"{results['rows_removed']}")
                
                # Class Distribution
                st.subheader("🎯 Class Distribution")
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Enrolled Students", 
                             f"{results['enrolled_count']} ({results['enrolled_percentage']:.1f}%)")
                with col2:
                    st.metric("Dropped Students", 
                             f"{results['dropped_count']} ({results['dropped_percentage']:.1f}%)")
                
                # Training/Test Split
                st.subheader("📊 Training/Test Split")
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Training Data", f"{results['X_train_size']} students")
                with col2:
                    st.metric("Test Data", f"{results['X_test_size']} students")
                
                # Model Performance
                st.subheader("📈 Model Performance")
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Test Accuracy", f"{results['accuracy']:.3f}")
                with col2:
                    st.metric("Full Data Accuracy", f"{results['full_accuracy']:.3f}")
                with col3:
                    avg_cv = np.mean(results['cv_scores'])
                    st.metric("Avg CV Score", f"{avg_cv:.3f}")
                
                # Cross-validation scores
                st.write("**Cross-validation Scores:**")
                st.write([f'{s:.3f}' for s in results['cv_scores']])
                
                # Feature Importance
                st.subheader("🔍 Feature Importance")
                importance_df = pd.DataFrame({
                    'Feature': ['HasSideJob', 'MonthlyFamilyIncome'],
                    'Importance': results['feature_importance']
                }).sort_values('Importance', ascending=False)
                
                fig_importance = px.bar(importance_df, x='Feature', y='Importance', 
                                       title='Feature Importance in Prediction Model',
                                       color='Importance')
                st.plotly_chart(fig_importance, use_container_width=True)
                
                # Confusion Matrix
                st.subheader("📊 Confusion Matrix")
                cm = results['confusion_matrix']
                
                fig_cm, ax_cm = plt.subplots(figsize=(8, 6))
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax_cm,
                          xticklabels=['Predicted\nDropped', 'Predicted\nEnrolled'],
                          yticklabels=['Actual\nDropped', 'Actual\nEnrolled'],
                          cbar_kws={'label': 'Number of Students'},
                          annot_kws={'size': 12, 'weight': 'bold'})
                
                ax_cm.set_xlabel('\nPredicted Status', fontsize=12, fontweight='bold')
                ax_cm.set_ylabel('Actual Status\n', fontsize=12, fontweight='bold')
                ax_cm.set_title('Confusion Matrix - Student Enrollment Prediction\n', 
                              fontsize=14, fontweight='bold')
                
                for i in range(3):
                    ax_cm.axhline(i, color='white', linewidth=2)
                    ax_cm.axvline(i, color='white', linewidth=2)
                
                interpretation = (
                    "Interpretation:\n"
                    "• Top-Left: Actually Dropped, Correctly Predicted ✓\n"
                    "• Top-Right: Actually Dropped, Wrongly Predicted as Enrolled ✗\n"
                    "• Bottom-Left: Actually Enrolled, Wrongly Predicted as Dropped ✗\n"
                    "• Bottom-Right: Actually Enrolled, Correctly Predicted ✓"
                )
                
                ax_cm.text(2.5, 0.5, interpretation, transform=ax_cm.transAxes, fontsize=10,
                         verticalalignment='center', bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7))
                
                plt.tight_layout()
                st.pyplot(fig_cm)
                
                # Prediction Results Table
                st.subheader("📋 Prediction Results")
                
                # Create results dataframe
                results_df = results['X'].copy()
                results_df['Actual_Status'] = results['y'].map({0: 'Dropped', 1: 'Enrolled'})
                results_df['Predicted_Status'] = results['y_pred_full']
                results_df['Predicted_Status'] = results_df['Predicted_Status'].map({0: 'Dropped', 1: 'Enrolled'})
                results_df = results_df.reset_index()
                
                # Show first 50 rows
                st.write(f"Showing first 50 of {len(results_df)} predictions:")
                st.dataframe(results_df.head(50))
                
                # Statistics
                st.subheader("📊 Prediction Statistics")
                correct_predictions = (results['y_pred_full'] == results['y']).sum()
                total_predictions = len(results['y_pred_full'])
                accuracy_percentage = correct_predictions / total_predictions * 100
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Total Predictions", total_predictions)
                with col2:
                    st.metric("Correct Predictions", correct_predictions)
                with col3:
                    st.metric("Prediction Accuracy", f"{accuracy_percentage:.1f}%")
                
                # Download predictions
                st.subheader("💾 Download Results")
                csv = results_df.to_csv(index=False)
                st.download_button(
                    label="Download Prediction Results",
                    data=csv,
                    file_name="enrollment_predictions.csv",
                    mime="text/csv"
                )
                
                # Detailed classification report
                with st.expander("View Detailed Classification Report"):
                    report = classification_report(results['y'], results['y_pred_full'], 
                                                 target_names=['Dropped', 'Enrolled'],
                                                 output_dict=True)
                    report_df = pd.DataFrame(report).transpose()
                    st.dataframe(report_df)
    
    elif st.session_state.prediction_results:
        # If results already exist, show them
        results = st.session_state.prediction_results
        
        st.info("Previous prediction results loaded. Click the button above to run new predictions.")
        
        # Quick summary
        st.subheader("📊 Previous Results Summary")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Accuracy", f"{results['accuracy']:.3f}")
        with col2:
            st.metric("Total Students", results['clean_size'])

# Page 6: Recommendations (NEW PAGE)
elif page == "Recommendations" and st.session_state.df is not None:
    st.header("👨‍🏫 Faculty Recommendations")
    
    st.info("Find the best faculty members for a specific subject or topic.")
    
    # Check if NER module is available
    if not NER_AVAILABLE:
        st.warning("""
        **Note:** The NER module is not available. Using simple text-based matching.
        For better recommendations, ensure `ner.py` is in the same directory and 
        spaCy is installed:
        ```
        pip install spacy
        python -m spacy download en_core_web_sm
        ```
        """)
    
    # Input for subject
    col1, col2 = st.columns([3, 1])
    with col1:
        subject = st.text_input(
            "Enter subject/topic to find best faculty for:",
            placeholder="e.g., Machine Learning, Data Science, Calculus, Physics",
            help="Enter a subject, topic, or area of expertise"
        )
    with col2:
        top_n = st.number_input(
            "Number of results:",
            min_value=1,
            max_value=20,
            value=10,
            help="Number of top recommendations to show"
        )
    
    if subject:
        if st.button("Find Recommendations", type="primary"):
            with st.spinner("Analyzing faculty profiles..."):
                # Use NER module if available, otherwise use simple matching
                if NER_AVAILABLE:
                    recommendations = run_recommendations_streamlit(
                        st.session_state.df, 
                        subject, 
                        top_n=top_n
                    )
                else:
                    recommendations = simple_recommendation(
                        st.session_state.df,
                        subject,
                        top_n=top_n
                    )
                
                if recommendations:
                    st.session_state.recommendation_results = recommendations
                    
                    # Display results
                    st.success(f"Found {len(recommendations)} recommendations for '{subject}'")
                    
                    # Top match summary
                    if recommendations:
                        top_match = recommendations[0]
                        st.subheader(f"🏆 Top Match: {top_match['name']}")
                        
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            st.metric("Relevance Score", f"{top_match['score']:.2f}")
                        with col2:
                            st.metric("Age", str(top_match['age']))
                        with col3:
                            st.metric("Gender", str(top_match['gender']))
                        with col4:
                            st.metric("Experience", str(top_match['years_experience']))
                        
                        # Field of expertise with highlighting
                        st.write("**Field of Expertise:**")
                        expertise = str(top_match['field_of_expertise'])
                        # Simple highlighting of subject in expertise
                        subject_lower = subject.lower()
                        expertise_lower = expertise.lower()
                        
                        if subject_lower in expertise_lower:
                            start_idx = expertise_lower.find(subject_lower)
                            highlighted = (
                                expertise[:start_idx] +
                                f"<span class='highlight'>{expertise[start_idx:start_idx+len(subject)]}</span>" +
                                expertise[start_idx+len(subject):]
                            )
                            st.markdown(highlighted, unsafe_allow_html=True)
                        else:
                            st.write(expertise[:200] + "..." if len(expertise) > 200 else expertise)
                    
                    # All recommendations
                    st.subheader("📋 All Recommendations")
                    
                    # Create dataframe for table view
                    rec_df = pd.DataFrame([
                        {
                            'Rank': i+1,
                            'Name': rec['name'],
                            'Score': f"{rec['score']:.2f}",
                            'Age': rec['age'],
                            'Gender': rec['gender'],
                            'Experience': rec['years_experience'],
                            'Expertise': (rec['field_of_expertise'][:100] + '...' 
                                         if len(str(rec['field_of_expertise'])) > 100 
                                         else rec['field_of_expertise'])
                        }
                        for i, rec in enumerate(recommendations)
                    ])
                    
                    st.dataframe(rec_df, use_container_width=True)
                    
                    # Detailed view for each recommendation
                    st.subheader("🔍 Detailed Faculty Profiles")
                    
                    for i, rec in enumerate(recommendations):
                        with st.expander(f"{i+1}. {rec['name']} (Score: {rec['score']:.2f})"):
                            col1, col2 = st.columns(2)
                            with col1:
                                st.write(f"**Age:** {rec['age']}")
                                st.write(f"**Gender:** {rec['gender']}")
                            with col2:
                                st.write(f"**Years of Experience:** {rec['years_experience']}")
                                st.write(f"**Index in Dataset:** {rec['index']}")
                            
                            st.write("**Field of Expertise:**")
                            st.write(rec['field_of_expertise'])
                            
                            # Show full text if available and different from expertise
                            if 'text' in rec and rec['text'] != rec['field_of_expertise']:
                                with st.expander("View Full Profile"):
                                    st.write(rec['text'])
                    
                    # Visualizations
                    st.subheader("📊 Recommendation Analysis")
                    
                    # Score distribution
                    fig_scores, ax_scores = plt.subplots(figsize=(10, 4))
                    scores = [r['score'] for r in recommendations]
                    names = [r['name'][:20] + '...' if len(r['name']) > 20 else r['name'] 
                            for r in recommendations]
                    
                    bars = ax_scores.barh(range(len(scores)), scores, color='#4ECDC4')
                    ax_scores.set_yticks(range(len(scores)))
                    ax_scores.set_yticklabels(names)
                    ax_scores.set_xlabel('Relevance Score')
                    ax_scores.set_title('Faculty Relevance Scores')
                    ax_scores.invert_yaxis()  # Highest score at top
                    
                    # Add score values on bars
                    for bar, score in zip(bars, scores):
                        width = bar.get_width()
                        ax_scores.text(width, bar.get_y() + bar.get_height()/2,
                                     f'{score:.2f}', ha='left', va='center')
                    
                    plt.tight_layout()
                    st.pyplot(fig_scores)
                    
                    # Download recommendations
                    st.subheader("💾 Download Recommendations")
                    
                    # Prepare data for download
                    download_df = pd.DataFrame(recommendations)
                    csv = download_df.to_csv(index=False)
                    
                    st.download_button(
                        label="Download Recommendations as CSV",
                        data=csv,
                        file_name=f"faculty_recommendations_{subject.replace(' ', '_')}.csv",
                        mime="text/csv"
                    )
                    
                    # Also show as JSON for API-like usage
                    with st.expander("View Raw Recommendation Data"):
                        st.json(recommendations)
                
                else:
                    st.error("No recommendations found. Try a different subject or check your data.")
    
    elif st.session_state.recommendation_results:
        # If previous results exist
        st.info("Previous recommendation results available. Enter a new subject or click the button above.")
        
        # Quick summary of previous results
        prev_results = st.session_state.recommendation_results
        if prev_results:
            st.subheader("📊 Previous Results Summary")
            st.write(f"Last search returned {len(prev_results)} recommendations")
            st.write(f"Top match: **{prev_results[0]['name']}** (Score: {prev_results[0]['score']:.2f})")
    
    # Data requirements
    with st.expander("📋 Data Requirements for Recommendations"):
        st.write("""
        For best results, your dataset should include:
        
        **Required columns (or similar names):**
        - `Name` or `Faculty` - Faculty member's name
        - `FieldOfExpertise` or similar - Description of expertise
        
        **Optional but helpful columns:**
        - `Age` - Faculty age
        - `Gender` - Faculty gender
        - `YearsExperience` or `Experience` - Years of teaching experience
        - `Bio` or `Description` - Detailed profile description
        
        **Example CSV structure:**
        ```
        Name,FieldOfExpertise,Age,Gender,YearsExperience,Bio
        Dr. Smith,Machine Learning and AI,45,Male,20,Expert in ML with PhD from...
        Dr. Johnson,Data Science and Statistics,38,Female,12,Specializes in data...
        ```
        """)

# Page 7: Statistics
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
4. Run clustering, predictions, or recommendations
""")

# Additional setup instructions
st.sidebar.markdown("---")
st.sidebar.subheader("🔧 Setup for Recommendations")
st.sidebar.write("""
For the recommendation feature to work optimally:

1. Ensure `ner.py` is in the same directory
2. Install spaCy: `pip install spacy`
3. Download the model: `python -m spacy download en_core_web_sm`
""")

# Run with: streamlit run app.py