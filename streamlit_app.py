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
import file_io
from visualization import create_plot


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

# Use preprocessing helpers
from preprocessing import normalize_columns, find_column

# Use clustering and prediction modules
from clustering import cluster_students, cluster_faculty
from prediction import run_predictions
# Use recommendation helpers
from recommendation import run_recommendations, simple_recommendation

# Use file upload helper
from file_io import upload_file

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
    
    # Plot using helper
    fig = create_plot(df, x_column, y_column, chart_type)
    
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
                results = cluster_students(df)
                
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
                results = cluster_faculty(df)
                
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
            results = run_predictions(st.session_state.df)
            
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
                # Use recommendation helper (handles NER fallback internally)
                recommendations = run_recommendations(st.session_state.df, subject, top_n=top_n)
                
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

# Run with: streamlit run app.py