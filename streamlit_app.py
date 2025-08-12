import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import warnings

# Silence warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="seaborn")
warnings.filterwarnings("ignore", category=FutureWarning, module="pandas")
warnings.filterwarnings("ignore", category=FutureWarning, module="sklearn")

st.set_page_config(
    page_title="Churn Analysis Dashboard",
    page_icon="📊",
    layout="wide"
)

st.title("📊 Customer Churn Analysis Dashboard")
st.markdown("Upload your customer data CSV file to perform comprehensive churn analysis")

# File upload
uploaded_file = st.file_uploader(
    "Choose a CSV file",
    type=['csv'],
    help="Upload a CSV file with customer data including a 'churn' column"
)

if uploaded_file is not None:
    try:
        # Load data
        df = pd.read_csv(uploaded_file)
        st.success(f"✅ Data loaded successfully! Shape: {df.shape[0]} rows × {df.shape[1]} columns")
        
        # Display data preview
        st.subheader("📋 Data Preview")
        st.dataframe(df.head(), use_container_width=True)
        
        # Basic info
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Customers", len(df))
        with col2:
            st.metric("Features", len(df.columns))
        with col3:
            if 'churn' in df.columns:
                churn_rate = df['churn'].mean() * 100
                st.metric("Churn Rate", f"{churn_rate:.1f}%")
            else:
                st.metric("Churn Column", "Not Found")
        
        # Missing values analysis
        st.subheader("🔍 Missing Values Analysis")
        missing = df.isnull().sum()
        if missing.sum() > 0:
            fig, ax = plt.subplots(figsize=(10, 6))
            missing.plot(kind='bar', ax=ax)
            plt.title("Missing Values by Column")
            plt.xticks(rotation=45)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()
        else:
            st.success("✅ No missing values found in the dataset")
        
        # Data types and summary
        st.subheader("📊 Data Summary")
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Data Types:**")
            st.write(df.dtypes.value_counts())
            
        with col2:
            st.write("**Numeric Columns Summary:**")
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 0:
                st.write(df[numeric_cols].describe())
            else:
                st.write("No numeric columns found")
        
        # Distribution plots for numeric columns
        if len(numeric_cols) > 0:
            st.subheader("📈 Feature Distributions")
            selected_col = st.selectbox("Select a numeric column to visualize:", numeric_cols)
            
            fig, axes = plt.subplots(1, 3, figsize=(15, 4))
            
            # Histogram
            sns.histplot(df[selected_col].dropna(), ax=axes[0], kde=True)
            axes[0].set_title(f"Histogram of {selected_col}")
            
            # Boxplot
            sns.boxplot(x=df[selected_col], ax=axes[1])
            axes[1].set_title(f"Boxplot of {selected_col}")
            
            # Violin plot
            sns.violinplot(x=df[selected_col], ax=axes[2])
            axes[2].set_title(f"Violin Plot of {selected_col}")
            
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()
        
        # Correlation analysis
        if len(numeric_cols) > 1:
            st.subheader("🔗 Correlation Analysis")
            corr = df[numeric_cols].corr()
            
            fig, ax = plt.subplots(figsize=(10, 8))
            sns.heatmap(corr, annot=True, cmap='coolwarm', center=0, ax=ax)
            plt.title("Correlation Matrix")
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()
            
            # High correlation pairs
            high_corr = []
            for i in range(len(corr.columns)):
                for j in range(i+1, len(corr.columns)):
                    if abs(corr.iloc[i, j]) >= 0.8:
                        high_corr.append({
                            'Feature 1': corr.columns[i],
                            'Feature 2': corr.columns[j],
                            'Correlation': corr.iloc[i, j]
                        })
            
            if high_corr:
                st.write("**Highly Correlated Features (|r| ≥ 0.8):**")
                st.dataframe(pd.DataFrame(high_corr))
        
        # Clustering analysis
        if len(numeric_cols) > 0:
            st.subheader("🎯 Customer Segmentation (K-Means)")
            
            # Prepare data for clustering
            numeric_data = df[numeric_cols].fillna(0)
            scaler = StandardScaler()
            scaled_data = scaler.fit_transform(numeric_data)
            
            # K-means clustering
            kmeans = KMeans(n_clusters=3, random_state=42, n_init="auto")
            clusters = kmeans.fit_predict(scaled_data)
            df['cluster'] = clusters
            
            # Cluster profiles
            cluster_profile = df.groupby('cluster')[numeric_cols].mean()
            
            col1, col2 = st.columns(2)
            with col1:
                st.write("**Cluster Centers:**")
                st.dataframe(cluster_profile)
            
            with col2:
                if 'churn' in df.columns:
                    churn_by_cluster = df.groupby('cluster')['churn'].agg(['mean', 'count'])
                    churn_by_cluster.columns = ['Churn Rate', 'Customer Count']
                    st.write("**Churn Rate by Cluster:**")
                    st.dataframe(churn_by_cluster)
            
            # PCA visualization
            st.subheader("📊 PCA Visualization")
            pca = PCA(n_components=2)
            pca_result = pca.fit_transform(scaled_data)
            
            fig, ax = plt.subplots(figsize=(10, 8))
            scatter = ax.scatter(pca_result[:, 0], pca_result[:, 1], c=clusters, cmap='viridis')
            ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)')
            ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)')
            ax.set_title('Customer Segments (PCA)')
            plt.colorbar(scatter)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()
        
        # Machine Learning - Random Forest
        if 'churn' in df.columns and len(numeric_cols) > 0:
            st.subheader("🤖 Churn Prediction Model")
            
            # Prepare features
            X = df[numeric_cols].fillna(0)
            y = df['churn']
            
            # Train model
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
            
            rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
            rf_model.fit(X_train, y_train)
            
            # Predictions
            y_pred = rf_model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Model Accuracy", f"{accuracy:.1%}")
            
            with col2:
                st.metric("Training Samples", len(X_train))
                st.metric("Test Samples", len(X_test))
            
            # Feature importance
            feature_importance = pd.DataFrame({
                'Feature': numeric_cols,
                'Importance': rf_model.feature_importances_
            }).sort_values('Importance', ascending=False)
            
            st.write("**Feature Importance:**")
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.barplot(data=feature_importance.head(10), x='Importance', y='Feature', ax=ax)
            plt.title("Top 10 Most Important Features")
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()
        
        # Download results
        st.subheader("💾 Download Results")
        
        # Create summary report
        if st.button("Generate Analysis Report"):
            report_data = {
                'Metric': ['Total Customers', 'Features', 'Numeric Columns', 'Missing Values'],
                'Value': [len(df), len(df.columns), len(numeric_cols), missing.sum()]
            }
            
            if 'churn' in df.columns:
                report_data['Metric'].extend(['Churn Rate', 'Model Accuracy'])
                report_data['Value'].extend([f"{df['churn'].mean():.1%}", f"{accuracy:.1%}" if 'accuracy' in locals() else "N/A"]
            )
            
            report_df = pd.DataFrame(report_data)
            st.download_button(
                label="Download Analysis Report (CSV)",
                data=report_df.to_csv(index=False),
                file_name="churn_analysis_report.csv",
                mime="text/csv"
            )
            
    except Exception as e:
        st.error(f"❌ Error processing file: {str(e)}")
        st.write("Please make sure your CSV file is properly formatted and contains the expected columns.")

else:
    st.info("👆 Please upload a CSV file to begin the analysis")
    
    # Sample data structure
    st.subheader("📋 Expected Data Structure")
    st.write("""
    Your CSV file should contain:
    - **Customer features** (e.g., age, income, usage metrics)
    - **Churn column** (binary: 1 for churned, 0 for retained)
    - **No missing values** (or minimal missing data)
    
    Example columns:
    - customer_id, age, income, monthly_charges, total_charges, churn
    """)
    
    # Create sample data
    if st.button("Generate Sample Data"):
        np.random.seed(42)
        n_customers = 1000
        
        sample_data = pd.DataFrame({
            'customer_id': range(1, n_customers + 1),
            'age': np.random.normal(45, 15, n_customers).astype(int),
            'income': np.random.normal(50000, 20000, n_customers).astype(int),
            'monthly_charges': np.random.normal(65, 20, n_customers),
            'total_charges': np.random.normal(2000, 800, n_customers),
            'tenure_months': np.random.poisson(24, n_customers),
            'churn': np.random.binomial(1, 0.2, n_customers)
        })
        
        # Ensure reasonable values
        sample_data['age'] = sample_data['age'].clip(18, 80)
        sample_data['income'] = sample_data['income'].clip(20000, 150000)
        sample_data['monthly_charges'] = sample_data['monthly_charges'].clip(20, 150)
        sample_data['total_charges'] = sample_data['total_charges'].clip(100, 5000)
        sample_data['tenure_months'] = sample_data['tenure_months'].clip(1, 72)
        
        csv = sample_data.to_csv(index=False)
        st.download_button(
            label="Download Sample Data (CSV)",
            data=csv,
            file_name="sample_churn_data.csv",
            mime="text/csv"
        )
        
        st.success("✅ Sample data generated! Download and use it to test the dashboard.") 