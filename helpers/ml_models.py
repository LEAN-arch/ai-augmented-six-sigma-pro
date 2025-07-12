"""
helpers/ml_models.py

This module contains functions related to Machine Learning model training,
prediction, and interpretation (e.g., using SHAP).

By centralizing ML logic, we can easily manage model versions, apply consistent
preprocessing, and create a clear boundary between the model and the application
that uses it. Caching is used heavily to prevent re-training models on every
user interaction, ensuring a responsive user experience.

Author: AI Engineering SME
Version: 23.1 (Commercial Grade Refactor)
Date: 2023-10-26
"""

import pandas as pd
import streamlit as st
import shap
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, DBSCAN

# A constant for reproducibility of the Random Forest model and other stochastic processes
MODEL_SEED = 42

# ==============================================================================
# 1. REGRESSION & EXPLAINABILITY (XAI)
# ==============================================================================

@st.cache_resource(show_spinner="Training ML models...")
def train_regression_models(df: pd.DataFrame, target_column: str) -> dict:
    """
    Trains both a Linear Regression and a Random Forest Regressor model on the
    provided data.

    This function is cached using @st.cache_resource because it returns complex,
    non-serializable objects (the trained model objects). Caching prevents
    costly re-training on every app interaction.

    :param df: The input DataFrame containing features and the target.
    :type df: pd.DataFrame
    :param target_column: The name of the column to be used as the target variable.
    :type target_column: str
    :return: A dictionary containing the trained models, feature data (X),
             target data (y), and model performance metrics.
    :rtype: dict
    """
    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' not found in DataFrame.")

    X = df.drop(columns=[target_column])
    y = df[target_column]

    # --- Train Linear Regression Model ---
    lin_reg = LinearRegression().fit(X, y)
    y_pred_lin = lin_reg.predict(X)
    r2_lin = lin_reg.score(X, y)

    # --- Train Random Forest Regressor ---
    # n_estimators is a key hyperparameter. 100 is a reasonable default.
    # oob_score provides a robust estimate of generalization performance without
    # needing a separate validation set.
    rf_reg = RandomForestRegressor(
        n_estimators=100,
        random_state=MODEL_SEED,
        oob_score=True,
        n_jobs=-1  # Use all available CPU cores
    ).fit(X, y)
    y_pred_rf = rf_reg.predict(X)
    r2_rf = rf_reg.oob_score_

    return {
        "X": X,
        "y": y,
        "linear_model": lin_reg,
        "linear_predictions": y_pred_lin,
        "linear_r2": r2_lin,
        "rf_model": rf_reg,
        "rf_predictions": y_pred_rf,
        "rf_oob_r2": r2_rf,
    }

@st.cache_resource(show_spinner="Calculating SHAP values...")
def get_shap_explanation(_model: RandomForestRegressor, X: pd.DataFrame):
    """
    Computes SHAP values to explain a trained tree-based model.

    Uses @st.cache_resource as SHAP explainers and values can be large objects
    and computationally intensive to generate. This separates the SHAP
    calculation from the plotting logic.

    :param _model: The trained scikit-learn tree-based model (e.g., RandomForest).
                   The underscore prefix indicates it's used to create the resource.
    :type _model: RandomForestRegressor
    :param X: The dataframe of features used for training the model.
    :type X: pd.DataFrame
    :return: A SHAP explanation object containing the values and base rates.
    """
    # SHAP (SHapley Additive exPlanations) is a game theoretic approach to
    # explain the output of any machine learning model. TreeExplainer is
    # optimized for tree-based models like Random Forest.
    # Reference: https://shap.readthedocs.io/en/latest/
    explainer = shap.TreeExplainer(_model)
    shap_values = explainer(X)
    return shap_values

# ==============================================================================
# 2. CLUSTERING & UNSUPERVISED LEARNING
# ==============================================================================

@st.cache_data(show_spinner="Clustering risk signals...")
def perform_risk_signal_clustering(df: pd.DataFrame) -> pd.DataFrame:
    """
    Applies DBSCAN clustering to identify groups and anomalies in process data.

    DBSCAN is chosen as it does not require specifying the number of clusters
    beforehand and is excellent at identifying noise/outliers.

    :param df: DataFrame with numeric columns for clustering (e.g., 'Temp_C', 'Pressure_psi').
    :return: The original DataFrame with an added 'cluster' column.
    """
    # Use only numeric columns for clustering
    numeric_cols = df.select_dtypes(include=['number']).columns
    if len(numeric_cols) < 1:
        raise ValueError("DataFrame must contain numeric columns for clustering.")
        
    X_cluster = df[numeric_cols]

    # DBSCAN parameters (eps and min_samples) are crucial. These defaults are
    # chosen for the synthetic data but would need tuning for real data.
    dbscan = DBSCAN(eps=3, min_samples=3).fit(X_cluster)
    
    df_clustered = df.copy()
    df_clustered['cluster'] = [str(c) for c in dbscan.labels_]
    # Label the noise points identified by DBSCAN as 'Outlier/Anomaly'
    df_clustered.loc[df_clustered['cluster'] == '-1', 'cluster'] = 'Outlier/Anomaly'
    
    return df_clustered

@st.cache_data(show_spinner="Analyzing adverse event text...")
def perform_text_clustering(df: pd.DataFrame, text_column: str) -> pd.DataFrame:
    """
    Performs NLP-based clustering on unstructured text data (e.g., event logs).

    This pipeline uses TF-IDF to vectorize text, PCA for dimensionality reduction,
    and K-Means for clustering.

    :param df: DataFrame containing the text data.
    :param text_column: The name of the column with the text narratives.
    :return: DataFrame with added columns for cluster labels and PCA coordinates.
    """
    # 1. Vectorize Text using TF-IDF
    # Term Frequency-Inverse Document Frequency is a standard for converting
    # text into a meaningful numerical representation.
    vectorizer = TfidfVectorizer(stop_words='english', max_features=1000)
    X_text = vectorizer.fit_transform(df[text_column])

    # 2. Reduce Dimensionality with PCA
    # This is necessary to visualize the clusters in 2D space.
    pca = PCA(n_components=2, random_state=MODEL_SEED)
    X_pca = pca.fit_transform(X_text.toarray())

    # 3. Cluster with K-Means
    # We pre-specify 4 clusters based on domain knowledge of the synthetic data.
    # n_init='auto' is the modern default to avoid FutureWarning.
    kmeans = KMeans(n_clusters=4, random_state=MODEL_SEED, n_init='auto')
    
    df_clustered = df.copy()
    df_clustered['cluster'] = kmeans.fit_predict(X_text)
    df_clustered['x_pca'] = X_pca[:, 0]
    df_clustered['y_pca'] = X_pca[:, 1]
    
    return df_clustered

# ==============================================================================
# 3. NATURAL LANGUAGE PROCESSING (NLP)
# ==============================================================================

@st.cache_data
def perform_topic_modeling_on_capa(df: pd.DataFrame, description_col: str) -> pd.DataFrame:
    """
    Performs a simple, keyword-based topic modeling on CAPA descriptions.

    In a real-world scenario, more advanced methods like Latent Dirichlet
    Allocation (LDA) or transformer-based models would be used. For this app,
    a rule-based approach is sufficient and highly interpretable.

    :param df: The DataFrame containing CAPA logs.
    :param description_col: The column with the text descriptions.
    :return: A DataFrame summarizing the frequency of each identified topic.
    """
    # Define keywords that characterize each topic.
    # The '|' acts as an OR operator in the regex pattern.
    topics = {
        "Reagent/Storage Issue": "enzyme|degradation|lot|freezer|stored|mobile phase",
        "Contamination": "contamination|aerosolization|negative control",
        "Hardware Failure": "thermal cycler|calibration|overshoot|robot|clogged|nozzle",
        "Human Error": "pipetting|inconsistent volumes|mis-labeling|mix-up"
    }
    
    topic_counts = {
        topic: df[description_col].str.contains(pattern, case=False, regex=True).sum()
        for topic, pattern in topics.items()
    }

    # Convert the dictionary to a DataFrame for easier plotting.
    df_topics = pd.DataFrame(
        list(topic_counts.items()),
        columns=['Topic', 'Frequency']
    ).sort_values('Frequency', ascending=False)
    
    return df_topics
