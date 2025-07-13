"""
helpers/ml_models.py

This module contains functions related to Machine Learning model training,
prediction, and interpretation (e.g., using SHAP).

By centralizing ML logic, we can manage model versions, apply consistent
preprocessing, and create a clear boundary between the model and the application
that uses it. Caching is used heavily to prevent re-training models on every
user interaction, ensuring a responsive user experience.

Author: AI Engineering SME
Version: 29.1 (Definitive Final Build)
Date: 2025-07-13

Changelog from v28.1:
- [ROBUSTNESS] Added comprehensive checks in all functions to handle empty or
  invalid input DataFrames gracefully, preventing runtime errors and providing
  clear warnings to the user.
- [MAINTAINABILITY] Corrected the caching decorator usage to align with
  Streamlit's best practices (`@st.cache_resource` for non-serializable
  objects like models, `@st.cache_data` for serializable data like DataFrames).
- [MAINTAINABILITY] Upgraded `KMeans` instantiation in `perform_text_clustering`
  to use `n_init='auto'`, which is the modern default and avoids a
  `FutureWarning`, ensuring long-term compatibility.
- [DOC] Upgraded all docstrings to a professional standard (Google-style) with
  clear explanations of parameters, return values, and caching rationale.
- [STYLE] Reviewed and confirmed comprehensive type hints for all function
  signatures.
"""

import pandas as pd
import streamlit as st
import numpy as np
import shap
from typing import Dict, Any
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, DBSCAN

# A constant for reproducibility of stochastic processes
MODEL_SEED = 42


# ==============================================================================
# 1. REGRESSION & EXPLAINABILITY (XAI)
# ==============================================================================

@st.cache_resource(show_spinner="Training ML models...")
def train_regression_models(df: pd.DataFrame, target_column: str) -> Dict[str, Any]:
    """Trains Linear Regression and Random Forest models.

    Uses `@st.cache_resource` because it returns complex, non-serializable
    objects (the trained scikit-learn model objects). Caching prevents costly
    re-training on every app interaction.

    Args:
        df: Input DataFrame with features and the target.
        target_column: The name of the target variable column.

    Returns:
        A dictionary with trained models, data, predictions, and metrics.
    """
    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' not found in DataFrame.")
    if df.empty:
        st.warning("Input data for model training is empty.")
        return {}

    X = df.drop(columns=[target_column])
    y = df[target_column]

    lin_reg = LinearRegression().fit(X, y)
    y_pred_lin = lin_reg.predict(X)
    r2_lin = lin_reg.score(X, y)

    rf_reg = RandomForestRegressor(
        n_estimators=100, random_state=MODEL_SEED, oob_score=True, n_jobs=-1
    ).fit(X, y)
    y_pred_rf = rf_reg.predict(X)
    r2_rf = rf_reg.oob_score_

    return {
        "X": X, "y": y,
        "linear_model": lin_reg, "linear_predictions": y_pred_lin, "linear_r2": r2_lin,
        "rf_model": rf_reg, "rf_predictions": y_pred_rf, "rf_oob_r2": r2_rf,
    }


@st.cache_resource(show_spinner="Calculating SHAP values...")
def get_shap_explanation(_model: RandomForestRegressor, X: pd.DataFrame) -> shap.Explanation:
    """Computes SHAP values to explain a trained tree-based model.

    Uses `@st.cache_resource` as SHAP explainers and values can be large,
    non-serializable objects and are computationally intensive to generate.

    Args:
        _model: The trained scikit-learn tree-based model.
        X: The DataFrame of features used for training.

    Returns:
        A SHAP explanation object.
    """
    if X.empty:
        st.warning("Input data for SHAP explanation is empty.")
        return shap.Explanation(values=np.array([[]]), base_values=np.array([]), data=np.array([[]]), feature_names=[])

    explainer = shap.TreeExplainer(_model)
    shap_values = explainer(X)
    return shap_values


# ==============================================================================
# 2. CLUSTERING & UNSUPERVISED LEARNING
# ==============================================================================

@st.cache_data(show_spinner="Clustering risk signals...")
def perform_risk_signal_clustering(df: pd.DataFrame) -> pd.DataFrame:
    """Applies DBSCAN clustering to identify groups and anomalies in process data.

    Uses `@st.cache_data` because the function takes and returns a serializable
    DataFrame, making this the most efficient caching strategy.

    Args:
        df: DataFrame with numeric columns for clustering.

    Returns:
        The original DataFrame with an added 'cluster' column.
    """
    numeric_cols = df.select_dtypes(include=np.number).columns
    if len(numeric_cols) < 1:
        st.warning("No numeric columns found for clustering. Returning original data.")
        return df.copy()

    X_cluster = df[numeric_cols].dropna()
    if X_cluster.empty:
        st.warning("No data remains after dropping NaNs for clustering.")
        return df.copy()

    dbscan = DBSCAN(eps=3, min_samples=3).fit(X_cluster)
    
    df_clustered = df.copy()
    df_clustered['cluster'] = pd.Series(dbscan.labels_, index=X_cluster.index).astype(str)
    df_clustered['cluster'] = df_clustered['cluster'].replace('-1', 'Outlier/Anomaly').fillna('N/A')
    
    return df_clustered


@st.cache_data(show_spinner="Analyzing adverse event text...")
def perform_text_clustering(df: pd.DataFrame, text_column: str) -> pd.DataFrame:
    """Performs NLP-based clustering on unstructured text data.

    This pipeline uses TF-IDF, PCA, and K-Means. Uses `@st.cache_data` for
    efficiency with data I/O.

    Args:
        df: DataFrame containing the text data.
        text_column: The name of the column with text narratives.

    Returns:
        A new DataFrame with added columns for cluster labels and PCA coordinates.
    """
    if text_column not in df.columns or df[text_column].dropna().empty:
        st.warning(f"Text column '{text_column}' not found or is empty.")
        return pd.DataFrame(columns=list(df.columns) + ['cluster', 'x_pca', 'y_pca'])

    text_data = df[text_column].dropna()
    
    vectorizer = TfidfVectorizer(stop_words='english', max_features=1000)
    X_text = vectorizer.fit_transform(text_data)

    pca = PCA(n_components=2, random_state=MODEL_SEED)
    X_pca = pca.fit_transform(X_text.toarray())

    kmeans = KMeans(n_clusters=4, random_state=MODEL_SEED, n_init='auto')
    cluster_labels = kmeans.fit_predict(X_text)
    
    df_clustered = df.loc[text_data.index].copy()
    df_clustered['cluster'] = cluster_labels
    df_clustered['x_pca'] = X_pca[:, 0]
    df_clustered['y_pca'] = X_pca[:, 1]
    
    return df_clustered


# ==============================================================================
# 3. NATURAL LANGUAGE PROCESSING (NLP)
# ==============================================================================

@st.cache_data
def perform_topic_modeling_on_capa(df: pd.DataFrame, description_col: str) -> pd.DataFrame:
    """Performs simple, keyword-based topic modeling on CAPA descriptions.

    Uses `@st.cache_data` as it transforms a DataFrame into another DataFrame.

    Args:
        df: The DataFrame containing CAPA logs.
        description_col: The column with the text descriptions.

    Returns:
        A DataFrame summarizing the frequency of each identified topic.
    """
    if description_col not in df.columns or df[description_col].dropna().empty:
        st.warning(f"Description column '{description_col}' not found or is empty.")
        return pd.DataFrame(columns=['Topic', 'Frequency'])

    text_series = df[description_col].dropna()

    topics = {
        "Reagent/Storage Issue": "enzyme|degradation|lot|freezer|stored|mobile phase",
        "Contamination": "contamination|aerosolization|negative control",
        "Hardware Failure": "thermal cycler|calibration|overshoot|robot|clogged|nozzle",
        "Human Error": "pipetting|inconsistent volumes|mis-labeling|mix-up"
    }
    
    topic_counts = {
        topic: text_series.str.contains(pattern, case=False, regex=True).sum()
        for topic, pattern in topics.items()
    }

    df_topics = pd.DataFrame(
        list(topic_counts.items()),
        columns=['Topic', 'Frequency']
    ).sort_values('Frequency', ascending=False)
    
    return df_topics
