"""
helpers/ml_models.py

This module contains functions related to Machine Learning model training,
prediction, and interpretation (e.g., using SHAP).

By centralizing ML logic, we can easily manage model versions, apply consistent
preprocessing, and create a clear boundary between the model and the application
that uses it. Caching is used heavily to prevent re-training models on every
user interaction, ensuring a responsive user experience.

Author: AI Engineering SME
Version: 24.1 (SME Refactored Build)
Date: 2024-05-21

Changelog from v23.1:
- [FIX] Corrected the use of Streamlit caching decorators.
    - `train_regression_models` and `get_shap_explanation` now correctly use
      `@st.cache_resource` as they return complex, non-serializable objects
      (models, SHAP explainers).
    - `perform_risk_signal_clustering`, `perform_text_clustering`, and
      `perform_topic_modeling_on_capa` now correctly use `@st.cache_data` as
      they return serializable DataFrames. This is more efficient and aligns
      with Streamlit's caching strategy.
- [FIX] In `perform_topic_modeling_on_capa`, added a check for empty
  `description_col` to prevent potential errors with `str.contains`.
- [REFACTOR] Upgraded `KMeans` instantiation in `perform_text_clustering` to
  use `n_init='auto'`, which is the modern default and avoids a
  `FutureWarning`.
- [STYLE] Added comprehensive type hints to all function signatures and docstrings
  for improved code quality and maintainability.
- [DOC] Updated docstrings to clarify the rationale for choosing specific
  caching decorators for each function.
"""

import pandas as pd
import streamlit as st
import shap
from typing import Dict, Any
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
def train_regression_models(df: pd.DataFrame, target_column: str) -> Dict[str, Any]:
    """
    Trains Linear Regression and Random Forest models.

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

    X = df.drop(columns=[target_column])
    y = df[target_column]

    # --- Train Linear Regression Model ---
    lin_reg = LinearRegression().fit(X, y)
    y_pred_lin = lin_reg.predict(X)
    r2_lin = lin_reg.score(X, y)

    # --- Train Random Forest Regressor ---
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
def get_shap_explanation(_model: RandomForestRegressor, X: pd.DataFrame) -> shap.Explanation:
    """
    Computes SHAP values to explain a trained tree-based model.

    Uses `@st.cache_resource` as SHAP explainers and values can be large,
    non-serializable objects and are computationally intensive to generate.

    Args:
        _model: The trained scikit-learn tree-based model. The underscore
                prefix indicates it's used to create the cached resource.
        X: The DataFrame of features used for training.

    Returns:
        A SHAP explanation object containing the values and base rates.
    """
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

    Uses `@st.cache_data` because the function takes a DataFrame and returns a
    DataFrame, which is a serializable object. This is more efficient for
    data-only transformations. DBSCAN is chosen for its ability to find
    outliers without pre-specifying cluster counts.

    Args:
        df: DataFrame with numeric columns for clustering.

    Returns:
        The original DataFrame with an added 'cluster' column.
    """
    numeric_cols = df.select_dtypes(include=['number']).columns
    if len(numeric_cols) < 1:
        st.warning("No numeric columns found for clustering. Returning original data.")
        return df
        
    X_cluster = df[numeric_cols]

    # These parameters may need tuning for real-world data.
    dbscan = DBSCAN(eps=3, min_samples=3).fit(X_cluster)
    
    df_clustered = df.copy()
    # Convert labels to string for consistent categorical plotting.
    df_clustered['cluster'] = [str(c) for c in dbscan.labels_]
    # Label noise points as 'Outlier/Anomaly' for clarity.
    df_clustered.loc[df_clustered['cluster'] == '-1', 'cluster'] = 'Outlier/Anomaly'
    
    return df_clustered


@st.cache_data(show_spinner="Analyzing adverse event text...")
def perform_text_clustering(df: pd.DataFrame, text_column: str) -> pd.DataFrame:
    """
    Performs NLP-based clustering on unstructured text data.

    This pipeline uses TF-IDF to vectorize text, PCA for dimensionality
    reduction, and K-Means for clustering. Uses `@st.cache_data` for
    efficiency with data I/O.

    Args:
        df: DataFrame containing the text data.
        text_column: The name of the column with text narratives.

    Returns:
        DataFrame with added columns for cluster labels and PCA coordinates.
    """
    if text_column not in df.columns or df[text_column].empty:
        st.warning(f"Text column '{text_column}' not found or is empty.")
        return df

    # 1. Vectorize Text using TF-IDF
    vectorizer = TfidfVectorizer(stop_words='english', max_features=1000)
    X_text = vectorizer.fit_transform(df[text_column])

    # 2. Reduce Dimensionality with PCA
    pca = PCA(n_components=2, random_state=MODEL_SEED)
    X_pca = pca.fit_transform(X_text.toarray())

    # 3. Cluster with K-Means
    # REFACTOR: Use n_init='auto' to adopt the modern scikit-learn default.
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
    Performs simple, keyword-based topic modeling on CAPA descriptions.

    Uses `@st.cache_data` as it transforms a DataFrame into another DataFrame.
    A rule-based approach is used for simplicity and interpretability.

    Args:
        df: The DataFrame containing CAPA logs.
        description_col: The column with the text descriptions.

    Returns:
        A DataFrame summarizing the frequency of each identified topic.
    """
    if description_col not in df.columns or df[description_col].isna().all():
        st.warning(f"Description column '{description_col}' not found or is empty.")
        return pd.DataFrame(columns=['Topic', 'Frequency'])

    # Define keywords that characterize each topic.
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

    df_topics = pd.DataFrame(
        list(topic_counts.items()),
        columns=['Topic', 'Frequency']
    ).sort_values('Frequency', ascending=False)
    
    return df_topics
