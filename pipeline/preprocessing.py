"""Preprocessing utilities: normalization, feature selection, AnnData construction."""

from __future__ import annotations

import numpy as np
import pandas as pd
import anndata as ad
from sklearn.preprocessing import StandardScaler


def double_z_norm(data: pd.DataFrame | np.ndarray) -> np.ndarray:
    """Double z-score normalization.

    1. Z-score across features (columns)  – StandardScaler on columns.
    2. Z-score across samples (rows) – StandardScaler on transposed, then transpose back.

    Parameters
    ----------
    data : pd.DataFrame or np.ndarray
        Expression matrix (samples × features).

    Returns
    -------
    np.ndarray
        Normalised array of the same shape.
    """
    arr = np.asarray(data, dtype=np.float64)
    # Step 1: z-score across features (columns)
    scaler1 = StandardScaler()
    step1 = scaler1.fit_transform(arr)
    # Step 2: z-score across samples (rows)
    scaler2 = StandardScaler()
    normalised = scaler2.fit_transform(step1.T).T
    return normalised


def clean_column_names(df: pd.DataFrame, suffix: str = "Biomarker Exp") -> pd.DataFrame:
    """Strip a common suffix from column names.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe.
    suffix : str
        Suffix to remove (default: ``"Biomarker Exp"``).

    Returns
    -------
    pd.DataFrame
        Copy with cleaned column names.
    """
    df = df.copy()
    df.columns = df.columns.str.replace(suffix, "", regex=False).str.strip()
    return df


def select_features(
    df: pd.DataFrame,
    selected_markers: list[str],
    meta_columns: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split a combined dataframe into selected features, unselected features, and metadata.

    Parameters
    ----------
    df : pd.DataFrame
        Cleaned combined dataframe.
    selected_markers : list[str]
        Marker columns to use for clustering.
    meta_columns : list[str]
        Metadata columns to preserve.

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]
        ``(selected_feat_exp, unselected_feat_exp, metadata)``
    """
    feature_cols = df.columns.difference(meta_columns)
    unselected = feature_cols.difference(selected_markers)

    selected_feat_exp = df[selected_markers].copy()
    unselected_feat_exp = df[unselected].copy()
    metadata = df[df.columns.difference(selected_markers)].copy()

    return selected_feat_exp, unselected_feat_exp, metadata


def build_adata(
    selected_feat_exp: pd.DataFrame,
    metadata: pd.DataFrame,
    selected_markers: list[str],
) -> ad.AnnData:
    """Construct an AnnData object from normalised features and metadata.

    Parameters
    ----------
    selected_feat_exp : pd.DataFrame
        Selected marker expression values (raw, will be normalised here).
    metadata : pd.DataFrame
        Observation metadata.
    selected_markers : list[str]
        Names of the selected markers.

    Returns
    -------
    ad.AnnData
        AnnData with ``X`` = double-z-normalised selected features.
    """
    scaled = double_z_norm(selected_feat_exp)
    adata = ad.AnnData(X=scaled)
    adata.var_names = pd.Index(selected_markers)
    adata.obs = metadata.reset_index(drop=True)
    return adata
