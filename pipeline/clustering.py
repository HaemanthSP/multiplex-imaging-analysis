"""Clustering and subclustering routines."""

from __future__ import annotations

import numpy as np
import pandas as pd
import scanpy as sc
import anndata as ad

from .preprocessing import double_z_norm


# ---------------------------------------------------------------------------
# Initial clustering (Stage 3)
# ---------------------------------------------------------------------------


def run_leiden_pipeline(
    adata: ad.AnnData,
    n_neighbors: int = 9,
    n_pcs: int = 30,
    resolution: float = 1.0,
    random_seed: int = 42,
) -> ad.AnnData:
    """Run PCA → KNN → Leiden clustering on an AnnData object.

    Parameters
    ----------
    adata : ad.AnnData
        AnnData with normalised features in ``X``.
    n_neighbors, n_pcs, resolution : int/float
        Leiden pipeline hyperparameters.
    random_seed : int
        Random seed for reproducibility.

    Returns
    -------
    ad.AnnData
        Same AnnData with ``leiden`` column added to ``.obs``.
    """
    np.random.seed(random_seed)
    sc.tl.pca(adata, svd_solver="arpack")
    sc.pp.neighbors(adata, n_neighbors=n_neighbors, n_pcs=n_pcs)
    sc.tl.leiden(adata, resolution=resolution)
    return adata


def compute_top_markers(
    adata: ad.AnnData,
    groupby: str = "leiden",
    method: str = "wilcoxon",
    top_k: int = 5,
) -> pd.DataFrame:
    """Run differential expression and return top markers per cluster.

    Parameters
    ----------
    adata : ad.AnnData
        AnnData with cluster labels.
    groupby : str
        Column in ``adata.obs`` to group by.
    method : str
        DE method (default: ``'wilcoxon'``).
    top_k : int
        Number of top markers per cluster.

    Returns
    -------
    pd.DataFrame
        Top markers sorted by log fold change within each group.
    """
    sc.tl.rank_genes_groups(adata, groupby=groupby, method=method)
    marker_df = sc.get.rank_genes_groups_df(adata, group=None)
    top_markers = (
        marker_df.sort_values(["group", "logfoldchanges"], ascending=[True, False])
        .groupby("group")
        .head(top_k)
    )
    return top_markers


# ---------------------------------------------------------------------------
# Cluster means / pooling (Stages 3 & 5)
# ---------------------------------------------------------------------------


def prepare_cluster_means(
    adata: ad.AnnData,
    selected_features: list[str],
    unselected_features: list[str] | pd.Index,
    cluster_key: str = "leiden",
    custom_labels: dict[int, str] | None = None,
) -> pd.DataFrame:
    """Compute mean expression per cluster for all markers.

    Parameters
    ----------
    adata : ad.AnnData
        AnnData with cluster assignments.
    selected_features : list[str]
        Primary marker columns (from adata.X).
    unselected_features : list[str]
        Secondary marker columns (from adata.obs); will be double-z-normalised.
    cluster_key : str
        Column containing cluster IDs.
    custom_labels : dict, optional
        Mapping of cluster int → label string.

    Returns
    -------
    pd.DataFrame
        Cluster means with rows = clusters, columns = markers.
    """
    # Expression from X
    expr_df = pd.DataFrame(
        adata.X if not hasattr(adata.X, "toarray") else adata.X.toarray(),
        columns=selected_features,
    )

    # Add unselected features (double-z normalised)
    unselected_list = list(unselected_features)
    if unselected_list:
        unsel_data = adata.obs[unselected_list].values
        unsel_norm = double_z_norm(unsel_data)
        unsel_df = pd.DataFrame(unsel_norm, columns=unselected_list)
        expr_df = pd.concat([expr_df, unsel_df], axis=1)

    expr_df[cluster_key] = adata.obs[cluster_key].values

    cluster_means = expr_df.groupby(cluster_key).mean()

    if custom_labels is not None:
        cluster_means.index = cluster_means.index.map(
            lambda x: f"{x}:{custom_labels.get(int(x), 'Unspecific')}"
        )

    return cluster_means


def apply_cluster_labels(
    adata: ad.AnnData,
    labels_df: pd.DataFrame,
    cluster_key: str = "leiden",
) -> ad.AnnData:
    """Apply human-assigned labels and groups to an AnnData.

    Updates ``adata.obs`` with ``cluster_label`` and ``cluster_group`` columns.

    Parameters
    ----------
    adata : ad.AnnData
        AnnData with cluster IDs.
    labels_df : pd.DataFrame
        Must have columns: ``cluster_id``, ``label``, ``group``.
    cluster_key : str
        Column containing original cluster IDs.

    Returns
    -------
    ad.AnnData
        Updated AnnData.
    """
    label_map = dict(zip(labels_df["cluster_id"].astype(str), labels_df["label"]))
    group_map = dict(zip(labels_df["cluster_id"].astype(str), labels_df["group"]))

    adata.obs["cluster_label"] = (
        adata.obs[cluster_key].astype(str).map(label_map).fillna("Unlabelled")
    )
    adata.obs["cluster_group"] = (
        adata.obs[cluster_key].astype(str).map(group_map).fillna("Ungrouped")
    )

    # Create combined label: "cluster_id:label"
    adata.obs["cluster_labelled"] = (
        adata.obs[cluster_key].astype(str) + ":" + adata.obs["cluster_label"]
    )

    return adata


def pool_clusters(cluster_means: pd.DataFrame) -> pd.DataFrame:
    """Pool clusters sharing the same cell-type label.

    Expects index in format ``"cluster_id:label"``.

    Parameters
    ----------
    cluster_means : pd.DataFrame
        Cluster means with labelled index.

    Returns
    -------
    pd.DataFrame
        Pooled means with cell-type names as index.
    """
    pooled_names = cluster_means.index.map(
        lambda x: x.split(":", 1)[1] if ":" in x else x
    )
    temp = cluster_means.copy()
    temp["pooled_cluster"] = pooled_names
    pooled = temp.groupby("pooled_cluster").mean()

    print(f"Original clusters: {len(cluster_means)}")
    print(f"Pooled clusters: {len(pooled)}")
    return pooled


# ---------------------------------------------------------------------------
# Subclustering (Stage 6)
# ---------------------------------------------------------------------------


def subcluster_group(
    adata: ad.AnnData,
    group_name: str,
    group_col: str = "cluster_group",
    resolution: float = 0.5,
    n_neighbors: int = 9,
    n_pcs: int = 30,
    unselected_features: list[str] | None = None,
) -> ad.AnnData:
    """Subcluster cells belonging to a specific meta-cluster group.

    Extracts cells, re-normalises, and runs a fresh Leiden clustering.
    When *unselected_features* is provided the extra marker columns
    (stored in ``adata.obs``) are double-z-normalised and concatenated
    with ``X`` so that **all** markers drive the subclustering.

    Parameters
    ----------
    adata : ad.AnnData
        Full AnnData (cross-machine merged).
    group_name : str
        Value of *group_col* to subset.
    group_col : str
        Column identifying meta-cluster groups.
    resolution : float
        Leiden resolution for subclustering.
    n_neighbors, n_pcs : int
        KNN parameters.
    unselected_features : list[str], optional
        Extra marker columns in ``adata.obs`` to include in the
        subclustering feature matrix (will be double-z-normalised).

    Returns
    -------
    ad.AnnData
        Subset AnnData with ``sub_cluster`` in ``.obs``.
        If *unselected_features* was provided, ``X`` and ``var_names``
        are expanded to cover all markers.
    """
    mask = adata.obs[group_col] == group_name
    subset = adata[mask].copy()

    if subset.n_obs < 10:
        print(f"Warning: only {subset.n_obs} cells in group '{group_name}', skipping")
        subset.obs["sub_cluster"] = "0"
        return subset

    from sklearn.preprocessing import StandardScaler

    # --- Build combined feature matrix (selected + unselected) ----------
    selected_arr = np.asarray(
        subset.X if not hasattr(subset.X, "toarray") else subset.X.toarray(),
        dtype=np.float64,
    )
    selected_names = list(subset.var_names)

    if unselected_features:
        # Keep only columns that actually exist in this subset's obs
        extra_cols = [c for c in unselected_features if c in subset.obs.columns]
        if extra_cols:
            extra_arr = subset.obs[extra_cols].values.astype(np.float64)
            extra_arr = double_z_norm(extra_arr)
            combined = np.concatenate([selected_arr, extra_arr], axis=1)
            all_names = selected_names + extra_cols
        else:
            combined = selected_arr
            all_names = selected_names
    else:
        combined = selected_arr
        all_names = selected_names

    # Rebuild subset AnnData with expanded feature matrix
    subset = ad.AnnData(
        X=combined,
        obs=subset.obs.copy(),
    )
    subset.var_names = pd.Index(all_names)

    # Re-normalise the full X
    scaler = StandardScaler()
    subset.X = scaler.fit_transform(subset.X)

    effective_pcs = min(n_pcs, subset.n_vars - 1, subset.n_obs - 1)
    sc.tl.pca(subset, svd_solver="arpack", n_comps=effective_pcs)
    sc.pp.neighbors(subset, n_neighbors=min(n_neighbors, subset.n_obs - 1), n_pcs=effective_pcs)
    sc.tl.leiden(subset, resolution=resolution, key_added="sub_cluster")

    return subset


def subcluster_at_resolutions(
    adata: ad.AnnData,
    group_name: str,
    resolutions: list[float],
    group_col: str = "cluster_group",
    n_neighbors: int = 9,
    n_pcs: int = 30,
    unselected_features: list[str] | None = None,
) -> dict[float, ad.AnnData]:
    """Run subclustering at multiple resolutions for comparison.

    Parameters
    ----------
    adata : ad.AnnData
        Full merged AnnData.
    group_name : str
        Meta-cluster group to subcluster.
    resolutions : list[float]
        List of Leiden resolutions to try.
    group_col : str
        Column with group labels.
    n_neighbors, n_pcs : int
        KNN parameters.
    unselected_features : list[str], optional
        Extra marker columns in ``adata.obs`` to include.

    Returns
    -------
    dict[float, ad.AnnData]
        Mapping of resolution → subclustered AnnData.
    """
    results = {}
    for res in resolutions:
        print(f"  Subclustering '{group_name}' at resolution={res:.2f}")
        results[res] = subcluster_group(
            adata, group_name, group_col, resolution=res,
            n_neighbors=n_neighbors, n_pcs=n_pcs,
            unselected_features=unselected_features,
        )
        n_clusters = results[res].obs["sub_cluster"].nunique()
        print(f"    → {n_clusters} sub-clusters, {results[res].n_obs} cells")
    return results


def subcluster_builtin(
    adata: ad.AnnData,
    cluster_key: str = "leiden",
    clusters_to_split: list[str] | None = None,
    resolutions: dict[str, float] | float | None = None,
    key_added: str = "leiden_subclustered",
) -> ad.AnnData:
    """Subcluster selected clusters using scanpy's restrict_to mechanism.

    Parameters
    ----------
    adata : ad.AnnData
        AnnData with existing clusters.
    cluster_key : str
        Column with original cluster IDs.
    clusters_to_split : list[str]
        Cluster IDs to subcluster.
    resolutions : dict or float, optional
        Per-cluster resolution or uniform value.
    key_added : str
        Name for the result column.

    Returns
    -------
    ad.AnnData
        Updated AnnData with subclustered labels.
    """
    if clusters_to_split is None:
        clusters_to_split = []

    if resolutions is None:
        resolutions = {c: 0.5 for c in clusters_to_split}
    elif isinstance(resolutions, (int, float)):
        resolutions = {c: resolutions for c in clusters_to_split}

    adata.obs[key_added] = adata.obs[cluster_key].astype(str).copy()

    for cluster_id in clusters_to_split:
        cid = str(cluster_id)
        res = resolutions.get(cid, 0.5)
        print(f"Subclustering cluster {cid} at resolution {res}")

        temp_key = f"_temp_sub_{cid}"
        sc.tl.leiden(
            adata,
            restrict_to=(cluster_key, [cid]),
            resolution=res,
            key_added=temp_key,
        )

        mask = adata.obs[cluster_key].astype(str) == cid
        new_labels = adata.obs.loc[mask, temp_key].astype(str).str.replace(",", ".")
        adata.obs.loc[mask, key_added] = new_labels
        adata.obs.drop(columns=[temp_key], inplace=True)

        n_sub = adata.obs.loc[mask, key_added].nunique()
        print(f"  → {n_sub} subclusters")

    adata.obs[key_added] = adata.obs[key_added].astype("category")
    return adata
