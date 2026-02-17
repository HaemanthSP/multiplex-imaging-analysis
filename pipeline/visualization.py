"""Visualization utilities: dot plots, heatmaps, spatial plots."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
import seaborn as sns


def plot_dotplot(
    cluster_means: pd.DataFrame,
    x_labels: list[str],
    y_labels: list[str],
    save_path: str | Path = "dotplot.svg",
    title: str = "Mean Marker Expression per Cluster",
    show: bool = True,
) -> tuple[plt.Figure, plt.Axes]:
    """Create a dot plot of mean marker expression per cluster.

    Parameters
    ----------
    cluster_means : pd.DataFrame
        Rows = clusters, columns = markers.
    x_labels : list[str]
        Cluster labels for x-axis.
    y_labels : list[str]
        Marker labels for y-axis.
    save_path : str or Path
        Output file path.
    title : str
        Plot title.
    show : bool
        Whether to call ``plt.show()``.

    Returns
    -------
    tuple[Figure, Axes]
    """
    fig_height = max(5, len(y_labels) * 0.3)
    fig, ax = plt.subplots(figsize=(len(x_labels) * 0.3, fig_height))

    vmax = cluster_means.values.max()
    vmin = cluster_means.values.min()
    norm = plt.Normalize(vmin=vmin, vmax=vmax)
    cmap = plt.cm.RdYlBu_r

    for x_idx, cluster in enumerate(x_labels):
        for y_idx, gene in enumerate(y_labels):
            val = cluster_means.loc[cluster, gene]
            safe_val = max(val, 0.001) if not np.isnan(val) else 0.001
            dot_size = np.sqrt(safe_val) * 39

            ax.scatter(
                x_idx,
                y_idx,
                s=dot_size,
                color=cmap(norm(val)),
                edgecolors="grey",
                linewidth=0.3,
            )

    ax.set_xticks(range(len(x_labels)))
    ax.set_xticklabels(x_labels, rotation=89, ha="right", fontsize=6)
    ax.set_yticks(range(len(y_labels)))
    ax.set_yticklabels(y_labels, fontsize=5)
    ax.set_xlim(-0.5, len(x_labels) - 0.5)
    ax.set_ylim(-0.5, len(y_labels) - 0.5)
    ax.set_title(title, fontsize=7)

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, fraction=0.05, pad=0.04, shrink=0.6)
    cbar.set_label("Mean Expression", fontsize=7)

    ax.grid(False)
    plt.tight_layout()

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=299, bbox_inches="tight")
    if show:
        plt.show()

    return fig, ax


def plot_heatmap(
    cluster_means: pd.DataFrame,
    x_labels: list[str],
    y_labels: list[str],
    save_path: str | Path = "heatmap.svg",
    title: str = "Mean Marker Expression per Cluster",
    show: bool = True,
) -> plt.Figure:
    """Create a heatmap of mean marker expression per cluster.

    Parameters
    ----------
    cluster_means : pd.DataFrame
        Rows = clusters, columns = markers.
    x_labels : list[str]
        Cluster labels.
    y_labels : list[str]
        Marker labels.
    save_path : str or Path
        Output file path.
    title : str
        Plot title.
    show : bool
        Whether to call ``plt.show()``.

    Returns
    -------
    Figure
    """
    data = cluster_means[y_labels].T

    num_markers = len(y_labels)
    num_clusters = len(x_labels)
    fig_width = max(7, num_clusters * 0.4)

    if num_markers <= 40:
        fig_height = max(5, num_markers * 0.3)
        y_fontsize = 5
    elif num_markers <= 99:
        fig_height = max(9, min(20, num_markers * 0.1))
        y_fontsize = 3
    else:
        fig_height = max(14, min(30, num_markers * 0.1))
        y_fontsize = 3

    fig = plt.figure(figsize=(fig_width, fig_height))

    sns.heatmap(
        data,
        cmap="RdYlBu_r",
        cbar_kws={"label": "Mean Expression", "shrink": 0.8},
        linewidths=0.5,
        linecolor="lightgrey",
        yticklabels=True,
    )

    plt.title(title, fontsize=7)
    plt.xlabel("Clusters", fontsize=7)
    plt.ylabel("Marker Genes", fontsize=7)
    plt.xticks(rotation=89, ha="right", fontsize=8)

    ax = plt.gca()
    ax.set_yticks(range(len(y_labels)))
    ax.set_yticklabels(y_labels, rotation=0, fontsize=y_fontsize)

    plt.tight_layout()

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=299, bbox_inches="tight")
    if show:
        plt.show()

    return fig


def plot_spatial(
    adata,
    save_path: str | Path,
    cluster_col: str = "leiden",
    x_col: str = "Nuc X",
    y_col: str = "Nuc Y Inv",
    title: str = "Spatial View of Clusters",
    show: bool = True,
) -> plt.Figure:
    """Plot spatial coordinates colored by cluster assignment.

    Parameters
    ----------
    adata : AnnData
        AnnData with spatial coordinates and cluster assignments.
    save_path : str or Path
        Output file path.
    cluster_col : str
        Column with cluster labels.
    x_col, y_col : str
        Spatial coordinate columns.
    title : str
        Plot title.
    show : bool
        Whether to call ``plt.show()``.

    Returns
    -------
    Figure
    """
    fig = plt.figure(figsize=(7, 6))
    sns.scatterplot(
        x=adata.obs[x_col],
        y=adata.obs[y_col],
        hue=adata.obs[cluster_col],
        palette="tab20",
        s=9,
        linewidth=0,
    )
    plt.title(title)
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.legend(title="Cluster", bbox_to_anchor=(0.05, 1), loc="upper left", fontsize=5)
    plt.tight_layout()

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(f"{save_path}.png", dpi=299, bbox_inches="tight")
    if show:
        plt.show()

    return fig


def plot_umap(
    adata,
    color: str = "leiden",
    title: str = "Leiden Clustering (UMAP)",
    show: bool = True,
) -> None:
    """Compute and plot UMAP colored by a given column.

    Parameters
    ----------
    adata : AnnData
        AnnData with neighbours computed.
    color : str
        Column to color by.
    title : str
        Plot title.
    show : bool
        Whether to show the plot.
    """
    if "X_umap" not in adata.obsm:
        sc.tl.umap(adata)
    sc.pl.umap(adata, color=color, title=title, palette="tab20", show=show)
