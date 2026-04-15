"""I/O utilities: reading/writing artifacts between stages."""

from __future__ import annotations

import os
import re
import sys
from pathlib import Path
from typing import Optional

import anndata as ad
import pandas as pd


# ---------------------------------------------------------------------------
# Feature file loading
# ---------------------------------------------------------------------------


def combine_feature_files(
    input_dir: str | Path,
    feature_files: list[str],
    region_list: list[str] | None = None,
) -> pd.DataFrame:
    """Read and concatenate per-ROI feature CSVs.

    Parameters
    ----------
    input_dir : str or Path
        Base directory containing CSVs.
    feature_files : list[str]
        Relative paths (within *input_dir*) to each ROI CSV.
    region_list : list[str], optional
        Human-readable region aliases (must match length of *feature_files*).

    Returns
    -------
    pd.DataFrame
        Combined dataframe with ``region_num`` and ``unique_region`` columns
        added and DAPI autofluorescence columns removed.
    """
    if region_list is not None and len(region_list) != len(feature_files):
        raise ValueError("Total number of feature files and region aliases must match")

    def _is_dapi_autofluo(col: str) -> bool:
        return bool(re.search(r"DAPI C\d+ Biomarker Exp", col))

    frames: list[pd.DataFrame] = []
    for i, fname in enumerate(feature_files):
        tmp = pd.read_csv(os.path.join(input_dir, fname))
        tmp["region_num"] = str(i)
        if region_list is not None:
            tmp["unique_region"] = str(region_list[i])
        frames.append(tmp)

    df = pd.concat(frames, ignore_index=True)

    # Drop DAPI autofluorescence columns
    invalid_cols = [c for c in df.columns if _is_dapi_autofluo(c)]
    if invalid_cols:
        print(f"Dropping {len(invalid_cols)} DAPI autofluorescence columns")
        df = df.drop(columns=invalid_cols)

    # Report any NaN columns
    nan_cols = df.columns[df.isna().any()].tolist()
    if nan_cols:
        print(f"Warning: columns with NaN values: {nan_cols}")

    return df


def discover_feature_files(input_dir: str | Path) -> tuple[list[str], list[str]]:
    """Walk a directory and return (feature_files, region_list).

    Parameters
    ----------
    input_dir : str or Path
        Directory to search for CSV files.

    Returns
    -------
    tuple[list[str], list[str]]
        ``(feature_files, region_list)`` – relative paths and stem names.
    """
    feature_files: list[str] = []
    region_list: list[str] = []
    for path, _subdirs, files in os.walk(input_dir):
        for name in sorted(files):
            fname, ext = os.path.splitext(name)
            if ext.lower() == ".csv":
                rel_path = os.path.relpath(path, input_dir)
                feature_files.append(os.path.join(rel_path, name))
                region_list.append(os.path.join(rel_path, fname))
    return feature_files, region_list


# ---------------------------------------------------------------------------
# AnnData I/O
# ---------------------------------------------------------------------------


def save_adata(adata: ad.AnnData, path: str | Path) -> None:
    """Save an AnnData object to ``.h5ad``."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    adata.write_h5ad(path)
    print(f"Saved AnnData → {path}")


def load_adata(path: str | Path) -> ad.AnnData:
    """Load an AnnData object from ``.h5ad``."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"AnnData file not found: {path}")
    return ad.read_h5ad(path)


# ---------------------------------------------------------------------------
# Cluster labels I/O
# ---------------------------------------------------------------------------


def save_cluster_labels(labels_df: pd.DataFrame, path: str | Path) -> None:
    """Save cluster labels CSV (cluster_id, label, group).

    Parameters
    ----------
    labels_df : pd.DataFrame
        Must contain columns: ``cluster_id``, ``label``, ``group``.
    path : str or Path
        Output CSV path.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    required = {"cluster_id", "label", "group"}
    missing = required - set(labels_df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    labels_df.to_csv(path, index=False)
    print(f"Saved cluster labels → {path}")


def load_cluster_labels(path: str | Path) -> pd.DataFrame:
    """Load cluster labels CSV.

    Returns
    -------
    pd.DataFrame
        With columns ``cluster_id``, ``label``, ``group``.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Cluster labels file not found: {path}")
    df = pd.read_csv(path)
    required = {"cluster_id", "label", "group"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Cluster labels file missing columns: {missing}")
    df["cluster_id"] = df["cluster_id"].astype(str)
    return df


# ---------------------------------------------------------------------------
# Per-ROI cluster assignment output
# ---------------------------------------------------------------------------


def save_roi_assignments(
    adata: ad.AnnData,
    output_dir: str | Path,
    region_col: str = "unique_region",
    extra_cols: list[str] | None = None,
) -> None:
    """Split AnnData by region and save per-ROI cluster assignment CSVs.

    Parameters
    ----------
    adata : ad.AnnData
        Annotated data with cluster assignments in ``.obs``.
    output_dir : str or Path
        Directory to write per-ROI CSVs.
    region_col : str
        Column in ``adata.obs`` identifying the ROI/region.
    extra_cols : list[str], optional
        Additional ``.obs`` columns to include in output.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for region in adata.obs[region_col].unique():
        mask = adata.obs[region_col] == region
        region_data = adata.obs.loc[mask].copy()
        if extra_cols:
            for col in extra_cols:
                if col in adata.obs.columns and col not in region_data.columns:
                    region_data[col] = adata.obs.loc[mask, col]

        # Sanitise string columns for naive CSV parsers (e.g. MACSiQview):
        # replace commas (break column alignment) and spaces (may act as
        # delimiters or truncate labels) with safe alternatives.
        str_cols = region_data.select_dtypes(include="object").columns
        for col in str_cols:
            region_data[col] = (
                region_data[col]
                .str.replace(",", ";", regex=False)
                .str.replace(" ", "_", regex=False)
            )

        safe_name = str(region).replace("/", "_").replace("\\", "_")
        outpath = output_dir / f"{safe_name}_cluster_assignments.csv"
        region_data.to_csv(outpath, index=False)

    print(f"Saved {len(adata.obs[region_col].unique())} per-ROI assignment files → {output_dir}")


# ---------------------------------------------------------------------------
# Stage validation helpers
# ---------------------------------------------------------------------------


def validate_stage_inputs(required_paths: list[Path], stage_name: str) -> None:
    """Check that all required input files/dirs exist.

    Raises
    ------
    FileNotFoundError
        If any required path is missing.
    """
    missing = [p for p in required_paths if not p.exists()]
    if missing:
        msg = f"[{stage_name}] Missing required inputs:\n"
        msg += "\n".join(f"  - {p}" for p in missing)
        raise FileNotFoundError(msg)
    print(f"[{stage_name}] All required inputs found ✓")
