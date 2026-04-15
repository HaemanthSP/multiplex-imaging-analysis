"""Configuration loading and validation for the pipeline."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class SegmentationConfig:
    """Segmentation stage parameters."""

    image_mpp: float = 0.176
    maxima_threshold: float = 0.075
    maxima_smooth: int = 0
    interior_threshold: float = 0.2
    interior_smooth: int = 2
    small_objects_threshold: int = 15
    fill_holes_threshold: int = 15
    radius: int = 2
    dapi_only: bool = True

    @property
    def postprocess_kwargs(self) -> dict:
        return {
            "maxima_threshold": self.maxima_threshold,
            "maxima_smooth": self.maxima_smooth,
            "interior_threshold": self.interior_threshold,
            "interior_smooth": self.interior_smooth,
            "small_objects_threshold": self.small_objects_threshold,
            "fill_holes_threshold": self.fill_holes_threshold,
            "radius": self.radius,
        }


@dataclass
class ClusteringConfig:
    """Initial clustering parameters."""

    n_neighbors: int = 9
    n_pcs: int = 30
    resolution: float = 1.0
    random_seed: int = 42


@dataclass
class SubclusterGroupConfig:
    """Resolution config for one subclustering group (fine / coarse)."""

    clusters: list[str] = field(default_factory=list)
    resolution_min: float = 0.3
    resolution_max: float = 1.5
    resolution_step: float = 0.5


@dataclass
class SubclusteringConfig:
    """Subclustering parameters."""

    n_neighbors: int = 15
    n_pcs: int = 15
    groups: dict[str, SubclusterGroupConfig] = field(default_factory=dict)


@dataclass
class MachineConfig:
    """Configuration for a single machine (imaging instrument/batch)."""

    name: str
    raw_tiff_dir: str = ""
    feature_csv_dir: str = ""
    roi_list: list[str] = field(default_factory=list)


@dataclass
class ExperimentConfig:
    """Top-level experiment configuration."""

    experiment_name: str = "experiment"
    base_dir: str = "."

    machines: dict[str, MachineConfig] = field(default_factory=dict)

    meta_columns: list[str] = field(
        default_factory=lambda: [
            "Cell Id",
            "Nuc X",
            "Nuc Y Inv",
            "region_num",
            "unique_region",
        ]
    )
    selected_markers: list[str] = field(default_factory=list)

    segmentation: SegmentationConfig = field(default_factory=SegmentationConfig)
    clustering: ClusteringConfig = field(default_factory=ClusteringConfig)
    subclustering: SubclusteringConfig = field(default_factory=SubclusteringConfig)

    # --- Derived paths ---

    @property
    def experiment_dir(self) -> Path:
        return Path(self.base_dir) / self.experiment_name

    def stage_dir(self, stage: int, machine: str | None = None) -> Path:
        """Return the artifact directory for a given stage and optional machine."""
        stage_names = {
            1: "stage1_segmentation",
            2: "stage2_features",
            3: "stage3_clustering",
            4: "stage4_analysis",
            5: "stage5_pooled",
            6: "stage6_subclustering",
        }
        path = self.experiment_dir / stage_names[stage]
        if machine:
            path = path / machine
        return path

    def ensure_stage_dir(self, stage: int, machine: str | None = None) -> Path:
        """Create and return the stage directory."""
        path = self.stage_dir(stage, machine)
        path.mkdir(parents=True, exist_ok=True)
        return path


def load_config(config_path: str | Path) -> ExperimentConfig:
    """Load experiment configuration from a YAML file.

    Parameters
    ----------
    config_path : str or Path
        Path to the YAML configuration file.

    Returns
    -------
    ExperimentConfig
        Parsed configuration object.
    """
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, "r") as f:
        raw = yaml.safe_load(f)

    # Build machine configs
    machines = {}
    for name, mdata in raw.get("machines", {}).items():
        machines[name] = MachineConfig(
            name=name,
            raw_tiff_dir=mdata.get("raw_tiff_dir", ""),
            feature_csv_dir=mdata.get("feature_csv_dir", ""),
            roi_list=mdata.get("roi_list", []),
        )

    # Build sub-configs
    seg_raw = raw.get("segmentation", {})
    seg_config = SegmentationConfig(**{k: v for k, v in seg_raw.items() if k in SegmentationConfig.__dataclass_fields__})

    clust_raw = raw.get("clustering", {})
    clust_config = ClusteringConfig(**{k: v for k, v in clust_raw.items() if k in ClusteringConfig.__dataclass_fields__})

    sub_raw = raw.get("subclustering", {})
    sub_groups: dict[str, SubclusterGroupConfig] = {}
    for gname, gdata in sub_raw.get("groups", {}).items():
        sub_groups[gname] = SubclusterGroupConfig(
            clusters=gdata.get("clusters", []),
            resolution_min=gdata.get("resolution_min", 0.3),
            resolution_max=gdata.get("resolution_max", 1.5),
            resolution_step=gdata.get("resolution_step", 0.5),
        )
    sub_config = SubclusteringConfig(
        n_neighbors=sub_raw.get("n_neighbors", 15),
        n_pcs=sub_raw.get("n_pcs", 15),
        groups=sub_groups,
    )

    default_meta = ["Cell Id", "Nuc X", "Nuc Y Inv", "region_num", "unique_region"]

    return ExperimentConfig(
        experiment_name=raw.get("experiment_name", "experiment"),
        base_dir=raw.get("base_dir", "."),
        machines=machines,
        meta_columns=raw.get("meta_columns", default_meta),
        selected_markers=raw.get("selected_markers", []),
        segmentation=seg_config,
        clustering=clust_config,
        subclustering=sub_config,
    )
