"""Dataset path helpers."""

import os
from pathlib import Path


def get_project_root() -> Path:
    return Path(__file__).resolve().parent.parent


def get_processed_data_dir(project_root: Path | None = None) -> Path:
    override = os.getenv("HGV_PROCESSED_DIR")
    if override:
        return Path(override).expanduser().resolve()
    root = project_root if project_root is not None else get_project_root()
    return root / "data_generation" / "data" / "processed"


def get_raw_trajectories_npz_path(project_root: Path | None = None) -> Path:
    return get_processed_data_dir(project_root) / "raw_hgv_trajectories.npz"
