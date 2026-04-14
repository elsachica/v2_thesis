"""Projektrotsrelativa standardvägar (data/raw, data/processed, output/plots)."""

from __future__ import annotations

from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent

DATA_RAW = _ROOT / "data" / "raw"
DATA_PROCESSED = _ROOT / "data" / "processed"
OUTPUT_PLOTS = _ROOT / "output" / "plots"

DEFAULT_RAW_PARQUET = DATA_RAW / "lyckeboskolan_absence_lasaret2425_v6.parquet"
DEFAULT_STUDENT_FEATURES = DATA_PROCESSED / "student_features.csv"
DEFAULT_CLUSTERED_STUDENTS = DATA_PROCESSED / "clustered_students.parquet"

DEFAULT_ELBOW_PLOT = OUTPUT_PLOTS / "elbow_plot.png"
DEFAULT_CLUSTER_2D = OUTPUT_PLOTS / "cluster_2d_validation.png"
DEFAULT_CLUSTER_DEMOGRAPHICS = OUTPUT_PLOTS / "cluster_demographics.png"
DEFAULT_STABILITY_PCA = OUTPUT_PLOTS / "stability_test_pca.png"
DEFAULT_FEATURE_DISTRIBUTIONS = OUTPUT_PLOTS / "feature_distributions.png"


def project_root() -> Path:
    return _ROOT
