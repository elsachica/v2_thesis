#!/usr/bin/env python3
"""
K-Means-klustring av elever utifrån beteendefeatures i student_features.csv (EDM).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from project_paths import (
    DEFAULT_CLUSTERED_STUDENTS,
    DEFAULT_STUDENT_FEATURES,
)

# Välj vilka kolumner som ska ingå i klustring (måste finnas i student_features.csv).
# Kommentera bort/ändra rader nedan — endast aktiva rader räknas.
FEATURES = [
    "morning_absence",
    "afternoon_absence",
    "subject_variance",
    "punctuality_score",
    "trend_score",
    "fragmentation_index",
    "weekday_variance",
]

RESERVED_TYPE_COLS = [
    "reserved_absence_type_none",
    "reserved_absence_type_valid",
    "reserved_absence_type_invalid",
]


def load_and_clean(
    path: Path,
    min_lessons: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Läs CSV, filtrera bort låg-databrukare, imputera NaN, ta bort noll-profiler.

    Returnerar (df_clean, df_dropped_all_zero) för loggning.
    """
    df = pd.read_csv(path)

    missing_feat = [c for c in FEATURES if c not in df.columns]
    if missing_feat:
        raise ValueError(f"Saknade kolumner i CSV: {missing_feat}")

    for c in RESERVED_TYPE_COLS:
        if c not in df.columns:
            raise ValueError(f"Saknad kolumn för lektionsantal: {c}")

    df["_total_lessons"] = df[RESERVED_TYPE_COLS].sum(axis=1)
    df = df[df["_total_lessons"] >= min_lessons].copy()
    df.drop(columns=["_total_lessons"], inplace=True)

    for _rate in ("morning_absence", "afternoon_absence"):
        if _rate in FEATURES:
            df[_rate] = df[_rate].fillna(0)

    for _ratio in (
        "fragmentation_index",
        "weekday_variance",
        "punctuality_score",
        "subject_variance",
        "trend_score",
    ):
        if _ratio in FEATURES and _ratio in df.columns:
            df[_ratio] = df[_ratio].fillna(0)

    all_zero = (df[FEATURES] == 0).all(axis=1)
    dropped = df.loc[all_zero].copy()
    df = df.loc[~all_zero].copy()

    return df, dropped


def main() -> None:
    p = argparse.ArgumentParser(
        description="KMeans-klustring på student_features.csv"
    )
    p.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_STUDENT_FEATURES,
        help="Indata-CSV från preprocess.py (data/processed/)",
    )
    p.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_CLUSTERED_STUDENTS,
        help="Utdata med cluster_id (data/processed/)",
    )
    p.add_argument(
        "--min-lessons",
        type=int,
        default=180,
        help="Minsta antal lektioner (summa reserved_absence_type-rader) för att behålla eleven",
    )
    p.add_argument(
        "--k",
        type=int,
        default=3,
        help="Antal kluster i slutlig KMeans (justera efter elbow_plot.png)",
    )
    p.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="För reproducerbarhet (KMeans)",
    )
    args = p.parse_args()

    df, dropped_zero = load_and_clean(args.input, args.min_lessons)
    if df.empty:
        raise SystemExit("Ingen data kvar efter filtrering; sänk --min-lessons eller kontrollera CSV.")

    X = df[FEATURES].to_numpy(dtype=float)

    # K-Means minimerar kvadratiska avstånd till centroid; features med större
    # spridning/skala dominerar annars avståndet. StandardScaler gör varje
    # dimension medelvärdesnoll och enhetsvarians så att alla beteendedimensioner
    # vägs lika i Euklidiskt avstånd.
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    n_clusters = args.k
    if n_clusters < 1:
        raise SystemExit("--k måste vara minst 1")
    if n_clusters > len(df):
        raise SystemExit(f"--k ({n_clusters}) får inte överstiga antal elever ({len(df)}).")

    final_km = KMeans(
        n_clusters=n_clusters,
        random_state=args.random_state,
        n_init=10,
    )
    labels = final_km.fit_predict(X_scaled)
    df = df.copy()
    df["cluster_id"] = labels

    args.output.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(args.output, index=False, engine="pyarrow")
    print(f"{args.output.resolve()}  (rader: {len(df)})")


if __name__ == "__main__":
    main()
