#!/usr/bin/env python3
"""
K-Means-klustring av elever utifrån beteendefeatures i student_features.csv (EDM).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

# Välj vilka kolumner som ska ingå i klustring (måste finnas i student_features.csv).
# Kommentera bort/ändra rader nedan — endast aktiva rader räknas.
FEATURES = [
    "morning_absence",
    "afternoon_absence",
    "subject_variance",
    "invalid_ratio",
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
        "invalid_ratio",
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


def plot_cluster_demographics(
    df: pd.DataFrame,
    *,
    out_path: Path,
    cluster_col: str = "cluster_id",
    grade_col: str = "grade",
    gender_col: str = "gender",
) -> None:
    """
    Valideringsfigur: visar om beteendearketyper (kluster) är överrepresenterade
    i vissa årskurser eller könskategorier.

    Vi plottar P(cluster | grupp), dvs. normaliserar per grade respektive per gender
    så att varje stapel summerar till 1.0 inom gruppen.
    """
    need = {cluster_col, grade_col, gender_col}
    missing = [c for c in need if c not in df.columns]
    if missing:
        raise ValueError(f"Saknade kolumner för demografiplott: {missing}")

    ct_grade = pd.crosstab(df[grade_col], df[cluster_col], normalize="index").sort_index()
    ct_gender = pd.crosstab(df[gender_col], df[cluster_col], normalize="index").sort_index()

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    ct_grade.plot(kind="bar", stacked=True, ax=axes[0], colormap="tab10")
    axes[0].set_title("Cluster-fördelning per grade (normaliserat)")
    axes[0].set_xlabel("grade")
    axes[0].set_ylabel("andel")
    axes[0].grid(True, axis="y", alpha=0.3)

    ct_gender.plot(kind="bar", stacked=True, ax=axes[1], colormap="tab10")
    axes[1].set_title("Cluster-fördelning per gender (normaliserat)")
    axes[1].set_xlabel("gender")
    axes[1].set_ylabel("andel")
    axes[1].grid(True, axis="y", alpha=0.3)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        title="cluster_id",
        loc="lower center",
        ncol=min(len(labels), 8),
        frameon=False,
    )
    for ax in axes:
        leg = ax.get_legend()
        if leg is not None:
            leg.remove()

    plt.tight_layout(rect=[0, 0.08, 1, 1])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_2d_analysis(
    df: pd.DataFrame,
    x_feat: str,
    y_feat: str,
    labels: np.ndarray,
    *,
    out_path: Path | None = None,
    title: str = "",
    ax: plt.Axes | None = None,
) -> float:
    """
    Scatter av råa x/y, färg = kluster, OLS-linje, Spearman rho i titel.
    Returnerar Spearman rho (NaN om för få giltiga punkter).
    """
    x = df[x_feat].to_numpy(dtype=float)
    y = df[y_feat].to_numpy(dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() >= 2:
        rho = float(pd.Series(x[mask]).corr(pd.Series(y[mask]), method="spearman"))
    else:
        rho = float("nan")

    own_fig = ax is None
    if own_fig:
        plt.figure(figsize=(8, 6))
        ax = plt.gca()

    if len(labels):
        v0, v1 = float(np.min(labels)), float(np.max(labels))
    else:
        v0, v1 = 0.0, 1.0
    sc = ax.scatter(
        x,
        y,
        c=labels,
        cmap="tab10",
        alpha=0.65,
        s=22,
        vmin=v0,
        vmax=v1 if v1 > v0 else v0 + 1,
    )
    if mask.sum() >= 2:
        coef = np.polyfit(x[mask], y[mask], 1)
        x_lo = float(np.nanmin(x[mask]))
        x_hi = float(np.nanmax(x[mask]))
        xs = np.linspace(x_lo, x_hi, 100)
        ax.plot(xs, np.poly1d(coef)(xs), color="crimson", lw=2, zorder=5, label="OLS")
        ax.legend(loc="best", fontsize=8)
    ax.set_xlabel(x_feat)
    ax.set_ylabel(y_feat)
    rho_txt = f"{rho:.4f}" if rho == rho else "n/a"
    ax.set_title(f"{title}\nSpearman rho = {rho_txt}".strip())
    ax.grid(True, alpha=0.3)
    fig = ax.get_figure()
    if fig is not None:
        fig.colorbar(sc, ax=ax, label="cluster_id", fraction=0.046, pad=0.04)

    if own_fig:
        plt.tight_layout()
        if out_path is not None:
            out_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close()

    return rho


def elbow_plot(
    X_scaled: np.ndarray,
    k_max: int,
    out_path: Path,
    random_state: int,
) -> list[float]:
    """Beräkna inertia för k=1..k_max och spara elbow-graf."""
    inertias: list[float] = []
    k_range = range(1, k_max + 1)
    for k in k_range:
        km = KMeans(
            n_clusters=k,
            random_state=random_state,
            n_init=10,
        )
        km.fit(X_scaled)
        inertias.append(float(km.inertia_))

    plt.figure(figsize=(8, 5))
    plt.plot(list(k_range), inertias, marker="o")
    plt.xlabel("k (antal kluster)")
    plt.ylabel("Within-Cluster Sum of Squares (inertia)")
    plt.title("Elbow Method: KMeans inertia vs k")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150)
    plt.close()
    return inertias


def main() -> None:
    p = argparse.ArgumentParser(
        description="KMeans-klustring på student_features.csv"
    )
    p.add_argument(
        "--input",
        type=Path,
        default=Path("student_features.csv"),
        help="Indata-CSV från preprocess.py",
    )
    p.add_argument(
        "--output",
        type=Path,
        default=Path("clustered_students.csv"),
        help="Utdata med cluster_id",
    )
    p.add_argument(
        "--min-lessons",
        type=int,
        default=100,
        help="Minsta antal lektioner (summa reserved_absence_type-rader) för att behålla eleven",
    )
    p.add_argument(
        "--k",
        type=int,
        default=3,
        help="Antal kluster i slutlig KMeans (justera efter elbow_plot.png)",
    )
    p.add_argument(
        "--elbow-max-k",
        type=int,
        default=10,
        help="Största k i elbow-kurvan (1 till detta värde)",
    )
    p.add_argument(
        "--elbow-output",
        type=Path,
        default=Path("elbow_plot.png"),
        help="Var elbow-grafen sparas",
    )
    p.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="För reproducerbarhet (KMeans)",
    )
    p.add_argument(
        "--scatter-output",
        type=Path,
        default=Path("cluster_2d_validation.png"),
        help="När exakt 2 features används: spara 2D scatter + OLS + Spearman (sätt tom för att hoppa över)",
    )
    p.add_argument(
        "--demographics-output",
        type=Path,
        default=Path("cluster_demographics.png"),
        help="Validering: figur cluster_id vs grade/gender",
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

    elbow_plot(
        X_scaled,
        k_max=args.elbow_max_k,
        out_path=args.elbow_output,
        random_state=args.random_state,
    )
    print(f"Sparade elbow-plot: {args.elbow_output.resolve()}")

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

    if len(FEATURES) == 2 and args.scatter_output is not None:
        plot_2d_analysis(
            df,
            FEATURES[0],
            FEATURES[1],
            labels,
            out_path=args.scatter_output,
            title=f"KMeans k={n_clusters} (seed={args.random_state})",
        )
        print(f"Sparade 2D-validering: {args.scatter_output.resolve()}")
    else:
        print("2D-validering: n/a (kräver exakt 2 features)")

    if args.demographics_output is not None:
        plot_cluster_demographics(df, out_path=args.demographics_output)
        print(f"Sparade demografi-validering: {args.demographics_output.resolve()}")

    if n_clusters >= 2 and len(df) > n_clusters:
        sil = silhouette_score(X_scaled, labels, random_state=args.random_state)
        print(f"Silhouette score (k={n_clusters}): {sil:.4f}")
    else:
        print("Silhouette score: n/a (kräver minst 2 kluster och fler än k elever)")

    if "reserved_absence_minutes_total" in df.columns:
        val_table = (
            df.groupby("cluster_id", sort=True)["reserved_absence_minutes_total"]
            .agg(mean_minutes="mean", median_minutes="median", n="count")
            .reset_index()
        )
        print("\nValidering (volym, ej input): reserved_absence_minutes_total per kluster")
        print(val_table.to_string(index=False))

    summary = df.groupby("cluster_id", sort=True)[FEATURES].mean()
    counts = df.groupby("cluster_id").size().rename("n_students")
    summary = summary.join(counts, how="left")
    print("\nKlusterprofiler (medelvärden av features, oskalade):")
    print(summary.round(6).to_string())

    args.output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output, index=False)
    print(f"\nSparade: {args.output.resolve()}  (rader: {len(df)})")
    if len(dropped_zero):
        print(f"Antal rader borttagna (alla features = 0): {len(dropped_zero)}")


if __name__ == "__main__":
    main()
