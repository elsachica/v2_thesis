#!/usr/bin/env python3
"""
Stabilitets- och valideringssvit för KMeans på student_features.csv.
Samma rensning/skalning som train_kmeans.py; centroid-alignering till Run 1 via L2 + Hungarian.

Standard: k-sensitivitet (k=3,4,5), fyra seeds; sammanfattningstabell; detaljer + PCA/boxplot för bästa k.
"""

from __future__ import annotations

import argparse
import warnings
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

from project_paths import (
    DEFAULT_FEATURE_DISTRIBUTIONS,
    DEFAULT_STABILITY_PCA,
    DEFAULT_STUDENT_FEATURES,
)
from train_kmeans import FEATURES, load_and_clean, plot_2d_analysis

SEEDS = (10, 20, 30, 40)
SIZE_STD_FRAC = 0.05
DRIFT_MSE_THRESHOLD = 0.05
DEFAULT_K_LIST = (3, 4, 5)


@dataclass
class StabilityRunResult:
    k: int
    silhouettes: list[float]
    mean_silhouette: float
    mse_drift: float
    max_size_std: float
    std_per_cluster: np.ndarray
    aligned_centroids_list: list[np.ndarray]
    aligned_labels_list: list[np.ndarray]


def _validate_full_k_clusters(labels: np.ndarray, k: int, run_label: str) -> None:
    uniq = np.unique(labels)
    if len(uniq) != k or set(int(x) for x in uniq) != set(range(k)):
        msg = (
            f"{run_label}: förväntade {k} icke-tomma kluster (0..{k-1}), "
            f"fick {sorted(uniq.tolist())}. Tomma eller saknade kluster — avbryter."
        )
        warnings.warn(msg, UserWarning)
        raise SystemExit(msg)


def _align_to_reference(
    C_ref: np.ndarray,
    C_run: np.ndarray,
    raw_labels: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    cost = cdist(C_ref, C_run, metric="euclidean")
    row_ind, col_ind = linear_sum_assignment(cost)
    run_to_ref: dict[int, int] = {}
    aligned_centroids = np.zeros_like(C_ref)
    for t in range(len(row_ind)):
        ref_i = int(row_ind[t])
        run_j = int(col_ind[t])
        aligned_centroids[ref_i] = C_run[run_j]
        run_to_ref[run_j] = ref_i
    aligned_labels = np.array([run_to_ref[int(r)] for r in raw_labels], dtype=int)
    return aligned_centroids, aligned_labels


def _print_run_tables(
    run_idx: int,
    seed: int,
    k: int,
    aligned_sizes: pd.Series,
    aligned_centroids_df: pd.DataFrame,
    aligned_centroids_unscaled_df: pd.DataFrame,
    validation: pd.DataFrame,
    silhouette: float,
) -> None:
    print(f"\n{'=' * 60}")
    print(f"Run {run_idx + 1}  |  k={k}  |  random_state={seed}")
    print(f"{'=' * 60}")
    print("\nAlignerade klusterstorlekar (n elever):")
    print(aligned_sizes.to_frame("n").to_string())
    print("\nAlignerade centroids (skalat utrymme):")
    print(aligned_centroids_df.round(6).to_string())
    print("\nAlignerade centroids (originalenheter):")
    print(aligned_centroids_unscaled_df.round(6).to_string())
    print("\nValidering: medel reserved_absence_minutes_total per kluster:")
    print(validation.round(2).to_string())
    print(f"\nSilhouette score: {silhouette:.6f}")


def _avg_distance_to_centroid(
    X_scaled: np.ndarray,
    labels: np.ndarray,
    centroids_scaled: np.ndarray,
    k: int,
) -> np.ndarray:
    out = np.zeros(k, dtype=float)
    for c in range(k):
        mask = labels == c
        if not np.any(mask):
            out[c] = float("nan")
            continue
        d = np.linalg.norm(X_scaled[mask] - centroids_scaled[c], axis=1)
        out[c] = float(np.mean(d))
    return out


def _typical_representatives(
    df: pd.DataFrame,
    X_scaled: np.ndarray,
    labels: np.ndarray,
    centroids_scaled: np.ndarray,
    k: int,
    n_rep: int = 3,
) -> None:
    print(f"\n{'=' * 60}")
    print("Typical Cluster Representatives (närmast centroid i skalat utrymme)")
    print(f"{'=' * 60}")
    for c in range(k):
        mask = labels == c
        idx_all = np.flatnonzero(mask)
        if len(idx_all) == 0:
            print(f"\nKluster {c}: inga elever (hoppar över).")
            continue
        Xc = X_scaled[mask]
        dists = np.linalg.norm(Xc - centroids_scaled[c], axis=1)
        order = np.argsort(dists)[: min(n_rep, len(dists))]
        chosen = idx_all[order]
        print(f"\nKluster {c}")
        sub = df.iloc[chosen][["anon_student_id", *FEATURES]].copy()
        sub["_dist_to_centroid_scaled"] = dists[order]
        print(sub.to_string(index=False))


def _save_feature_boxplots(
    df: pd.DataFrame,
    labels: np.ndarray,
    features: list[str],
    out_path: Path,
    k: int,
) -> None:
    n = len(features)
    ncols = 2
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.5 * ncols, 4 * nrows))
    axes_arr = np.atleast_1d(axes).ravel()
    for i, feat in enumerate(features):
        ax = axes_arr[i]
        data = [df.loc[labels == c, feat].to_numpy() for c in range(k)]
        ax.boxplot(data, tick_labels=[f"C{c}" for c in range(k)])
        ax.set_ylabel(feat)
        ax.set_title(f"{feat} per kluster (k={k})")
        ax.grid(True, axis="y", alpha=0.3)
    for j in range(len(features), len(axes_arr)):
        axes_arr[j].set_visible(False)
    plt.suptitle(f"Featurefördelning per kluster — k={k}", y=1.02)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()


def _output_with_k(path: Path, k: int) -> Path:
    return path.with_name(f"{path.stem}_k{k}{path.suffix}")


def run_stability_for_k(
    k: int,
    df: pd.DataFrame,
    X_scaled: np.ndarray,
    scaler: StandardScaler,
    random_state_extra: int,
    verbose_runs: bool,
) -> StabilityRunResult:
    """Fyra KMeans-körningar med alignering; returnerar mått och listor för bästa-k-diagnostik."""
    aligned_centroids_list: list[np.ndarray] = []
    aligned_labels_list: list[np.ndarray] = []
    silhouettes: list[float] = []

    for run_idx, seed in enumerate(SEEDS):
        km = KMeans(n_clusters=k, random_state=seed, n_init=10)
        raw_labels = km.fit_predict(X_scaled)
        _validate_full_k_clusters(raw_labels, k, f"k={k} Run {run_idx + 1} (seed={seed})")
        C_run = km.cluster_centers_

        if run_idx == 0:
            C_ref = C_run.copy()
            aligned_centroids = C_run.copy()
            aligned_labels = raw_labels.copy()
        else:
            aligned_centroids, aligned_labels = _align_to_reference(
                C_ref, C_run, raw_labels
            )

        aligned_centroids_list.append(aligned_centroids)
        aligned_labels_list.append(aligned_labels)

        aligned_sizes = pd.Series(aligned_labels).value_counts().sort_index()
        aligned_centroids_df = pd.DataFrame(
            aligned_centroids, columns=FEATURES, index=range(k)
        )
        aligned_centroids_unscaled_df = pd.DataFrame(
            scaler.inverse_transform(aligned_centroids),
            columns=FEATURES,
            index=range(k),
        )
        tmp = df.copy()
        tmp["_cid"] = aligned_labels
        validation = (
            tmp.groupby("_cid", sort=True)["reserved_absence_minutes_total"]
            .agg(mean_absence="mean", n="count")
            .reset_index()
            .rename(columns={"_cid": "cluster_id"})
        )
        sil = float(
            silhouette_score(X_scaled, aligned_labels, random_state=random_state_extra)
        )
        silhouettes.append(sil)

        if verbose_runs:
            _print_run_tables(
                run_idx,
                seed,
                k,
                aligned_sizes,
                aligned_centroids_df,
                aligned_centroids_unscaled_df,
                validation,
                sil,
            )

    size_matrix = np.array(
        [
            pd.Series(aligned_labels_list[r]).value_counts().reindex(range(k)).values
            for r in range(len(SEEDS))
        ]
    )
    std_per_cluster = np.std(size_matrix, axis=0, ddof=1)
    max_size_std = float(std_per_cluster.max())

    C_ref_final = aligned_centroids_list[0]
    C_run4_aligned = aligned_centroids_list[3]
    mse_drift = float(np.mean((C_ref_final - C_run4_aligned) ** 2))

    mean_sil = float(np.mean(silhouettes))

    return StabilityRunResult(
        k=k,
        silhouettes=silhouettes,
        mean_silhouette=mean_sil,
        mse_drift=mse_drift,
        max_size_std=max_size_std,
        std_per_cluster=std_per_cluster,
        aligned_centroids_list=aligned_centroids_list,
        aligned_labels_list=aligned_labels_list,
    )


def _pick_best_k(results: list[StabilityRunResult]) -> StabilityRunResult:
    """Högsta medel-silhouette; tie-break: lägre MSE, sedan lägre max_size_std."""
    results_sorted = sorted(
        results,
        key=lambda r: (-r.mean_silhouette, r.mse_drift, r.max_size_std),
    )
    return results_sorted[0]


def _print_best_k_full(
    res: StabilityRunResult,
    df: pd.DataFrame,
    X_scaled: np.ndarray,
    scaler: StandardScaler,
    N: int,
    random_state_extra: int,
    pca_path: Path,
    boxplot_path: Path,
) -> None:
    k = res.k
    labels_final = res.aligned_labels_list[-1]
    centroids_final = res.aligned_centroids_list[-1]

    print(f"\n{'#' * 60}")
    print(f"# Detaljerad diagnostik för valt k = {k} (bästa enligt regel)")
    print(f"{'#' * 60}")

    print(f"\nPer-run silhouette: {[round(s, 6) for s in res.silhouettes]}")
    print(f"\nMeta-analys (k={k}): std av klusterstorlek per alignerat id över 4 körningar")
    for i in range(k):
        print(f"  Kluster {i}: std = {res.std_per_cluster[i]:.4f}")
    print(f"  Max std: {res.max_size_std:.4f}")
    print(f"\nCentroid drift MSE (Run 1 ref vs Run 4 alignerad): {res.mse_drift:.8f}")

    avg_dist = _avg_distance_to_centroid(X_scaled, labels_final, centroids_final, k)
    print(f"\n{'=' * 60}")
    print(f"Cluster density (k={k}, sista körningen seed=40): medelavstånd punkt–centroid (skalat L2)")
    print(f"{'=' * 60}")
    for c in range(k):
        print(f"  Kluster {c}: avg distance = {avg_dist[c]:.6f}")
    _typical_representatives(df, X_scaled, labels_final, centroids_final, k, n_rep=3)

    _save_feature_boxplots(df, labels_final, FEATURES, boxplot_path, k)
    print(f"\nSparade boxplot-figur: {boxplot_path.resolve()}")

    if len(FEATURES) == 2:
        x_feat, y_feat = FEATURES[0], FEATURES[1]
        print(
            f"\n2D-läge (exakt två features): hoppar över PCA — rå scatter {x_feat} vs {y_feat} per seed."
        )
        fig, axes = plt.subplots(2, 2, figsize=(11, 10))
        for ax, run_idx, seed in zip(axes.ravel(), range(len(SEEDS)), SEEDS):
            sil = float(
                silhouette_score(
                    X_scaled,
                    res.aligned_labels_list[run_idx],
                    random_state=random_state_extra,
                )
            )
            plot_2d_analysis(
                df,
                x_feat,
                y_feat,
                res.aligned_labels_list[run_idx],
                out_path=None,
                title=f"Run {run_idx + 1} (seed={seed})  Silhouette={sil:.3f}",
                ax=ax,
            )
        plt.suptitle(
            f"Rå 2D (k={k}) — färg = alignerat kluster-id — OLS + Spearman i panel",
            y=1.02,
        )
        plt.tight_layout()
        pca_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(pca_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"\nSparade 2D-stabilitetsfigur: {pca_path.resolve()}")
    else:
        pca = PCA(n_components=2, random_state=random_state_extra)
        X_pca = pca.fit_transform(X_scaled)
        evr = pca.explained_variance_ratio_
        print(
            f"\nPCA explained variance ratio — PC1: {evr[0]:.6f}, PC2: {evr[1]:.6f}, summa: {evr.sum():.6f}"
        )
        comps = pca.components_
        print("\nPCA loadings (tolkningstöd): top-3 features per axel (|loading| störst)")
        for pc_idx, name in enumerate(["PC1", "PC2"]):
            loadings = np.asarray(comps[pc_idx], dtype=float)
            order = np.argsort(np.abs(loadings))[::-1]
            top = order[: min(3, len(order))]
            parts = [
                f"{FEATURES[int(j)]} (loading={loadings[int(j)]:+.4f})" for j in top
            ]
            print(f"  {name}: " + ", ".join(parts))

        fig, axes = plt.subplots(2, 2, figsize=(10, 9))
        for ax, run_idx, seed in zip(axes.ravel(), range(len(SEEDS)), SEEDS):
            sil = silhouette_score(
                X_scaled,
                res.aligned_labels_list[run_idx],
                random_state=random_state_extra,
            )
            ax.scatter(
                X_pca[:, 0],
                X_pca[:, 1],
                c=res.aligned_labels_list[run_idx],
                cmap="viridis",
                alpha=0.65,
                s=12,
            )
            ax.set_xlabel("PC1")
            ax.set_ylabel("PC2")
            ax.set_title(f"Run {run_idx + 1} (seed={seed})\nSilhouette={sil:.3f}")
            ax.grid(True, alpha=0.3)
        plt.suptitle(
            f"PCA (k={k}) — osynliga beteendemönster i 2D — färg = alignerat kluster-id",
            y=1.02,
        )
        plt.tight_layout()
        pca_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(pca_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"\nSparade PCA-figur: {pca_path.resolve()}")

    threshold_size = SIZE_STD_FRAC * N
    stable_size = res.max_size_std < threshold_size
    stable_drift = res.mse_drift < DRIFT_MSE_THRESHOLD
    print(f"\n{'=' * 60}")
    print(f"Is the model stable? (heuristik, k={k})")
    print(f"{'=' * 60}")
    print(
        f"  Regel: max std(klusterstorlekar) < {SIZE_STD_FRAC} * N  "
        f"(N={N}, tröskel={threshold_size:.2f})"
    )
    print(f"  Max std observerad: {res.max_size_std:.4f}  ->  {'OK' if stable_size else 'VARNING'}")
    print(
        f"  Regel: centroid MSE (ref vs Run4 alignerad) < {DRIFT_MSE_THRESHOLD}  "
        f"->  {'OK' if stable_drift else 'VARNING'}"
    )
    print(f"  MSE observerad: {res.mse_drift:.8f}")
    if stable_size and stable_drift:
        print("\n  Slutsats: Modellen verkar stabil under dessa kriterier.")
    else:
        print(
            "\n  Slutsats: Minst ett stabilitetskriterium är inte uppfyllt; "
            "granska elbow, k eller datarensning."
        )


def main() -> None:
    p = argparse.ArgumentParser(
        description="KMeans stabilitet: k-sensitivitet (3,4,5), fyra seeds, bästa k får PCA/boxplot"
    )
    p.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_STUDENT_FEATURES,
        help="Vanligtvis data/processed/student_features.csv",
    )
    p.add_argument(
        "--min-lessons",
        type=int,
        default=100,
        help="Ska matcha preprocess (min reported lessons; default 100)",
    )
    p.add_argument(
        "--output-figure",
        type=Path,
        default=DEFAULT_STABILITY_PCA,
        help="Basnamn under output/plots/; fil sparas som ..._k{bästa_k}.png",
    )
    p.add_argument(
        "--output-boxplot",
        type=Path,
        default=DEFAULT_FEATURE_DISTRIBUTIONS,
        help="Basnamn under output/plots/; fil sparas som ..._k{bästa_k}.png",
    )
    p.add_argument(
        "--k-list",
        type=str,
        default="3,4,5",
        help="Kommaseparerade k-värden (sensitivitetsanalys)",
    )
    p.add_argument(
        "--verbose-all-k",
        action="store_true",
        help="Skriv full per-run-tabell för varje k (annars endast kompakt + bästa k)",
    )
    p.add_argument("--random-state-extra", type=int, default=42)
    args = p.parse_args()

    k_list = tuple(int(x.strip()) for x in args.k_list.split(",") if x.strip())
    if not k_list:
        raise SystemExit("--k-list är tom.")

    df, _ = load_and_clean(args.input, args.min_lessons)
    if df.empty:
        raise SystemExit("Ingen data efter rensning.")

    X = df[FEATURES].to_numpy(dtype=float)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    N = len(df)

    print("\nPearson-korrelation (linjärt; känslig för outliers):")
    print(df[FEATURES].corr(method="pearson").round(4).to_string())
    print("\nSpearman-korrelation (monotona samband; ofta lämpligare vid skev frånvarodata):")
    print(df[FEATURES].corr(method="spearman").round(4).to_string())

    print(f"\n{'=' * 60}")
    print(f"k-sensitivitet: seeds {SEEDS}, k ∈ {k_list}")
    print(f"{'=' * 60}")

    results: list[StabilityRunResult] = []
    for k in k_list:
        verbose_runs = args.verbose_all_k
        res = run_stability_for_k(
            k, df, X_scaled, scaler, args.random_state_extra, verbose_runs
        )
        results.append(res)
        print(f"\n--- k = {k} (kompakt) ---")
        print(f"  Medel silhouette (4 körningar): {res.mean_silhouette:.6f}")
        print(f"  Centroid drift MSE (Run1 vs Run4 alignerad): {res.mse_drift:.8f}")
        print(f"  Max std (klusterstorlek över körningar): {res.max_size_std:.4f}")

    summary = pd.DataFrame(
        {
            "k": [r.k for r in results],
            "mean_silhouette": [r.mean_silhouette for r in results],
            "mse_drift": [r.mse_drift for r in results],
            "max_size_std": [r.max_size_std for r in results],
        }
    )
    print(f"\n{'=' * 60}")
    print("SAMMANFATTNING: jämförelse av k (högst mean_silhouette + lägst MSE vid lika)")
    print(f"{'=' * 60}")
    print(summary.to_string(index=False))

    best = _pick_best_k(results)
    diag = "2D scatter + boxplot" if len(FEATURES) == 2 else "PCA + boxplot"
    print(
        f"\nValt k för {diag}/detaljer: k = {best.k} "
        f"(mean_silhouette={best.mean_silhouette:.6f}, MSE={best.mse_drift:.8f}, max_size_std={best.max_size_std:.4f})"
    )

    pca_path = _output_with_k(args.output_figure, best.k)
    box_path = _output_with_k(args.output_boxplot, best.k)
    _print_best_k_full(
        best,
        df,
        X_scaled,
        scaler,
        N,
        args.random_state_extra,
        pca_path,
        box_path,
    )


if __name__ == "__main__":
    main()
