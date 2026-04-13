#!/usr/bin/env bash
# Kör hela pipelinen från projektroten: rådata -> bearbetad CSV -> KMeans -> stabilitet.
# Kräver: data/raw/lyckeboskolan_absence_lasaret2425_v6.parquet (eller sätt PARQUET)
set -e

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

export MPLBACKEND="${MPLBACKEND:-Agg}"

MIN_LESSONS="${MIN_LESSONS:-180}"
K="${K:-3}"

mkdir -p data/raw data/processed output/plots

PRE_ARGS=(--min-reported-lessons "$MIN_LESSONS")
if [[ -n "${PARQUET:-}" ]]; then
  PRE_ARGS+=(--input "$PARQUET")
fi

echo "==> 1/3 Preprocess (rådata -> data/processed/student_features.csv)"
python3 src/preprocess.py "${PRE_ARGS[@]}"

echo "==> 2/3 Train KMeans (k=${K}) -> data/processed/clustered_students.csv"
python3 src/train_kmeans.py --k "$K" --min-lessons "$MIN_LESSONS"

echo "==> 3/3 Stability (--k-list ${K}) -> output/plots/"
python3 src/test_kmeans_stability.py --min-lessons "$MIN_LESSONS" --k-list "$K"

echo "Klart. Artefakter: data/processed/*.csv, output/plots/*.png"
