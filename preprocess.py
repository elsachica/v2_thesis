#!/usr/bin/env python3
"""
Aggregera lektionsrader till elevnivå för EDM / klustring.

Endast rader med report_status == REPORTED används (UNREPORTED slängs).
Frånvaro för beteendefeatures definieras som is_true_absence: present == 0 och
cause_ext inte i (OTHERACTIVITY, WORKBASEDLEARNING) — sanktionerad närvaro räknas inte som frånvaro.

Elever med färre än min_reported_lessons rapporterade lektioner under året exkluderas.

Features för klustring listas i CLUSTERING_FEATURES (övriga kolumner är metadata eller reserverade).

morning_absence / afternoon_absence: andel lektioner med is_true_absence inom tidsfönster
(morgon: start före 09:00; eftermiddag: start efter 13:00, lokal tid).

trend_score: (VT frånvaroandel - HT frånvaroandel) där andel = sum(true_absence_minutes) / sum(schema_minutes) per termin.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

CLUSTERING_FEATURES = [
    "punctuality_score",
    "morning_absence",
    "afternoon_absence",
    "subject_variance",
    "trend_score",
]

LOCAL_TZ = "Europe/Stockholm"
MORNING_CUTOFF_MIN = 9 * 60
# Eftermiddag: lektioner som startar strikt efter 13:00 lokal tid.
AFTERNOON_CUTOFF_MIN = 13 * 60

SANCTIONED_CAUSES = frozenset({"OTHERACTIVITY", "WORKBASEDLEARNING"})

DEFAULT_MIN_REPORTED_LESSONS = 100


def _minutes_since_midnight_local(series_local: pd.Series) -> pd.Series:
    return (
        series_local.dt.hour.astype("int64") * 60
        + series_local.dt.minute.astype("int64")
        + series_local.dt.second.astype("float64") / 60.0
    )


def _subject_variance_from_rates(rates: pd.Series) -> float:
    if len(rates) <= 1:
        return 0.0
    return float(rates.var(ddof=1))


def build_student_features(
    df: pd.DataFrame,
    min_reported_lessons: int = DEFAULT_MIN_REPORTED_LESSONS,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    id_col = "anon_student_id"
    n_rows_in = len(df)

    if "report_status" not in df.columns:
        raise ValueError("Saknad kolumn: report_status")

    df = df.loc[df["report_status"].astype(str).eq("REPORTED")].copy()
    rows_dropped_unreported = n_rows_in - len(df)
    n_students_after_reported = df[id_col].nunique()

    lesson_counts = df.groupby(id_col).size()
    eligible = lesson_counts[lesson_counts >= min_reported_lessons].index
    n_students_excluded_threshold = int(n_students_after_reported - len(eligible))
    df = df.loc[df[id_col].isin(eligible)].copy()

    if df.empty:
        stats: dict[str, Any] = {
            "rows_dropped_unreported": rows_dropped_unreported,
            "students_after_reported": n_students_after_reported,
            "students_excluded_low_lessons": n_students_excluded_threshold,
            "students_in_output": 0,
            "min_reported_lessons": min_reported_lessons,
        }
        return pd.DataFrame(), stats

    if not pd.api.types.is_datetime64_any_dtype(df["lesson_start"]):
        df["lesson_start"] = pd.to_datetime(df["lesson_start"], utc=True)

    local = df["lesson_start"].dt.tz_convert(LOCAL_TZ)
    mins = _minutes_since_midnight_local(local)
    is_morning = mins < MORNING_CUTOFF_MIN
    is_afternoon = mins > AFTERNOON_CUTOFF_MIN

    df["is_true_absence"] = df["present"].eq(0) & ~df["cause_ext"].isin(
        SANCTIONED_CAUSES
    )
    df["_true_abs"] = df["is_true_absence"]
    df["_late_arrival"] = df["cause_ext"].eq("LATEARRIVAL")

    df["_true_absence_minutes"] = np.where(
        df["is_true_absence"], df["absence_minutes_total"].to_numpy(), 0.0
    )

    ids = df[id_col].unique()

    n_lessons = df.groupby(id_col).size()
    punctuality_score = df.groupby(id_col)["_late_arrival"].sum() / n_lessons

    morning_absence = (
        df.loc[is_morning].groupby(id_col)["_true_abs"].mean().reindex(ids)
    )
    afternoon_absence = (
        df.loc[is_afternoon].groupby(id_col)["_true_abs"].mean().reindex(ids)
    )

    sub_rates = df.groupby([id_col, "subject"], sort=False)["_true_abs"].mean()
    subject_variance = sub_rates.groupby(level=0, sort=False).agg(
        _subject_variance_from_rates
    )

    true_abs_by_term = df.pivot_table(
        index=id_col,
        columns="termin",
        values="_true_absence_minutes",
        aggfunc="sum",
        fill_value=0,
    )
    sched_by_term = df.pivot_table(
        index=id_col,
        columns="termin",
        values="schema_minutes",
        aggfunc="sum",
        fill_value=0,
    )

    def _term_series(pt: pd.DataFrame, term: str) -> pd.Series:
        if term in pt.columns:
            return pt[term].reindex(ids, fill_value=0)
        return pd.Series(0.0, index=ids)

    ht_abs = _term_series(true_abs_by_term, "HT")
    vt_abs = _term_series(true_abs_by_term, "VT")
    ht_sched = _term_series(sched_by_term, "HT")
    vt_sched = _term_series(sched_by_term, "VT")

    rate_ht = np.where(ht_sched > 0, ht_abs / ht_sched, 0.0)
    rate_vt = np.where(vt_sched > 0, vt_abs / vt_sched, 0.0)
    trend_score = pd.Series(rate_vt - rate_ht, index=ids)

    reserved_total = df.groupby(id_col)["absence_minutes_total"].sum()

    at = df["absence_type"].str.upper()
    type_counts = (
        df.assign(_at=at).groupby([id_col, "_at"]).size().unstack(fill_value=0)
    )
    for name in ["NONE", "VALID", "INVALID"]:
        if name not in type_counts.columns:
            type_counts[name] = 0
    reserved_none = type_counts["NONE"]
    reserved_valid = type_counts["VALID"]
    reserved_invalid = type_counts["INVALID"]

    meta = df.groupby(id_col).agg(
        school_name=("school_name", "first"),
        grade=("grade", "first"),
        gender=("gender", "first"),
    )

    out = pd.concat(
        [
            meta,
            punctuality_score.rename("punctuality_score"),
            morning_absence.rename("morning_absence"),
            afternoon_absence.rename("afternoon_absence"),
            subject_variance.rename("subject_variance"),
            trend_score.rename("trend_score"),
            reserved_total.rename("reserved_absence_minutes_total"),
            reserved_none.rename("reserved_absence_type_none"),
            reserved_valid.rename("reserved_absence_type_valid"),
            reserved_invalid.rename("reserved_absence_type_invalid"),
        ],
        axis=1,
    )
    out.index.name = id_col
    out = out.reset_index()

    stats = {
        "rows_dropped_unreported": rows_dropped_unreported,
        "students_after_reported": n_students_after_reported,
        "students_excluded_low_lessons": n_students_excluded_threshold,
        "students_in_output": len(out),
        "min_reported_lessons": min_reported_lessons,
    }
    return out, stats


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Aggregera lektions-parquet till elevnivå (student_features)."
    )
    p.add_argument("--input", required=True, type=Path)
    p.add_argument("--output", type=Path, default=Path("student_features.csv"))
    p.add_argument(
        "--min-reported-lessons",
        type=int,
        default=DEFAULT_MIN_REPORTED_LESSONS,
        help=f"Minsta antal REPORTED-lektioner per elev (default {DEFAULT_MIN_REPORTED_LESSONS}).",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    df = pd.read_parquet(args.input, engine="pyarrow")
    result, stats = build_student_features(
        df, min_reported_lessons=args.min_reported_lessons
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output, index=False)

    n = len(result)
    print()
    print("  +---------------------------+")
    print("  |   preprocess.py klar      |")
    print("  +---------------------------+")
    print("            |")
    print("            v")
    print("     [ Parquet inläst ]")
    print("            |")
    print("            v")
    print("     [ Data cleaning & agg   ]")
    print("            |")
    print("            v")
    print("     [ Sparat CSV ]")
    print("            |")
    print("            v")
    print(f"     Antal elever processade: {n}")
    print(f"     Utdatafil: {args.output.resolve()}")
    print()
    print("  --- Datakvalitet (logg) ---")
    print(f"     Rader borttagna (UNREPORTED): {stats['rows_dropped_unreported']}")
    print(
        f"     Elever exkluderade (< {stats['min_reported_lessons']} rapporterade lektioner): "
        f"{stats['students_excluded_low_lessons']}"
    )
    print(
        f"     Elever kvar efter REPORTED-filter (före tröskel): "
        f"{stats['students_after_reported']}"
    )
    print()


if __name__ == "__main__":
    main()
