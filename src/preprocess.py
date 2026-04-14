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

from project_paths import DEFAULT_RAW_PARQUET, DEFAULT_STUDENT_FEATURES

CLUSTERING_FEATURES = [
    "punctuality_score",
    "morning_absence",
    "afternoon_absence",
    "subject_variance",
    "fragmentation_index",
    "weekday_variance",
    "trend_score",
]

LOCAL_TZ = "Europe/Stockholm"
MORNING_CUTOFF_MIN = 9 * 60
# Eftermiddag: lektioner som startar strikt efter 13:00 lokal tid.
AFTERNOON_CUTOFF_MIN = 13 * 60

SANCTIONED_CAUSES = frozenset({"OTHERACTIVITY", "WORKBASEDLEARNING"})
NOCAUSE_VALUE = "NOCAUSE"

DEFAULT_MIN_REPORTED_LESSONS = 180
DEFAULT_FULL_DAY_THRESHOLD = 0.9
DEFAULT_FRAGMENTATION_MIN_ABSENCE_DAYS = 3


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


def _safe_ratio(numer: pd.Series, denom: pd.Series) -> pd.Series:
    numer_f = numer.astype("float64")
    denom_f = denom.astype("float64")
    out = pd.Series(0.0, index=numer_f.index)
    mask = denom_f > 0
    out.loc[mask] = (numer_f.loc[mask] / denom_f.loc[mask]).to_numpy(dtype=float)
    return out


def _weekday_variance_from_rates(rates: pd.Series) -> float:
    # Varians över veckodagar; 0 om för få dagar för varians.
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

    df["_date_local"] = local.dt.date
    df["_weekday_local"] = local.dt.weekday.astype("int64")  # 0=Mon..6=Sun

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

    # invalid_ratio: ogiltiga minuter / total absence_minutes_total, klippt till [0, 1].
    # Per rad: undvik dubbelräkning (invalid_absence_minutes + NOCAUSE på samma rad).
    # NaN i invalid_absence_minutes behandlas som 0.
    if "invalid_absence_minutes" not in df.columns:
        raise ValueError("Saknad kolumn: invalid_absence_minutes (krävs för invalid_ratio)")

    inv_abs = pd.to_numeric(df["invalid_absence_minutes"], errors="coerce").fillna(0.0).to_numpy(
        dtype=np.float64
    )
    abs_row = pd.to_numeric(df["absence_minutes_total"], errors="coerce").fillna(0.0).to_numpy(
        dtype=np.float64
    )
    is_nocause_abs = (
        df["present"].eq(0) & df["cause_ext"].astype(str).eq(NOCAUSE_VALUE)
    ).to_numpy()
    # NOCAUSE (frånvaro): hela radens frånvaro räknas som ogiltig; annars kolumnen invalid_absence_minutes.
    row_invalid_like = np.maximum(inv_abs, np.where(is_nocause_abs, abs_row, 0.0))
    row_invalid_like = np.minimum(row_invalid_like, abs_row)
    df["_row_invalid_like"] = row_invalid_like

    invalid_numer = df.groupby(id_col)["_row_invalid_like"].sum().reindex(ids, fill_value=0.0)
    total_abs_min = (
        pd.to_numeric(df["absence_minutes_total"], errors="coerce")
        .fillna(0.0)
        .groupby(df[id_col])
        .sum()
        .reindex(ids, fill_value=0.0)
    )
    denom_ok = total_abs_min.to_numpy(dtype=float) > 0
    ratio_raw = np.zeros(len(ids), dtype=float)
    ratio_raw[denom_ok] = (
        invalid_numer.to_numpy(dtype=float)[denom_ok] / total_abs_min.to_numpy(dtype=float)[denom_ok]
    )
    invalid_ratio = pd.Series(np.clip(ratio_raw, 0.0, 1.0), index=ids, name="invalid_ratio")
    df.drop(columns=["_row_invalid_like"], inplace=True, errors="ignore")

    # fragmentation_index: partial-day vs full-day på dagsnivå (elev+datum).
    day = (
        df.groupby([id_col, "_date_local"], sort=False)
        .agg(day_abs_min=("absence_minutes_total", "sum"), day_sched_min=("schema_minutes", "sum"))
        .reset_index()
    )
    day["day_abs_rate"] = np.where(
        day["day_sched_min"].to_numpy(dtype=float) > 0,
        day["day_abs_min"].to_numpy(dtype=float) / day["day_sched_min"].to_numpy(dtype=float),
        0.0,
    )
    full_day_threshold = DEFAULT_FULL_DAY_THRESHOLD
    day["_is_full_day"] = day["day_abs_rate"] >= full_day_threshold
    day["_is_partial_day"] = (day["day_abs_rate"] > 0) & (day["day_abs_rate"] < full_day_threshold)
    day_counts = (
        day.groupby(id_col, sort=False)
        .agg(
            n_absence_days=("day_abs_rate", lambda s: int(np.sum(np.asarray(s) > 0))),
            n_full_days=("_is_full_day", "sum"),
            n_partial_days=("_is_partial_day", "sum"),
        )
        .reindex(ids, fill_value=0)
    )
    denom_days = (day_counts["n_partial_days"] + day_counts["n_full_days"]).astype("float64")
    raw_fragmentation = _safe_ratio(day_counts["n_partial_days"], denom_days)
    # Edge-case: beräkna endast om minst X frånvarodagar, annars 0 för stabilt KMeans-input.
    fragmentation_index = raw_fragmentation.where(
        day_counts["n_absence_days"] >= DEFAULT_FRAGMENTATION_MIN_ABSENCE_DAYS, 0.0
    ).rename("fragmentation_index")

    # weekday_variance: varians i daglig frånvaroandel över veckodagar (Friday-effect).
    # Vi använder dagsnivå-rate och tar medel per weekday, sedan varians över weekday (0..6).
    day_week = day.merge(
        df[[id_col, "_date_local", "_weekday_local"]].drop_duplicates(),
        on=[id_col, "_date_local"],
        how="left",
        validate="many_to_one",
    )
    weekday_rates = (
        day_week.groupby([id_col, "_weekday_local"], sort=False)["day_abs_rate"]
        .mean()
        .groupby(level=0, sort=False)
        .agg(_weekday_variance_from_rates)
        .reindex(ids, fill_value=0.0)
        .rename("weekday_variance")
    )

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
            invalid_ratio,
            fragmentation_index,
            weekday_rates,
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
    p.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_RAW_PARQUET,
        help=f"Parquet indata (default {DEFAULT_RAW_PARQUET})",
    )
    p.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_STUDENT_FEATURES,
        help="Sparas i data/processed/ (default student_features.csv)",
    )
    p.add_argument(
        "--min-reported-lessons",
        type=int,
        default=DEFAULT_MIN_REPORTED_LESSONS,
        help=f"Minsta antal REPORTED-lektioner per elev (default {DEFAULT_MIN_REPORTED_LESSONS}).",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    inp = args.input.resolve()
    if not inp.is_file():
        raise SystemExit(
            f"Saknas indatafil: {inp}\n\n"
            f"Lägg din .parquet i data/raw/ som {DEFAULT_RAW_PARQUET.name}, "
            "eller kör t.ex.:\n"
            f"  PARQUET=\"/full/sökväg/din_fil.parquet\" ./scripts/run_project.sh\n"
            "eller:\n"
            f"  python3 src/preprocess.py --input /sökväg/din_fil.parquet\n"
        )
    df = pd.read_parquet(inp, engine="pyarrow")
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
