# utils.py
# 12



from __future__ import annotations

from datetime import datetime, date
from pathlib import Path
from typing import Dict, List, Optional, Set

import json
import pandas as pd


BASE_DIR = Path(__file__).resolve().parent
CACHE_DIR = BASE_DIR / "cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

FAVORITES_PATH = CACHE_DIR / "favorites.json"

# ✅ saved roadmaps persistence
ROADMAPS_PATH = CACHE_DIR / "roadmaps.json"


def load_saved_roadmaps() -> List[Dict]:
    """Load user-saved roadmaps from cache."""
    try:
        if ROADMAPS_PATH.exists():
            data = json.loads(ROADMAPS_PATH.read_text(encoding="utf-8"))
            if isinstance(data, list):
                # ensure dict items only
                out: List[Dict] = []
                for x in data:
                    if isinstance(x, dict):
                        out.append(x)
                return out
    except Exception:
        pass
    return []


def save_saved_roadmaps(items: List[Dict]) -> None:
    """Persist user-saved roadmaps to cache."""
    try:
        ROADMAPS_PATH.write_text(
            json.dumps(items, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    except Exception:
        # keep app running even if file write fails
        pass


# -----------------------------
# Date helpers 
# -----------------------------
def format_date(d: str) -> str:
    """Accepts YYYYMMDD or already-formatted strings."""
    s = str(d or "").strip()
    if len(s) == 8 and s.isdigit():
        try:
            return datetime.strptime(s, "%Y%m%d").strftime("%Y-%m-%d")
        except Exception:
            return s
    return s if s else "-"


def parse_yyyymmdd(d: str) -> Optional[date]:
    s = str(d or "").strip()
    if len(s) == 8 and s.isdigit():
        try:
            return datetime.strptime(s, "%Y%m%d").date()
        except Exception:
            return None
    return None


def calc_dday(end_yyyymmdd: str) -> Optional[int]:
    """Project fixed 'today' for stable demo/reproducibility."""
    dt = parse_yyyymmdd(end_yyyymmdd)
    if not dt:
        return None
    today = datetime.strptime("2026-01-15", "%Y-%m-%d").date() #재현성을 위해 날짜 고정(2026/1/15)
    return (dt - today).days


# -----------------------------
# Favorites (cache)
# -----------------------------
def load_favorites() -> Set[str]:
    try:
        if FAVORITES_PATH.exists():
            data = json.loads(FAVORITES_PATH.read_text(encoding="utf-8"))
            if isinstance(data, list):
                return set(str(x) for x in data)
    except Exception:
        pass
    return set()


def save_favorites(favs: Set[str]) -> None:
    try:
        FAVORITES_PATH.write_text(
            json.dumps(sorted(list(set(str(x) for x in favs))), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    except Exception:
        # keep app running even if file write fails
        pass


# -----------------------------
# CSV loaders
# -----------------------------
def get_certification_data() -> pd.DataFrame:
    """Load certification schedule CSV."""
    try:
        csv_path = BASE_DIR / "integrated_cert_roadmap2_2026.csv"
        return pd.read_csv(csv_path)
    except Exception:
        return pd.DataFrame()


def get_exam_dates_2026(item: str) -> List[date]:
    """Load 2026 exam dates for TOEIC / Korean History from CSV.

    Files:
      - toeic_exam_dates_2026.csv
      - history_exam_dates_2026.csv

    Schema: date (YYYY-MM-DD)
    """
    name = (item or "").strip().upper()
    if name in ("TOEIC", "TOEIC_LR"):
        csv_path = BASE_DIR / "toeic_exam_dates_2026.csv"
    elif name in ("KOREAN_HISTORY", "KOREANHISTORY", "HISTORY"):
        csv_path = BASE_DIR / "history_exam_dates_2026.csv"
    else:
        return []

    try:
        df = pd.read_csv(csv_path)
        if "date" not in df.columns:
            return []
        out: List[date] = []
        for v in df["date"].dropna().astype(str).tolist():
            try:
                out.append(datetime.strptime(v.strip(), "%Y-%m-%d").date())
            except Exception:
                continue
        # keep only 2026 dates + sorted
        out = sorted([d for d in out if d.year == 2026])
        return out
    except Exception:
        return []



# -----------------------------
# Certification filtering
# -----------------------------
def filter_available_certs(df: pd.DataFrame, selected_cert_names: List[str]) -> pd.DataFrame:
    """Return only schedules whose registration end >= fixed today."""
    if df.empty or not selected_cert_names:
        return pd.DataFrame()

    try:
        today = pd.to_datetime("2026-01-15")
        filtered_df = df[df["자격증명"].isin(selected_cert_names)].copy()

        date_cols = [
            "필기접수시작",
            "필기접수종료",
            "필기시험일",
            "최종발표일",
            "실기시험시작",
            "실기시험종료",
        ]
        for col in date_cols:
            if col in filtered_df.columns:
                filtered_df[col] = pd.to_datetime(filtered_df[col], errors="coerce")

        if "필기접수종료" not in filtered_df.columns:
            return pd.DataFrame()

        result_df = filtered_df[
            filtered_df["필기접수종료"].notna() & (filtered_df["필기접수종료"] >= today)
        ].sort_values(by="필기접수시작")
        return result_df
    except Exception:
        return pd.DataFrame()
