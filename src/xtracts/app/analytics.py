"""Spending analytics computed from the saved expenses (a DataFrame from db.load_expenses)."""
from __future__ import annotations

import pandas as pd


def _dated(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    return out.dropna(subset=["date"])


def kpis(df: pd.DataFrame) -> dict:
    n = len(df)
    total = float(df["total"].sum()) if n else 0.0
    return {
        "total_spent": total,
        "transactions": n,
        "average": total / n if n else 0.0,
        "total_tax": float(df["tax"].fillna(0).sum()) if n else 0.0,
        "merchants": int(df["merchant"].str.lower().nunique()) if n else 0,
    }


def daily(df: pd.DataFrame) -> pd.Series:
    d = _dated(df)
    return d.groupby(d["date"].dt.normalize())["total"].sum().sort_index()


def monthly(df: pd.DataFrame) -> pd.Series:
    d = _dated(df)
    s = d.groupby(d["date"].dt.to_period("M"))["total"].sum().sort_index()
    s.index = s.index.astype(str)
    return s


def by_merchant(df: pd.DataFrame, top: int = 10) -> pd.DataFrame:
    """Total spend and number of visits per merchant (case-insensitive), biggest first."""
    if df.empty:
        return pd.DataFrame(columns=["merchant", "total", "visits"])
    g = df.assign(key=df["merchant"].str.strip().str.lower()).groupby("key")
    out = pd.DataFrame({"merchant": g["merchant"].first(), "total": g["total"].sum(), "visits": g["total"].size()})
    return out.sort_values("total", ascending=False).head(top).reset_index(drop=True)
