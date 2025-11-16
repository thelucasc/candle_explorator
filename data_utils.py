#!/usr/bin/env python3
# data_utils.py — v1.0
# Contém: I/O (load_csv) e Helpers de DataFrame (slice, cut, union)

import pandas as pd
import numpy as np
import warnings
from typing import Optional, Tuple, Dict, List

# Silencia os FutureWarnings do Pandas
warnings.filterwarnings("ignore", category=FutureWarning, module="pandas")

# ===================== Utilities (I/O e Dados) =====================

def ensure_datetime_index(df: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(df.index, pd.DatetimeIndex):
        if "time" not in df.columns:
            raise ValueError("CSV missing 'time' column or DatetimeIndex.")
        try:
            idx = pd.to_datetime(df["time"], unit="s", utc=True, errors="raise")
        except Exception:
            idx = pd.to_datetime(df["time"], utc=True, errors="coerce")
        if idx.isna().any():
            raise ValueError("Failed to convert 'time' to datetime.")
        df = df.copy()
        df.index = idx
    if df.index.tz is not None:
        df.index = df.index.tz_convert(None)
    return df.sort_index()

def load_price_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    cols_lower_map = {c.lower().strip(): c for c in df.columns}
    required = ["time", "open", "high", "low", "close"]
    lower_cols = [c.lower().strip() for c in df.columns]
    missing = [c for c in required if c not in lower_cols]
    if missing:
        raise ValueError(f"CSV missing required columns: {missing}")
    rename = {
        cols_lower_map["time"]: "time",
        cols_lower_map["open"]: "open",
        cols_lower_map["high"]: "high",
        cols_lower_map["low"]: "low",
        cols_lower_map["close"]: "close",
    }
    if "volume" in cols_lower_map:
        rename[cols_lower_map["volume"]] = "volume"
    df = df.rename(columns=rename)
    df = ensure_datetime_index(df)
    cols = ["open", "high", "low", "close"]
    if "volume" in df.columns:
        cols.append("volume")
    return df[cols]

def slice_period(df: pd.DataFrame, days: int) -> pd.DataFrame:
    if df.empty:
        return df
    end = df.index.max()
    start = end - pd.Timedelta(days=days)
    cut = df.loc[(df.index >= start) & (df.index <= end)]
    return cut

def cut_matching(df_tf: Optional[pd.DataFrame], ref: pd.DataFrame) -> Optional[pd.DataFrame]:
    if df_tf is None: return None
    if ref.empty: 
        return pd.DataFrame(columns=df_tf.columns, index=pd.DatetimeIndex([]))
    min_ref, max_ref = ref.index.min(), ref.index.max()
    return df_tf.loc[(df_tf.index >= min_ref) & (df_tf.index <= max_ref)]

def union_indices(indices: List[pd.Index]) -> pd.Index:
    """Cria um índice unificado de múltiplos timeframes."""
    out = pd.Index([])
    for idx in indices:
        if idx is None or len(idx) == 0:
            continue
        out = out.union(idx)
    return out