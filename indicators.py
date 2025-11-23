#!/usr/bin/env python3
# indicators.py — v1.0
# Contém: Indicadores (EMA, RSI, etc) e Sinais (Cluster, Pine)

import pandas as pd
import numpy as np

# ===================== Funções de Cálculo (Indicadores) =====================

def sma(series: pd.Series, length: int) -> pd.Series:
    return series.rolling(length, min_periods=length).mean()

def ema(series: pd.Series, length: int) -> pd.Series:
    """Calcula a Média Móvel Exponencial (compatível com PineScript 'ema')"""
    return series.ewm(span=length, adjust=False).mean()

def pos_in_range(c: pd.Series, h: pd.Series, l: pd.Series) -> pd.Series:
    rng = (h - l).replace(0.0, np.nan)
    return ((c - l) / rng).fillna(0.5)

def rsi(series: pd.Series, length: int) -> pd.Series:
    delta = series.diff(1)
    gain = delta.where(delta > 0, 0.0).fillna(0)
    loss = -delta.where(delta < 0, 0.0).fillna(0)
    
    avg_gain = gain.ewm(alpha=1/length, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/length, adjust=False).mean()
    
    rs = avg_gain / avg_loss
    rs[avg_loss == 0] = np.inf 
    
    return 100.0 - (100.0 / (1.0 + rs))

# ===================== Funções de Cálculo (Sinais) =====================

def compute_pine_like_signals(df_in: pd.DataFrame, 
                              sma_length: int = 20,
                              pir_threshold: float = 0.85) -> pd.DataFrame:
    """Gera sinais 4H/8H e OS ADICIONA ao DataFrame.
    
    Args:
        sma_length: Comprimento da SMA (default: 20)
        pir_threshold: Threshold de PIR para sinal de compra (default: 0.85)
    """
    df = df_in.copy()
    c, o, h, l = df["close"], df["open"], df["high"], df["low"]

    ma = sma(c, sma_length)
    pir = pos_in_range(c, h, l)

    up_raw = (c > o) & (c > ma) & (pir >= pir_threshold)
    dn_raw = (c < o) & (c < ma)

    df["buy_signal"] = up_raw.shift(1).fillna(False) & (c > o)
    df["sell_signal"] = dn_raw.shift(1).fillna(False) & (c < o)
    return df

def compute_1D_cluster_signals(df_in: pd.DataFrame,
                               sma_length: int = 20,
                               trend_regime_threshold: float = 0.002,
                               trend_regime_tree_threshold: float = 1.5,
                               dist_ma_fast_threshold: float = 0.03,
                               rsi_length: int = 14,
                               rsi_threshold: float = 60,
                               pir_threshold_prev: float = 0.60,
                               pir_threshold_confirm: float = 0.40) -> pd.DataFrame:
    """Gera sinais 1D e OS ADICIONA ao DataFrame.
    
    Args:
        sma_length: Comprimento da SMA (default: 20)
        trend_regime_threshold: Threshold para regime de tendência (default: 0.002 = 0.2%)
        trend_regime_tree_threshold: Threshold para trend_regime_tree (default: 1.5)
        dist_ma_fast_threshold: Threshold de distância da SMA (default: 0.03 = 3%)
        rsi_length: Comprimento do RSI (default: 14)
        rsi_threshold: Threshold do RSI para compra (default: 60)
        pir_threshold_prev: Threshold de PIR no candle anterior (default: 0.60)
        pir_threshold_confirm: Threshold de PIR na confirmação (default: 0.40)
    """
    df = df_in.copy() 
    c, o, h, l = df["close"], df["open"], df["high"], df["low"]

    # --- 1. Calcular todos os features ---
    ma = sma(c, sma_length) 
    
    pct_from_sma = (c / ma) - 1.0
    trend_regime_flag = pd.Series(0.0, index=df.index)
    trend_regime_flag[pct_from_sma > trend_regime_threshold] = 1.0  
    trend_regime_flag[pct_from_sma < -trend_regime_threshold] = -1.0
    trend_regime_tree = trend_regime_flag + 1.0 
    
    above_ma_fast = (c > ma).astype(float) 
    dist_ma_fast = (c / ma - 1.0)
    rsi_val = rsi(c, rsi_length) 
    pir = pos_in_range(c, h, l)
    is_red = c < o
    
    # --- 2. Aplicar regras (para o candle ANTERIOR) ---
    buy_raw = (
        (trend_regime_tree >= trend_regime_tree_threshold) & 
        (above_ma_fast == 1.0) &
        (dist_ma_fast >= dist_ma_fast_threshold) &
        (rsi_val >= rsi_threshold) &
        (pir >= pir_threshold_prev)
    )
    
    sell_raw = (
        (trend_regime_tree <= 0.5) &
        (above_ma_fast == 0.0) &
        (dist_ma_fast <= -0.02) &
        (rsi_val <= 45) &
        (pir <= 0.40) &
        is_red 
    )
    
    # --- 3. Aplicar sinais (lógica de shift(1) + confirmação) ---
    df["buy_signal"] = buy_raw.shift(1).fillna(False) & (c > o) & (pir >= pir_threshold_confirm)
    df["sell_signal"] = sell_raw.shift(1).fillna(False) & (c < o) 
    return df