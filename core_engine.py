#!/usr/bin/env python3
# core_engine.py — v1.7-tbrs (Refatorado v2.0)
# Contém: Motor Numba, Preparador de Dados NumPy e Workers

import warnings
from typing import Optional, Dict, List, Tuple

import pandas as pd
import numpy as np

# (*** NOVO: Importa utilitários e o motor Numba isolado ***)
import data_utils as du
from numba_engine import _run_numba_loop

# ==================== (v5.15) Helper de Preparação NumPy ====================
def prepare_numpy_data(
    df1d: pd.DataFrame, 
    df4h: Optional[pd.DataFrame], 
    df8h: Optional[pd.DataFrame],
    ema_len_list: List[int]
) -> Dict:
    # (*** LÓGICA MODIFICADA: Captura índices primeiro ***)
    idx_1d = df1d.index if df1d is not None else pd.Index([])
    idx_4h = df4h.index if df4h is not None else pd.Index([])
    idx_8h = df8h.index if df8h is not None else pd.Index([])
    
    # (*** USA data_utils ***)
    all_idx = du.union_indices([idx_1d, idx_4h, idx_8h])
    all_idx = all_idx.sort_values()

    if all_idx.empty:
        return { 'empty': True, 'all_idx': all_idx }

    # --- 1. DataFrame de Eventos (Sinais) ---
    ev = pd.DataFrame(index=all_idx)
    ev["buy_4h"]  = df4h["buy_signal"].reindex(all_idx, fill_value=False) if df4h is not None and "buy_signal" in df4h.columns else False
    ev["sell_4h"] = df4h["sell_signal"].reindex(all_idx, fill_value=False) if df4h is not None and "sell_signal" in df4h.columns else False
    ev["buy_8h"]  = df8h["buy_signal"].reindex(all_idx, fill_value=False) if df8h is not None and "buy_signal" in df8h.columns else False
    ev["sell_8h"] = df8h["sell_signal"].reindex(all_idx, fill_value=False) if df8h is not None and "sell_signal" in df8h.columns else False
    ev["buy_1d"]  = df1d["buy_signal"].reindex(all_idx, fill_value=False) if "buy_signal" in df1d.columns else False
    ev["sell_1d"] = df1d["sell_signal"].reindex(all_idx, fill_value=False) if "sell_signal" in df1d.columns else False
    
    # --- 2. Mapas de Preços (Close e Low) ---
    price_series_list: List[pd.Series] = []
    low_series_list: List[pd.Series] = [] # (*** MUDANÇA v1.9: TBRS ***)
    
    if df4h is not None: 
        price_series_list.append(df4h['close'])
        low_series_list.append(df4h['low']) # (*** MUDANÇA v1.9: TBRS ***)
    if df8h is not None: 
        price_series_list.append(df8h['close'])
        low_series_list.append(df8h['low']) # (*** MUDANÇA v1.9: TBRS ***)
    if df1d is not None: 
        price_series_list.append(df1d['close'])
        low_series_list.append(df1d['low']) # (*** MUDANÇA v1.9: TBRS ***)
    
    if not price_series_list:
        prices_np = np.zeros(len(all_idx), dtype=np.float64)
        lows_np = np.zeros(len(all_idx), dtype=np.float64) # (*** MUDANÇA v1.9: TBRS ***)
    else:
        prices_s = pd.concat(price_series_list).groupby(level=0).last()
        prices_np = prices_s.reindex(all_idx).to_numpy(dtype=np.float64, na_value=0.0)
        
        # (*** MUDANÇA v1.9: TBRS - Processa Lows ***)
        lows_s = pd.concat(low_series_list).groupby(level=0).last()
        lows_np = lows_s.reindex(all_idx).to_numpy(dtype=np.float64, na_value=0.0)
        # (*** FIM DA MUDANÇA v1.9 ***)


    # --- 3. Mapas de EMA ---
    ema_arrays_np: Dict[int, np.ndarray] = {}
    
    for length in ema_len_list:
        col_name = f'ema_{length}'
        ema_series_list: List[pd.Series] = []
        
        if df4h is not None and col_name in df4h.columns: ema_series_list.append(df4h[col_name])
        if df8h is not None and col_name in df8h.columns: ema_series_list.append(df8h[col_name])
        if df1d is not None and col_name in df1d.columns: ema_series_list.append(df1d[col_name])
        
        if not ema_series_list:
            ema_arrays_np[length] = np.zeros(len(all_idx), dtype=np.float64)
        else:
            emas_s = pd.concat(ema_series_list).groupby(level=0).last()
            ema_np = emas_s.reindex(all_idx).to_numpy(dtype=np.float64, na_value=0.0)
            ema_arrays_np[length] = ema_np

    is_1d_candle_np = all_idx.isin(idx_1d)
    is_4h_candle_np = all_idx.isin(idx_4h)
    is_8h_candle_np = all_idx.isin(idx_8h)

    # --- 4. Retornar dicionário de arrays NumPy ---
    return {
        'empty': False,
        'all_idx': all_idx,
        'prices_np': prices_np,
        'lows_np': lows_np, # (*** MUDANÇA v1.9: TBRS ***)
        'ema_arrays_np': ema_arrays_np,
        'buy_4h_np': ev["buy_4h"].to_numpy(dtype=np.bool_),
        'sell_4h_np': ev["sell_4h"].to_numpy(dtype=np.bool_),
        'buy_8h_np': ev["buy_8h"].to_numpy(dtype=np.bool_),
        'sell_8h_np': ev["sell_8h"].to_numpy(dtype=np.bool_),
        'buy_1d_np': ev["buy_1d"].to_numpy(dtype=np.bool_),
        'sell_1d_np': ev["sell_1d"].to_numpy(dtype=np.bool_),
        
        'is_1d_candle_np': is_1d_candle_np,
        'is_4h_candle_np': is_4h_candle_np,
        'is_8h_candle_np': is_8h_candle_np,
    }

# ==================== Worker Setup (para Multiprocessing) ====================
g_data = {}

def init_worker(data_payload):
    global g_data
    g_data.update(data_payload)
    warnings.filterwarnings("ignore", category=FutureWarning)


def run_one_combo(combo):
    # (*** MUDANÇA v1.7: Unpack de 16 params ***)
    b4, s4, b8, s8, b1, s1, \
    sl_up, sl_up_amt, sl_down, sl_down_amt, \
    tp, tp_after, tp_sell, \
    tp_ema_pct, tp_ema_amt, ema_len = combo
    
    common_params = g_data['common_params']
    
    date_list = g_data['date_list']
    numpy_data_slices = g_data['numpy_data_slices']
    
    # --- 1. Preparar parâmetros (floats) ---
    initial_capital = common_params['initial_capital']
    commission_bps = common_params['commission_bps']
    max_exposure_pct = common_params['max_exposure_pct']
    
    use_sl_on_signal_price = common_params.get('stop_loss_signal', False)
    
    # (*** MUDANÇA v1.9: TBRS ***)
    two_bar_reversal_stop = common_params.get('two_bar_reversal_stop', False)
    
    stops_on_candle = common_params.get('stops_on_candle', ['1D', '4H', '8H'])
    run_stops_on_1d = "1D" in stops_on_candle
    run_stops_on_4h = "4H" in stops_on_candle
    run_stops_on_8h = "8H" in stops_on_candle
    
    sl_up_pct_val = sl_up if sl_up is not None else 0.0
    sl_up_sell_amount_val = sl_up_amt if sl_up_amt is not None else 100.0
    sl_down_pct_val = sl_down if sl_down is not None else 0.0
    sl_down_sell_amount_val = sl_down_amt if sl_down_amt is not None else 100.0
    
    tp_pct_val = tp if tp is not None else 0.0
    
    tp_after_pct_val = tp_after if tp_after is not None else 0.0
    tp_sell_pct_val = tp_sell if tp_sell is not None else 0.0
    
    tp_ema_pct_val = tp_ema_pct if tp_ema_pct is not None else 0.0
    tp_ema_sell_val = tp_ema_amt if tp_ema_amt is not None else 0.0

    default_ema_len = g_data['default_ema_len']

    results_by_date = {}

    for date_label in date_list:
        np_data = numpy_data_slices.get(date_label) 
        
        if np_data is None or np_data['empty']:
            # (*** MUDANÇA v1.9: 23 métricas ***)
            results_by_date[date_label] = (0.0, (0,) * 23)
            continue 

        ema_np = np_data['ema_arrays_np'].get(ema_len, np_data['ema_arrays_np'][default_ema_len])
        equity_out_np = np.empty_like(np_data['prices_np'], dtype=np.float64)

        # (*** CHAMA O MOTOR IMPORTADO ***)
        metrics_custom = _run_numba_loop(
            np_data['prices_np'], 
            np_data['lows_np'], # (*** MUDANÇA v1.9: TBRS ***)
            ema_np, 
            np_data['buy_4h_np'], np_data['sell_4h_np'],
            np_data['buy_8h_np'], np_data['sell_8h_np'],
            np_data['buy_1d_np'], np_data['sell_1d_np'],
            
            np_data['is_1d_candle_np'],
            np_data['is_4h_candle_np'],
            np_data['is_8h_candle_np'],
            
            b4, s4, b8, s8, b1, s1,
            
            sl_up_pct_val, sl_up_sell_amount_val,
            sl_down_pct_val, sl_down_sell_amount_val,
            
            tp_pct_val, 
            
            tp_after_pct_val, tp_sell_pct_val,
            
            tp_ema_pct_val, tp_ema_sell_val, 
            commission_bps, initial_capital, max_exposure_pct, 
            use_sl_on_signal_price, 
            
            run_stops_on_1d,
            run_stops_on_4h,
            run_stops_on_8h,
            
            two_bar_reversal_stop, # (*** MUDANÇA v1.9: TBRS ***)
            
            equity_out_np 
        )
        
        final_equity = equity_out_np[-1] if len(equity_out_np) > 0 else initial_capital
        pct_custom = (final_equity / initial_capital - 1.0) * 100.0
        
        results_by_date[date_label] = (pct_custom, metrics_custom)
    
    return ("multi-date", (*combo, results_by_date))