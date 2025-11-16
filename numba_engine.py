#!/usr/bin/env python3
# numba_engine.py — v1.0
# Contém APENAS o loop Numba de alta performance (o "cofre").

import numba
import numpy as np
from typing import Tuple

# ===================== O MOTOR NUMBA (v5.30 - TP-Aft + TBRS) =====================
@numba.jit(nopython=True, fastmath=True)
def _run_numba_loop(
    # Arrays (dados)
    prices: np.ndarray,
    lows: np.ndarray, # (*** MUDANÇA v1.9: TBRS ***)
    ema_values: np.ndarray,    
    buy_4h_sig: np.ndarray,   
    sell_4h_sig: np.ndarray,   
    buy_8h_sig: np.ndarray,   
    sell_8h_sig: np.ndarray,   
    buy_1d_sig: np.ndarray,     
    sell_1d_sig: np.ndarray,    
    
    is_1d_candle_np: np.ndarray,
    is_4h_candle_np: np.ndarray,
    is_8h_candle_np: np.ndarray,
    
    # Params (floats)
    buy4h_pct: float,
    sell4h_pct: float,
    buy8h_pct: float,
    sell8h_pct: float,
    buy1d_pct: float,
    sell1d_pct: float,
    
    sl_up_pct_val: float,
    sl_up_sell_amount_val: float,
    sl_down_pct_val: float,
    sl_down_sell_amount_val: float,
    
    tp_pct_val: float,
    
    tp_after_pct_val: float,
    tp_sell_pct_val: float,
    
    tp_ema_pct_trigger_val: float,
    tp_ema_sell_amount_val: float,
    commission_bps: float,
    initial_capital: float,
    max_exposure_pct: float,
    
    # Flags de Lógica
    use_sl_on_signal_price: bool,
    run_stops_on_1d: bool,
    run_stops_on_4h: bool,
    run_stops_on_8h: bool,
    two_bar_reversal_stop: bool, # (*** MUDANÇA v1.9: TBRS ***)
    
    # Output array (para ser preenchido)
    equity_out: np.ndarray 
) -> Tuple[int, int, int, int, int, int, int, int, int, int, int, int, int, int, int, int, int, int, int, int, int, int, int]: # <--- MUDANÇA v1.9 (23 métricas) 
    """
    Este é o loop principal, otimizado pelo Numba.
    (v5.30) Retorna uma tupla de 23 contadores de transações.
    """
    
    # --- Setup (sem dataclasses) ---
    cash = initial_capital
    qty = 0.0
    avg_cost = 0.0
    last_purchase_price = 0.0 
    
    last_signal_price = 0.0
    
    tp_after_base_price = 0.0
    
    # (*** MUDANÇA v1.9: TBRS ***)
    tbrs_4h_low = 0.0
    tbrs_8h_low = 0.0
    tbrs_1d_low = 0.0
    sell_actions_tbrs = 0
    # (*** FIM DA MUDANÇA v1.9 ***)
    
    sell_actions_sl = 0
    sell_actions_tp_fixed = 0
    sell_actions_tp_after = 0 
    sell_actions_tp_ema = 0
    
    buy_actions_4h = 0; buy_actions_8h = 0; buy_actions_1d = 0
    sell_actions_4h = 0; sell_actions_8h = 0; sell_actions_1d = 0
    buy_signals_4h = 0; buy_signals_8h = 0; buy_signals_1d = 0
    sell_signals_4h = 0; sell_signals_8h = 0; sell_signals_1d = 0
    buy_ignored_4h = 0; buy_ignored_8h = 0; buy_ignored_1d = 0
    sell_ignored_4h = 0; sell_ignored_8h = 0; sell_ignored_1d = 0
    
    max_exposure = max_exposure_pct / 100.0
    commission_rate = commission_bps / 10000.0
    
    use_sl_up = sl_up_pct_val > 0.0
    sl_up_pct = -abs(sl_up_pct_val)
    use_sl_up_partial = sl_up_sell_amount_val > 0.0 and sl_up_sell_amount_val < 100.0
    sl_up_sell_mult = sl_up_sell_amount_val / 100.0
    
    use_sl_down = sl_down_pct_val > 0.0
    sl_down_pct = -abs(sl_down_pct_val)
    use_sl_down_partial = sl_down_sell_amount_val > 0.0 and sl_down_sell_amount_val < 100.0
    sl_down_sell_mult = sl_down_sell_amount_val / 100.0
    
    use_tp = tp_pct_val > 0.0
    tp_pct = abs(tp_pct_val)
    
    use_tp_after = tp_after_pct_val > 0.0 and tp_sell_pct_val > 0.0
    tp_after_trigger_mult = 1.0 + (tp_after_pct_val / 100.0)
    tp_after_sell_mult = tp_sell_pct_val / 100.0
    
    use_tp_ema = tp_ema_pct_trigger_val > 0.0 and tp_ema_sell_amount_val > 0.0
    tp_ema_trigger_mult = 1.0 + (tp_ema_pct_trigger_val / 100.0)
    tp_ema_sell_mult = tp_ema_sell_amount_val / 100.0
    
    use_tbrs = two_bar_reversal_stop # (*** MUDANÇA v1.9: TBRS ***)
    
    n = len(prices)
    for i in range(n):
        px = prices[i]
        low = lows[i] # (*** MUDANÇA v1.9: TBRS ***)
        ema_val = ema_values[i] 
        
        if px <= 0.0:
            if i > 0:
                equity_out[i] = equity_out[i-1] 
            else:
                equity_out[i] = initial_capital
            continue

        # --- 0) Checar Sinais de Compra (para base do SL e contagem) ---
        is_buy_4h = buy_4h_sig[i] and buy4h_pct > 0.0
        is_buy_8h = buy_8h_sig[i] and buy8h_pct > 0.0
        is_buy_1d = buy_1d_sig[i] and buy1d_pct > 0.0
        
        is_any_buy_signal = is_buy_4h or is_buy_8h or is_buy_1d
        
        if is_any_buy_signal:
            last_signal_price = px

        if is_buy_4h: buy_signals_4h += 1
        if is_buy_8h: buy_signals_8h += 1
        if is_buy_1d: buy_signals_1d += 1
        
        run_stops_this_candle = (run_stops_on_1d and is_1d_candle_np[i]) or \
                                (run_stops_on_4h and is_4h_candle_np[i]) or \
                                (run_stops_on_8h and is_8h_candle_np[i])


        # --- (*** MUDANÇA v1.9: NOVO STOP TBRS - Prioridade Máxima ***) ---
        if qty > 0.0 and use_tbrs:
            
            # Checa 4H TBRS (só roda no candle 4H)
            if tbrs_4h_low > 0.0 and is_4h_candle_np[i]:
                if px < tbrs_4h_low:
                    # Vende tudo
                    gross = qty * px
                    net = gross - (gross * commission_rate)
                    cash += net
                    qty = 0.0
                    avg_cost = 0.0; last_purchase_price = 0.0
                    last_signal_price = 0.0; tp_after_base_price = 0.0
                    sell_actions_tbrs += 1
                
                tbrs_4h_low = 0.0 # Desarma (só checa uma vez)

            # Checa 8H TBRS (só roda no candle 8H)
            if tbrs_8h_low > 0.0 and is_8h_candle_np[i]:
                if px < tbrs_8h_low and qty > 0.0: # Checa qty de novo (4H pode ter vendido)
                    # Vende tudo
                    gross = qty * px
                    net = gross - (gross * commission_rate)
                    cash += net
                    qty = 0.0
                    avg_cost = 0.0; last_purchase_price = 0.0
                    last_signal_price = 0.0; tp_after_base_price = 0.0
                    sell_actions_tbrs += 1
                
                tbrs_8h_low = 0.0 # Desarma

            # Checa 1D TBRS (só roda no candle 1D)
            if tbrs_1d_low > 0.0 and is_1d_candle_np[i]:
                if px < tbrs_1d_low and qty > 0.0: # Checa qty de novo
                    # Vende tudo
                    gross = qty * px
                    net = gross - (gross * commission_rate)
                    cash += net
                    qty = 0.0
                    avg_cost = 0.0; last_purchase_price = 0.0
                    last_signal_price = 0.0; tp_after_base_price = 0.0
                    sell_actions_tbrs += 1
                
                tbrs_1d_low = 0.0 # Desarma
        # --- (*** FIM DA MUDANÇA v1.9 ***) ---


        # --- 1) Checagens de Saída (SL, TP Fixo, TP EMA) ---
        if qty > 0.0 and run_stops_this_candle: 
            
            # 1a. Stop-loss (Prioridade 2, depois do TBRS)
            sl_base_price = 0.0
            if use_sl_on_signal_price:
                sl_base_price = last_signal_price
            else:
                sl_base_price = last_purchase_price
                
            is_above_ema = px > ema_val and ema_val > 0.0
            
            if sl_base_price > 0.0:
                if is_above_ema:
                    # --- LÓGICA SL "UP" (Acima da EMA) ---
                    if use_sl_up:
                        sl_price = sl_base_price * (1.0 + sl_up_pct / 100.0)
                        if px <= sl_price:
                            sell_actions_sl += 1
                            if use_sl_up_partial:
                                sell_qty_sl = min(qty, qty * sl_up_sell_mult)
                                if sell_qty_sl > 0.0:
                                    gross = sell_qty_sl * px
                                    net = gross - (gross * commission_rate)
                                    cash += net
                                    qty -= sell_qty_sl
                            else:
                                gross = qty * px
                                net = gross - (gross * commission_rate)
                                cash += net
                                qty = 0.0
                            
                            if qty <= 1e-12:
                                qty = 0.0; avg_cost = 0.0; last_purchase_price = 0.0
                                last_signal_price = 0.0 
                                tp_after_base_price = 0.0
                                # (*** MUDANÇA v1.9: Desarma TBRS se sair por SL ***)
                                tbrs_4h_low = 0.0; tbrs_8h_low = 0.0; tbrs_1d_low = 0.0
                else:
                    # --- LÓGICA SL "DOWN" (Abaixo/Igual EMA, ou EMA == 0) ---
                    if use_sl_down:
                        sl_price = sl_base_price * (1.0 + sl_down_pct / 100.0)
                        if px <= sl_price:
                            sell_actions_sl += 1
                            if use_sl_down_partial:
                                sell_qty_sl = min(qty, qty * sl_down_sell_mult)
                                if sell_qty_sl > 0.0:
                                    gross = sell_qty_sl * px
                                    net = gross - (gross * commission_rate)
                                    cash += net
                                    qty -= sell_qty_sl
                            else:
                                gross = qty * px
                                net = gross - (gross * commission_rate)
                                cash += net
                                qty = 0.0
                            
                            if qty <= 1e-12:
                                qty = 0.0; avg_cost = 0.0; last_purchase_price = 0.0
                                last_signal_price = 0.0
                                tp_after_base_price = 0.0
                                # (*** MUDANÇA v1.9: Desarma TBRS se sair por SL ***)
                                tbrs_4h_low = 0.0; tbrs_8h_low = 0.0; tbrs_1d_low = 0.0


            # 1b. Take-profit (After %)
            if qty > 0.0 and use_tp_after and tp_after_base_price > 0.0:
                
                trigger_price = tp_after_base_price * tp_after_trigger_mult
                
                if px >= trigger_price:
                    sell_qty_after = min(qty, qty * tp_after_sell_mult)
                    if sell_qty_after > 0.0:
                        gross = sell_qty_after * px
                        net = gross - (gross * commission_rate)
                        cash += net
                        qty -= sell_qty_after
                        sell_actions_tp_after += 1
                        
                        tp_after_base_price = px 
                        
                        if qty <= 1e-12:
                            qty = 0.0
                            avg_cost = 0.0
                            last_purchase_price = 0.0
                            last_signal_price = 0.0
                            tp_after_base_price = 0.0
                            # (*** MUDANÇA v1.9: Desarma TBRS ***)
                            tbrs_4h_low = 0.0; tbrs_8h_low = 0.0; tbrs_1d_low = 0.0

            # 1c. Take-profit Fixo (base: avg_cost)
            if qty > 0.0 and use_tp and avg_cost > 0.0: 
                change_pct = (px / avg_cost - 1.0) * 100.0
                if change_pct >= tp_pct:
                    gross = qty * px 
                    net = gross - (gross * commission_rate)
                    cash += net
                    qty = 0.0
                    avg_cost = 0.0
                    last_purchase_price = 0.0
                    last_signal_price = 0.0 
                    tp_after_base_price = 0.0
                    sell_actions_tp_fixed += 1 
                    # (*** MUDANÇA v1.9: Desarma TBRS ***)
                    tbrs_4h_low = 0.0; tbrs_8h_low = 0.0; tbrs_1d_low = 0.0

            # 1d. Take-profit EMA (Dinâmico)
            if qty > 0.0 and use_tp_ema: 
                if ema_val > 0.0: 
                    trigger_price = ema_val * tp_ema_trigger_mult
                    
                    if px >= trigger_price:
                        sell_qty_ema = min(qty, qty * tp_ema_sell_mult)
                        if sell_qty_ema > 0.0:
                            gross = sell_qty_ema * px
                            net = gross - (gross * commission_rate)
                            cash += net
                            qty -= sell_qty_ema
                            sell_actions_tp_ema += 1
                            if qty <= 1e-12:
                                qty = 0.0
                                avg_cost = 0.0
                                last_purchase_price = 0.0
                                last_signal_price = 0.0 
                                tp_after_base_price = 0.0 
                                # (*** MUDANÇA v1.9: Desarma TBRS ***)
                                tbrs_4h_low = 0.0; tbrs_8h_low = 0.0; tbrs_1d_low = 0.0

        # --- 2) Process buys (4H -> 8H -> 1D) ---
        eq_now = cash + (qty * px) 
        
        cur_pos_value = qty * px
        max_pos_value = eq_now * max_exposure
        room_value = max(0.0, max_pos_value - cur_pos_value)
        
        # --- Buy 4H ---
        if is_buy_4h: 
            target_value = eq_now * (buy4h_pct / 100.0)
            alloc = min(target_value, cash, room_value)
            if alloc > 1.0: 
                net = alloc - (alloc * commission_rate)
                qty_buy = net / px
                new_value = (qty * avg_cost) + net
                cash -= alloc
                qty += qty_buy
                avg_cost = (new_value / qty) if qty > 0.0 else 0.0
                last_purchase_price = px 
                tp_after_base_price = px
                buy_actions_4h += 1 
                
                # (*** MUDANÇA v1.9: Armar TBRS ***)
                if use_tbrs:
                    tbrs_4h_low = low
                    tbrs_8h_low = 0.0
                    tbrs_1d_low = 0.0
                
                eq_now = cash + (qty * px); room_value = max(0.0, max_pos_value - (qty * px)) 
            else:
                buy_ignored_4h += 1 
        
        # --- Buy 8H ---
        if is_buy_8h: 
            target_value = eq_now * (buy8h_pct / 100.0)
            alloc = min(target_value, cash, room_value) 
            if alloc > 1.0:
                net = alloc - (alloc * commission_rate)
                qty_buy = net / px
                new_value = (qty * avg_cost) + net
                cash -= alloc
                qty += qty_buy
                avg_cost = (new_value / qty) if qty > 0.0 else 0.0
                last_purchase_price = px 
                tp_after_base_price = px
                buy_actions_8h += 1 
                
                # (*** MUDANÇA v1.9: Armar TBRS ***)
                if use_tbrs:
                    tbrs_4h_low = 0.0
                    tbrs_8h_low = low
                    tbrs_1d_low = 0.0
                
                eq_now = cash + (qty * px); room_value = max(0.0, max_pos_value - (qty * px)) 
            else:
                buy_ignored_8h += 1 

        # --- Buy 1D ---
        if is_buy_1d: 
            target_value = eq_now * (buy1d_pct / 100.0)
            alloc = min(target_value, cash, room_value)
            if alloc > 1.0:
                net = alloc - (alloc * commission_rate)
                qty_buy = net / px
                new_value = (qty * avg_cost) + net
                cash -= alloc
                qty += qty_buy
                avg_cost = (new_value / qty) if qty > 0.0 else 0.0
                last_purchase_price = px 
                tp_after_base_price = px
                buy_actions_1d += 1
                
                # (*** MUDANÇA v1.9: Armar TBRS ***)
                if use_tbrs:
                    tbrs_4h_low = 0.0
                    tbrs_8h_low = 0.0
                    tbrs_1d_low = low
            else:
                buy_ignored_1d += 1 
        
        
        # --- 3) Process sells (1D -> 8H -> 4H) ---
        is_sell_1d = sell_1d_sig[i] and sell1d_pct > 0.0
        is_sell_8h = sell_8h_sig[i] and sell8h_pct > 0.0
        is_sell_4h = sell_4h_sig[i] and sell4h_pct > 0.0

        if is_sell_1d: sell_signals_1d += 1
        if is_sell_8h: sell_signals_8h += 1
        if is_sell_4h: sell_signals_4h += 1

        is_any_sell = is_sell_1d or is_sell_8h or is_sell_4h

        if is_any_sell:
            if qty > 0.0: 
                # --- Sell 1D ---
                if is_sell_1d:
                    sell_qty = min(qty, qty * (sell1d_pct / 100.0))
                    if sell_qty > 0.0:
                        gross = sell_qty * px
                        net = gross - (gross * commission_rate)
                        cash += net
                        qty -= sell_qty
                        sell_actions_1d += 1 
                        if qty <= 1e-12:
                            qty = 0.0; avg_cost = 0.0; last_purchase_price = 0.0
                            last_signal_price = 0.0; tp_after_base_price = 0.0
                            # (*** MUDANÇA v1.9: Desarma TBRS ***)
                            tbrs_4h_low = 0.0; tbrs_8h_low = 0.0; tbrs_1d_low = 0.0
                
                # --- Sell 8H ---
                if is_sell_8h and qty > 0.0: 
                    sell_qty = min(qty, qty * (sell8h_pct / 100.0))
                    if sell_qty > 0.0:
                        gross = sell_qty * px
                        net = gross - (gross * commission_rate)
                        cash += net
                        qty -= sell_qty
                        sell_actions_8h += 1 
                        if qty <= 1e-12:
                            qty = 0.0; avg_cost = 0.0; last_purchase_price = 0.0
                            last_signal_price = 0.0; tp_after_base_price = 0.0
                            # (*** MUDANÇA v1.9: Desarma TBRS ***)
                            tbrs_4h_low = 0.0; tbrs_8h_low = 0.0; tbrs_1d_low = 0.0

                # --- Sell 4H ---
                if is_sell_4h and qty > 0.0: 
                    sell_qty = min(qty, qty * (sell4h_pct / 100.0))
                    if sell_qty > 0.0:
                        gross = sell_qty * px
                        net = gross - (gross * commission_rate)
                        cash += net
                        qty -= sell_qty
                        sell_actions_4h += 1 
                        if qty <= 1e-12:
                            qty = 0.0; avg_cost = 0.0; last_purchase_price = 0.0
                            last_signal_price = 0.0; tp_after_base_price = 0.0
                            # (*** MUDANÇA v1.9: Desarma TBRS ***)
                            tbrs_4h_low = 0.0; tbrs_8h_low = 0.0; tbrs_1d_low = 0.0
            else:
                if is_sell_1d: sell_ignored_1d += 1
                if is_sell_8h: sell_ignored_8h += 1
                if is_sell_4h: sell_ignored_4h += 1
                
        # --- 4) Track equity ---
        equity_out[i] = cash + (qty * px)
    
    # (*** MUDANÇA v1.9: 23 métricas ***)
    return (
        buy_actions_4h, buy_actions_8h, buy_actions_1d,
        sell_actions_4h, sell_actions_8h, sell_actions_1d,
        buy_signals_4h, buy_signals_8h, buy_signals_1d,
        sell_signals_4h, sell_signals_8h, sell_signals_1d,
        buy_ignored_4h, buy_ignored_8h, buy_ignored_1d,
        sell_ignored_4h, sell_ignored_8h, sell_ignored_1d,
        sell_actions_sl, sell_actions_tp_fixed, sell_actions_tp_after, sell_actions_tp_ema,
        sell_actions_tbrs # <--- NOVO
    )