#!/usr/bin/env python3
# sweep.py — v1.8-tbrs (Refatorado v2.0)
# Este script importa os novos módulos de utilitários

import argparse
import itertools
import csv
import random 
from datetime import datetime
from multiprocessing import Pool, cpu_count
import sys
from typing import Optional, Dict, List, Tuple

import pandas as pd
import numpy as np

# (*** NOVO: Importa os módulos componentizados ***)
import data_utils as du
import indicators as ind
import logging_utils as lu
import core_engine as ce

# ***** Silenciando warnings *****
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)


# (*** NOVO: Helper para processar os grids B/S ***)
def parse_bs_grid_arg(arg_list: Optional[List[str]], step_value: int) -> List[float]:
    """
    Processa o argumento do grid B/S.
    Pode ser None (-> [0.0]), ['step'] (-> [0, 5, 10...]), ou ['10', '20'] (-> [10.0, 20.0])
    """
    grid_off = [0.0]
    
    if arg_list is None:
        return grid_off  # Não especificado, retorna [0.0]
    
    # Caso 1: Usuário digitou '--buy1d step'
    if len(arg_list) == 1 and arg_list[0].lower() == 'step':
        if step_value <= 0:
            print("[ERRO] --step deve ser > 0 para usar o modo 'step'.")
            sys.exit(1)
        return list(range(0, 101, step_value))
    
    # Caso 2: Usuário digitou valores específicos, ex: '--buy1d 10 25 50'
    try:
        return [float(val) for val in arg_list]
    except ValueError as e:
        print(f"[ERRO] Valor inválido no grid B/S: {arg_list}. Use números (ex: 10 25 50) ou a palavra 'step'.")
        print(f"Erro: {e}")
        sys.exit(1)


# ==================== MAIN ====================
def main():
    # (*** MUDANÇA v1.6: Descrição para 16P ***)
    p = argparse.ArgumentParser(description="Sweep de parâmetros (IN-PROCESS, 16P) para backtest.py.")
    p.add_argument("--file",    default="BTCUSDT1D.csv",   help="CSV 1D (Obrigatório)")
    p.add_argument("--file-4h", default=None,  help="CSV 4H")
    p.add_argument("--file-8h", default=None,  help="CSV 8H")

    # (*** MUDANÇA v1.6: Aceita múltiplas datas ***)
    p.add_argument("--date", type=str, default=None, nargs='+', 
                   help="Período(s) do sweep: 'all', ou um/vários números de dias (ex: 1400 365). Default: 365 1825")

    p.add_argument("--step",    type=int, default=25,      help="Passo 0..100 (default: 25). Usado apenas se 'step' for invocado nos grids B/S.")
    p.add_argument("--workers", type=int, default=max(1, cpu_count()-1), help="Nº de processos (default: núcleos-1)")
    p.add_argument("--out",     default="sweep_results.csv", help="CSV de saída (salva o que for printado)")
    
    p.add_argument("--print-mode", default="all", choices=["all", "winners", "pos"],
                   help="Modo de print/save: 'all' (padrão), 'winners' (só os que batem B&H), 'pos' (só retornos > 0)")

    p.add_argument("--start-pct", type=float, default=0.0,
                   help="Percentual (0-100) para iniciar o sweep (ex: 50 para começar da metade).")

    # (*** LÓGICA DE GRID MODIFICADA v1.4: type=str ***)
    p.add_argument("--buy1d", type=str, default=None, nargs='+', help="Valores (ex: 10 25 50) ou a palavra 'step' para usar --step.")
    p.add_argument("--sell1d", type=str, default=None, nargs='+', help="Valores (ex: 10 25 50) ou a palavra 'step' para usar --step.")
    p.add_argument("--buy4h", type=str, default=None, nargs='+', help="Valores (ex: 10 25 50) ou a palavra 'step' para usar --step.")
    p.add_argument("--sell4h", type=str, default=None, nargs='+', help="Valores (ex: 10 25 50) ou a palavra 'step' para usar --step.")
    p.add_argument("--buy8h", type=str, default=None, nargs='+', help="Valores (ex: 10 25 50) ou a palavra 'step' para usar --step.")
    p.add_argument("--sell8h", type=str, default=None, nargs='+', help="Valores (ex: 10 25 50) ou a palavra 'step' para usar --step.")

    # Parâmetros de Repasse (Comuns)
    p.add_argument("--initial-capital", type=float, default=100000.0)
    p.add_argument("--commission-bps",  type=float, default=1.0) 
    p.add_argument("--max-exposure-pct",type=float, default=100.0)
    
    p.add_argument("--stop-loss-signal", action="store_true", help="Usa o preço do último SINAL de compra (mesmo ignorado) como base do SL.")
    
    p.add_argument("--stops-on-candle", type=str, default=None, nargs='+', choices=['1D', '4H', '8H'], 
                   help="Só checa SL/TP nos candles dos timeframes especificados (ex: 1D 4H). Default: todos.")
    
    # (*** MUDANÇA v1.9: Adiciona TBRS ***)
    p.add_argument("--two-bar-reversal-stop", action="store_true", 
                   help="Vende a posição se o candle seguinte (do mesmo TF da compra) fechar abaixo da mínima do candle de compra.")
    
    # Parâmetros de Sweep (Listas)
    
    # (*** MUDANÇA v1.5: SL Bifurcado ***)
    p.add_argument("--sl-up-pct", type=float, default=None, nargs='+', help="SL Pct se Preço > EMA")
    p.add_argument("--sl-up-amount", type=float, default=None, nargs='+', help="SL Amount se Preço > EMA")
    p.add_argument("--sl-down-pct", type=float, default=None, nargs='+', help="SL Pct se Preço <= EMA")
    p.add_argument("--sl-down-amount", type=float, default=None, nargs='+', help="SL Amount se Preço <= EMA")
    # (*** FIM DA MUDANÇA ***)
    
    p.add_argument("--tp-pct", type=float, default=None, nargs='+')
    
    # (*** MUDANÇA v1.7: Novos TPs ***)
    p.add_argument("--take-profit-after-percentage", type=float, default=None, nargs='+', 
                   help="Gatilho de Pct de lucro (base: último PREÇO de compra) para acionar TP.")
    p.add_argument("--take-profit-percentage", type=float, default=None, nargs='+', 
                   help="Percentual do saldo a vender quando --take-profit-after-percentage é atingido.")
    # (*** FIM DA MUDANÇA ***)
    
    p.add_argument("--tp-ema-pct", type=float, default=None, nargs='+')
    p.add_argument("--tp-ema-amount", type=float, default=None, nargs='+')
    p.add_argument("--emas", type=int, default=None, nargs='+')

    args = p.parse_args()
    
    # Validações
    use_any_4h = args.buy4h is not None or args.sell4h is not None
    use_any_8h = args.buy8h is not None or args.sell8h is not None
    
    if not (args.buy1d is not None or args.sell1d is not None or use_any_4h or use_any_8h):
        print("[ERRO] Nenhum grid B/S especificado. Use --buy1d 10 20 ou --buy1d step.")
        sys.exit(1)
    
    # (*** MUDANÇA v1.6: Checagem de TF ***)
    if not args.file_4h:
        print("[ERRO] --file-4h é obrigatório (para o índice unificado).")
        sys.exit(1)
    if not args.file_8h:
        print("[ERRO] --file-8h é obrigatório (para o índice unificado).")
        sys.exit(1)
    
    
    # --- 1. Carregar Dados (UMA VEZ) ---
    print(f"[INFO] Carregando CSVs (uma única vez)...")
    try:
        # (*** USA data_utils ***)
        df1d_thin = du.load_price_csv(args.file)
        df4h_thin = du.load_price_csv(args.file_4h) if args.file_4h else None
        df8h_thin = du.load_price_csv(args.file_8h) if args.file_8h else None
    except Exception as e:
        print(f"Erro ao carregar CSVs: {e}")
        sys.exit(1)

    # --- 2. Preparar Grids de Parâmetros ---
    
    grid_b4 = parse_bs_grid_arg(args.buy4h, args.step)
    grid_s4 = parse_bs_grid_arg(args.sell4h, args.step)
    grid_b8 = parse_bs_grid_arg(args.buy8h, args.step)
    grid_s8 = parse_bs_grid_arg(args.sell8h, args.step)
    grid_b1 = parse_bs_grid_arg(args.buy1d, args.step)
    grid_s1 = parse_bs_grid_arg(args.sell1d, args.step)
    
    sl_up_pct_grid = args.sl_up_pct if args.sl_up_pct is not None else [None]
    sl_up_amt_grid = args.sl_up_amount if args.sl_up_amount is not None else [100.0]
    sl_down_pct_grid = args.sl_down_pct if args.sl_down_pct is not None else [None]
    sl_down_amt_grid = args.sl_down_amount if args.sl_down_amount is not None else [100.0]
    
    tp_grid = args.tp_pct if args.tp_pct is not None else [None]
    
    # (*** MUDANÇA v1.7: Grids Novos TPs ***)
    tp_after_pct_grid = args.take_profit_after_percentage if args.take_profit_after_percentage is not None else [None]
    tp_sell_pct_grid = args.take_profit_percentage if args.take_profit_percentage is not None else [None]
    # (*** FIM DA MUDANÇA ***)
    
    tp_ema_pct_grid = args.tp_ema_pct if args.tp_ema_pct is not None else [None]
    tp_ema_amt_grid = args.tp_ema_amount if args.tp_ema_amount is not None else [None]
    ema_len_grid = args.emas if args.emas is not None else [20] 
    
    unique_ema_lens = sorted(list(set(int(l) for l in ema_len_grid if l is not None and l > 0)))
    if not unique_ema_lens:
        unique_ema_lens = [20] # Adiciona 20 como padrão se a lista estiver vazia
    
    default_ema_len = unique_ema_lens[0]

    # --- 3. (Otimização Nível 2) ---
    print("[INFO] Otimização Nível 2: Pré-calculando Sinais 1D...")
    # (*** USA indicators ***)
    df1d = ind.compute_1D_cluster_signals(df1d_thin)
    
    if df4h_thin is not None:
        print("[INFO] Pré-calculando Sinais 4H...")
        # (*** USA indicators ***)
        df4h = ind.compute_pine_like_signals(df4h_thin)
    else:
        df4h = None
        
    if df8h_thin is not None:
        print("[INFO] Pré-calculando Sinais 8H...")
        # (*** USA indicators ***)
        df8h = ind.compute_pine_like_signals(df8h_thin)
    else:
        df8h = None

    print(f"[INFO] Pré-calculando EMAs para comprimentos: {unique_ema_lens}...")
    for length in unique_ema_lens:
        col_name = f'ema_{length}'
        # (*** USA indicators ***)
        df1d[col_name] = ind.ema(df1d['close'], length)
        if df4h is not None: df4h[col_name] = ind.ema(df4h['close'], length)
        if df8h is not None: df8h[col_name] = ind.ema(df8h['close'], length)
    
    print("[INFO] Pré-cálculo de indicadores concluído.")
    # --- FIM DA OTIMIZAÇÃO ---


    # --- 4. Preparar Slices e Modo de Execução ---
    
    stops_on_candle = args.stops_on_candle if args.stops_on_candle is not None else ['1D', '4H', '8H']
    
    common_params = {
        'initial_capital': args.initial_capital,
        'commission_bps': args.commission_bps,
        'max_exposure_pct': args.max_exposure_pct,
        'stop_loss_signal': args.stop_loss_signal,
        'stops_on_candle': stops_on_candle,
        'two_bar_reversal_stop': args.two_bar_reversal_stop # (*** MUDANÇA v1.9: TBRS ***)
    }

    worker_data_payload = {
        'common_params': common_params,
        'default_ema_len': default_ema_len, 
    }
    
    # (*** MUDANÇA v1.6: Lógica Multi-Date ***)
    
    # Se nenhuma data for passada, usa 365d e 5 anos (1825 dias)
    date_list_str = args.date if args.date is not None else ['365', '1825']
    print(f"[INFO] Rodando em modo Multi-Date para períodos: {date_list_str}")

    bh_benchmarks = {}
    numpy_data_slices = {}
    custom_labels = [] # Lista ordenada de labels (ex: '1400', '365')

    for date_str in date_list_str:
        bh_custom_pct = 0.0
        
        if date_str.lower() == 'all':
            custom_days_label_for_logger = "ALL"
            df1d_custom = df1d 
            df4h_custom = df4h 
            df8h_custom = df8h 
        
        else:
            try:
                days_custom = int(date_str)
                if days_custom <= 0: raise ValueError()
                
                custom_days_label_for_logger = f"{days_custom}"
                # (*** USA data_utils ***)
                df1d_custom = du.slice_period(df1d, days_custom)
                
                # (*** GUARDA: Se o slice for maior que os dados, usa o que tem ***)
                if len(df1d_custom) < days_custom and len(df1d_custom) > 0:
                    print(f"[AVISO] Período '{days_custom}' é maior que os dados ({len(df1d_custom)} dias). Usando max disponível.")
                elif len(df1d_custom) == 0:
                    print(f"[AVISO] Período '{days_custom}' não contém dados (slice vazio).")
                
                # (*** USA data_utils ***)
                df4h_custom = du.cut_matching(df4h, df1d_custom)
                df8h_custom = du.cut_matching(df8h, df1d_custom)

            except ValueError:
                print(f"[ERRO] Argumento --date inválido: '{date_str}'. Use 'all' ou um número de dias.")
                sys.exit(1)
        
        # (*** GUARDA: Checa B&H e range ***)
        if not df1d_custom.empty and 'close' in df1d_custom.columns and len(df1d_custom['close']) > 1:
            p0 = float(df1d_custom['close'].iloc[0])
            p1 = float(df1d_custom['close'].iloc[-1])
            if p0 > 0:
                bh_custom_pct = (p1 / p0 - 1.0) * 100.0
            
            # (*** USA logging_utils ***)
            date_range_str = lu.get_date_range_str(df1d_custom)
            print(f"[INFO] Período '{custom_days_label_for_logger}': {date_range_str} | B&H: {bh_custom_pct:+.2f}%")
        
        else:
            print(f"[AVISO] Não foi possível calcular B&H para o período '{custom_days_label_for_logger}' (dataframe vazio).")

        
        print(f"[INFO] Preparando arrays NumPy ({custom_days_label_for_logger})...")
        # (*** USA core_engine ***)
        numpy_data_slices[custom_days_label_for_logger] = ce.prepare_numpy_data(
            df1d_custom, df4h_custom, df8h_custom, unique_ema_lens
        )
        
        bh_benchmarks[custom_days_label_for_logger] = bh_custom_pct
        custom_labels.append(custom_days_label_for_logger)
        
        del df1d_custom, df4h_custom, df8h_custom

    
    worker_data_payload.update({
        'date_list': custom_labels,
        'numpy_data_slices': numpy_data_slices,
        'bh_benchmarks': bh_benchmarks
    })
    # (*** FIM DA MUDANÇA v1.6 ***)
    
    del df1d, df4h, df8h, df1d_thin, df4h_thin, df8h_thin
    print("[INFO] Preparação de dados NumPy concluída. DataFrames limpos da memória.")


    # --- 5. Preparar Combinações ---
    
    # (*** MUDANÇA v1.7: 16 parâmetros ***)
    combos = list(itertools.product(
        grid_b4, grid_s4,
        grid_b8, grid_s8,
        grid_b1, grid_s1,
        sl_up_pct_grid,       
        sl_up_amt_grid,       
        sl_down_pct_grid,     
        sl_down_amt_grid,     
        tp_grid,
        tp_after_pct_grid,    # <--- NOVO
        tp_sell_pct_grid,     # <--- NOVO
        tp_ema_pct_grid,
        tp_ema_amt_grid,
        ema_len_grid 
    ))
    
    total = len(combos) 

    start_index = 0
    if args.start_pct > 0:
        if args.start_pct >= 100:
            print(f"ERRO: --start-pct ({args.start_pct}) deve ser menor que 100.")
            sys.exit(1)
        start_index = int(total * (args.start_pct / 100.0))
        print(f"[INFO] Iniciando em {args.start_pct}% (pulando as primeiras {start_index} de {total} combinações).")
    
    combos_to_run = combos[start_index:] 

    # --- 6. Preparar CSV de Saída ---
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        metrics_cols_base = [
            "b_exec_4h", "b_exec_8h", "b_exec_1d",
            "s_exec_4h", "s_exec_8h", "s_exec_1d",
            "b_sig_4h", "b_sig_8h", "b_sig_1d",
            "s_sig_4h", "s_sig_8h", "s_sig_1d",
            "b_ign_4h", "b_ign_8h", "b_ign_1d",
            "s_ign_4h", "s_ign_8h", "s_ign_1d",
            "sl", "tp", "tp_after", "ema",
            "tbrs" # (*** MUDANÇA v1.9: 23 métricas ***)
        ]
        
        params_header = [
            "buy4h","sell4h","buy8h","sell8h","buy1d","sell1d",
            "sl_up_pct", "sl_up_amount", "sl_down_pct", "sl_down_amount", 
            "tp_pct",
            "tp_after_pct", "tp_sell_pct", # <--- MUDANÇA v1.7
            "tp_ema_pct", "tp_ema_amount", "ema_len"
        ]
        
        # (*** MUDANÇA v1.6: Header dinâmico ***)
        full_header = [*params_header]
        for label in custom_labels:
            label_suffix = f"_{label}d" if label != "ALL" else "_ALL"
            full_header.append(f"strat{label_suffix}(%)")
            full_header.extend([f"{c}{label_suffix}" for c in metrics_cols_base])
        # (*** FIM DA MUDANÇA ***)

        csv.writer(f).writerow(full_header)

    # (*** CORREÇÃO: Readiciona a definição de format_grid_log ***)
    def format_grid_log(arg_val, grid_val):
        if arg_val is None:
            return "DESATIVADO (None)"
        if len(arg_val) == 1 and arg_val[0].lower() == 'step':
            return f"ATIVADO (STEP {args.step}: {grid_val})"
        return f"ATIVADO (GRID: {grid_val})"
    # (*** FIM DA CORREÇÃO ***)

    print(f"[INFO] CSV 1D : {args.file}")
    print(f"[INFO] CSV 4H : {args.file_4h or '(não usado)'}")
    print(f"[INFO] CSV 8H : {args.file_8h or '(não usado)'}")
    print(f"[INFO] Grid Buy 1D : {format_grid_log(args.buy1d, grid_b1)}")
    print(f"[INFO] Grid Sell 1D: {format_grid_log(args.sell1d, grid_s1)}")
    print(f"[INFO] Grid Buy 4H : {format_grid_log(args.buy4h, grid_b4)}")
    print(f"[INFO] Grid Sell 4H: {format_grid_log(args.sell4h, grid_s4)}")
    print(f"[INFO] Grid Buy 8H : {format_grid_log(args.buy8h, grid_b8)}")
    print(f"[INFO] Grid Sell 8H: {format_grid_log(args.sell8h, grid_s8)}")
    
    print(f"[INFO] Grid SL-UP %: {sl_up_pct_grid}")
    print(f"[INFO] Grid SL-UP Amt: {sl_up_amt_grid}")
    print(f"[INFO] Grid SL-DOWN %: {sl_down_pct_grid}")
    print(f"[INFO] Grid SL-DOWN Amt: {sl_down_amt_grid}")
    
    print(f"[INFO] Grid TP (Fixo): {tp_grid}")
    # (*** MUDANÇA v1.7: Log Novos TPs ***)
    print(f"[INFO] Grid TP-After % (Gatilho): {tp_after_pct_grid}")
    print(f"[INFO] Grid TP-After Amt (Venda): {tp_sell_pct_grid}")
    # (*** FIM DA MUDANÇA ***)
    print(f"[INFO] Grid TP-EMA %: {tp_ema_pct_grid}")
    print(f"[INFO] Grid TP-EMA Amt: {tp_ema_amt_grid}")
    unique_lens_str = ', '.join(map(str, unique_ema_lens))
    print(f"[INFO] Grid EMA Lens: {ema_len_grid} (Pré-calculado: {unique_lens_str})")
    
    print(f"[INFO] Total de combinações: {total}")
    if start_index > 0:
        print(f"[INFO] Combinações a rodar: {len(combos_to_run)} (de {start_index} até {total})")
    print(f"[INFO] Workers: {args.workers}")
    print(f"[INFO] Print Mode: {args.print_mode.upper()}")
    print(f"[INFO] Saída CSV: {args.out}\n")

    start = datetime.now() 
    processed = start_index 
    valid_results_count = 0 

    # (*** MUDANÇA v1.6: Dicionários de tracking ***)
    best_results_by_date = {label: None for label in custom_labels}
    best_pnl_by_date = {label: -float('inf') for label in custom_labels}
    worst_results_by_date = {label: None for label in custom_labels}
    worst_pnl_by_date = {label: float('inf') for label in custom_labels}
    
    # (*** CORREÇÃO: Lógica de "winners" restaurada ***)
    best_logged_winner_pnl_by_date = {label: -float('inf') for label in custom_labels}
    
    sample_results = []
    
    # --- 7. Função de Callback (para printar e salvar) ---
    def handle_result(res):
        nonlocal processed, valid_results_count, sample_results
        nonlocal best_results_by_date, best_pnl_by_date, worst_results_by_date, worst_pnl_by_date
        nonlocal best_logged_winner_pnl_by_date
        
        processed += 1 
        
        if processed % 50000 == 0 or total < 50: # Log mais frequente para testes pequenos 
            elapsed_time = datetime.now() - start
            elapsed_seconds = elapsed_time.total_seconds()
            items_this_run = processed - start_index 
            eta_str = "--:--:--" 
            
            if elapsed_seconds > 1 and items_this_run > 0:
                rate = items_this_run / elapsed_seconds
                items_remaining = total - processed 
                if rate > 0:
                    remaining_seconds = items_remaining / rate
                    # (*** USA logging_utils ***)
                    eta_str = lu.format_time_delta(remaining_seconds)
            
            # (*** USA logging_utils ***)
            elapsed_str = lu.format_time_delta(elapsed_seconds)
            
            print(f"[...]{processed}/{total} ({processed/total*100:.2f}%) "
                  f"Elapsed: {elapsed_str} | ETA: {eta_str}")

        if not res:
            return
        
        valid_results_count += 1
        
        # (*** MUDANÇA v1.6: Lógica de handle Multi-Date ***)
        
        run_mode_res = res[0] # Deve ser "multi-date"
        res_data = res[1]
        
        # (*** MUDANÇA v1.7: 16 parâmetros ***)
        params_tuple = res_data[0:16]
        results_by_date = res_data[16] # Dicionário: {'1400': (pct, metrics), '365': (pct, metrics)}
        
        # --- Tracking e Lógica de Print ---
        pnl_strs = []
        is_any_winner = False
        is_any_positive = False
        
        csv_row = [*params_tuple]
        
        # (*** CORREÇÃO: Usa 'worker_data_payload' em vez de 'g_data' ***)
        date_list_from_worker = worker_data_payload['date_list']
        
        for date_label in date_list_from_worker:
            # (*** MUDANÇA v1.9: 23 métricas ***)
            pct, metrics = results_by_date.get(date_label, (0.0, (0,) * 23)) # Pega com segurança
            
            # Adiciona ao CSV
            csv_row.extend([f"{pct:.6f}", *metrics])
            
            # Checa B&H
            # (*** CORREÇÃO: Usa 'worker_data_payload' em vez de 'g_data' ***)
            bh_pct = worker_data_payload['bh_benchmarks'].get(date_label, 0.0)
            is_winner = pct > bh_pct
            if is_winner: 
                is_any_winner = True
            if pct > 0:
                is_any_positive = True
            
            # Formata string de PnL
            label_suffix = f"d" if date_label != "ALL" else ""
            pnl_strs.append(f"{date_label}{label_suffix} {pct:+.2f}%")
            
            # Atualiza Best/Worst
            if pct > best_pnl_by_date[date_label]:
                best_pnl_by_date[date_label] = pct
                best_results_by_date[date_label] = (*params_tuple, pct, metrics)
            if pct < worst_pnl_by_date[date_label]:
                worst_pnl_by_date[date_label] = pct
                worst_results_by_date[date_label] = (*params_tuple, pct, metrics)

        # Amostragem
        if len(sample_results) < 10:
            sample_results.append(res_data) # Salva o res_data completo
        else:
            j = random.randint(0, valid_results_count - 1)
            if j < 10:
                sample_results[j] = res_data

        # --- Filtros de Print ---
        if args.print_mode == "winners":
            if not is_any_winner:
                return 
            
            # (*** CORREÇÃO: Lógica de "winners" restaurada ***)
            do_print = False
            for date_label in date_list_from_worker:
                # (*** MUDANÇA v1.9: 23 métricas ***)
                pct, _ = results_by_date.get(date_label, (0.0, (0,) * 23))
                # Só printa se for vencedor E for melhor que o último vencedor printado
                if pct > worker_data_payload['bh_benchmarks'].get(date_label, 0.0) and pct > best_logged_winner_pnl_by_date[date_label]:
                    best_logged_winner_pnl_by_date[date_label] = pct
                    do_print = True
            
            if not do_print:
                return # Não é um *novo* melhor vencedor em nenhuma categoria
                
        elif args.print_mode == "pos":
            if not is_any_positive:
                return 
        
        # --- Printar ---
        mark = lu.GREEN if is_any_winner else ""
        label = "[+] WIN" if is_any_winner else "[ ] RUN"
        
        # (*** MUDANÇA v1.7: 16 parâmetros ***)
        b4,s4,b8,s8,b1,s1, \
        sl_up_p, sl_up_amt, sl_down_p, sl_down_amt, \
        tp_p, tp_after_p, tp_sell_p, \
        tp_ema_pct, tp_ema_amt, ema_len = params_tuple
        
        sl_up_str = f"SL-UP {sl_up_p:.1f}%@{sl_up_amt:.0f}%" if sl_up_p is not None else "SL-UP None"
        sl_down_str = f"SL-DN {sl_down_p:.1f}%@{sl_down_amt:.0f}%" if sl_down_p is not None else "SL-DN None"
        sl_str = f"{sl_up_str} | {sl_down_str}"
        tp_str = f"TP {tp_p:.1f}" if tp_p is not None else "TP None"
        # (*** MUDANÇA v1.7: String TP-After ***)
        tp_after_str = f"TP-Aft {tp_after_p:.1f}%@{tp_sell_p:.1f}%" if (tp_after_p is not None and tp_sell_p is not None) else "TP-Aft None"
        
        tp_ema_str = f"EMA{ema_len} {tp_ema_pct:.1f}%@{tp_ema_amt:.1f}%" if (tp_ema_pct is not None and tp_ema_amt is not None) else ""
        
        # (*** MUDANÇA v1.9: TBRS String ***)
        is_tbrs_on = worker_data_payload['common_params'].get('two_bar_reversal_stop', False)
        tbrs_str = "TBRS On" if is_tbrs_on else ""
        
        # (*** MUDANÇA v1.9: Adiciona tbrs_str ***)
        print(f"{mark}{label:<10} B/S (4H {b4}/{s4} | 8H {b8}/{s8} | 1D {b1}/{s1}) | {sl_str} | {tp_str} | {tp_after_str} | {tp_ema_str} | {tbrs_str} | "
              f"PnL: [ {' | '.join(pnl_strs)} ]{lu.RESET}")

        # --- Salvar no CSV ---
        with open(args.out, "a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(csv_row)
            
        # (*** FIM DA MUDANÇA v1.6 ***)

    # --- 8. Executar o Sweep ---
    print("\n" + "="*80)
    print("===== INICIANDO SWEEP RÁPIDO (NUMBA) =====")
    print("="*80 + "\n")

    if args.workers > 1:
        print(f"[INFO] Rodando em modo multi-worker ({args.workers} workers)...")
        # (*** CORREÇÃO: Define g_data na main thread para o modo winners ***)
        global g_data
        g_data = worker_data_payload
        with Pool(processes=args.workers, initializer=ce.init_worker, initargs=(worker_data_payload,)) as pool:
            for res in pool.imap_unordered(ce.run_one_combo, combos_to_run, chunksize=128):
                handle_result(res)
    else:
        print("[INFO] Rodando em modo single-worker (workers=1)...")
        # (*** CORREÇÃO: Define g_data na main thread para o modo winners ***)
        g_data = worker_data_payload
        ce.init_worker(worker_data_payload)
        for combo in combos_to_run:
            res = ce.run_one_combo(combo)
            handle_result(res)

    # --- 9. Printar Sumário Final ---
    dur = datetime.now() - start
    print(f"\nConcluído em {dur}. Resultados salvos em -> {args.out}")
    
    print("\n" + "="*80)
    print("===== SUMÁRIO DA EXECUÇÃO =====")
    print("="*80 + "\n")
    
    # (*** MUDANÇA v1.6: Sumário Multi-Date ***)
    
    date_list_final = worker_data_payload['date_list']
    
    for date_label in date_list_final:
        label_periodo = f"{date_label} Dias" if date_label != "ALL" else "ALL DATA"
        label_short = f"BEST {date_label}D" if date_label != "ALL" else "BEST ALL"
        
        best_res = best_results_by_date[date_label]
        
        if best_res:
            print(f"--- MELHOR RESULTADO (por PnL {label_periodo}) ---")
            # (*** USA logging_utils ***)
            print(lu.format_result_line_custom(best_res, f"[{label_short}]", lu.GREEN, date_label))
        else:
            print(f"--- MELHOR RESULTADO (por PnL {label_periodo}) ---")
            print(f"Nenhum resultado válido foi encontrado para '{label_periodo}'.")

    # (*** LOG DE SL REMOVIDO ***)
            
    # --- Pior Resultado (para a primeira data) ---
    first_date_label = date_list_final[0]
    worst_res_for_log = worst_results_by_date[first_date_label]
    if worst_res_for_log:
        label_periodo = f"{first_date_label} Dias" if first_date_label != "ALL" else "ALL DATA"
        worst_label_short = f"WORST {first_date_label}D" if first_date_label != "ALL" else "WORST ALL"
        print(f"\n--- PIOR RESULTADO (por PnL {label_periodo}) ---")
        # (*** USA logging_utils ***)
        print(lu.format_result_line_custom(worst_res_for_log, f"[{worst_label_short}]", "", first_date_label))

    # --- Amostras Aleatórias ---
    print("\n--- 10 AMOSTRAS ALEATÓRIAS (Resultados do primeiro período) ---")
    if sample_results:
        for i, res in enumerate(sample_results):
            # res = (*combo, results_by_date)
            # Precisamos extrair o resultado do primeiro período para formatar
            # (*** MUDANÇA v1.7: 16 parâmetros ***)
            params = res[0:16]
            results_dict = res[16]
            
            # (*** GUARDA: Garante que a amostra aleatória tenha dados para este período ***)
            if first_date_label in results_dict:
                pct, metrics = results_dict[first_date_label]
                
                # Recria o tuplo de resultado único para a função de formatação
                single_result_tuple = (*params, pct, metrics)
                # (*** USA logging_utils ***)
                print(lu.format_result_line_custom(single_result_tuple, f"[S {i+1}]", "", first_date_label))
    else:
        print("Nenhuma amostra foi coletado.")
        
    print("\n" + "="*80)

    
if __name__ == "__main__":
    main()