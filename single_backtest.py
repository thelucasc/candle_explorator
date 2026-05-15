import sys
import argparse
import pandas as pd
import numpy as np
import json
import os
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

# Importa engines existentes
import core_engine as ce
import indicators as ind
import data_utils as du
import logging_utils as lu

def load_data(file_path):
    # (*** FIX: paridade com sweep.py — usar o mesmo loader (UTC, sort, strip tz) ***)
    # A função antiga divergia em timezone/ordenação e produzia np_data diferente,
    # mesmo com indicadores e slicing corretos.
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Arquivo nao encontrado: {file_path}")
    return du.load_price_csv(file_path)

def main():
    p = argparse.ArgumentParser(description="Single Backtest Runner")
    
    # --- Arquivos e Datas ---
    p.add_argument("--file", required=True, help="Arquivo CSV Diário (1D)")
    p.add_argument("--file-4h", default=None, help="Arquivo CSV 4H")
    p.add_argument("--file-8h", default=None, help="Arquivo CSV 8H")
    p.add_argument("--date", default="ALL", help="Filtro de data (dias ou 'ALL')")
    
    # --- Parâmetros de Compra/Venda ---
    p.add_argument("--buy1d", type=float, default=0.0)
    p.add_argument("--sell1d", type=float, default=0.0)
    p.add_argument("--buy4h", type=float, default=0.0)
    p.add_argument("--sell4h", type=float, default=0.0)
    p.add_argument("--buy8h", type=float, default=0.0)
    p.add_argument("--sell8h", type=float, default=0.0)
    
    # --- Stop Loss ---
    p.add_argument("--sl-up-pct", type=float, default=0.0)
    p.add_argument("--sl-up-amount", type=float, default=100.0)
    p.add_argument("--sl-down-pct", type=float, default=0.0)
    p.add_argument("--sl-down-amount", type=float, default=100.0)
    p.add_argument("--sl-on-last-buy", action='store_true')
    
    # --- Take Profit ---
    p.add_argument("--tp-pct", type=float, default=0.0)
    p.add_argument("--tp-after-pct", type=float, default=0.0)
    p.add_argument("--tp-sell-pct", type=float, default=0.0) # tp_after_amount
    p.add_argument("--tp-ema-pct", type=float, default=0.0)
    p.add_argument("--tp-ema-amount", type=float, default=0.0)
    p.add_argument("--ema-len", type=int, default=20)
    
    # --- Sinais 4H/8H ---
    p.add_argument("--sig-4h8h-sma", type=int, default=20)
    p.add_argument("--sig-4h8h-pir", type=float, default=0.0)
    
    # --- Sinais 1D ---
    p.add_argument("--sig-1d-sma", type=int, default=20)
    p.add_argument("--sig-1d-trend-reg", type=float, default=0.0)
    p.add_argument("--sig-1d-trend-tree", type=float, default=0.0)
    p.add_argument("--sig-1d-dist", type=float, default=0.0)
    p.add_argument("--sig-1d-rsi-len", type=int, default=14)
    p.add_argument("--sig-1d-rsi-th", type=float, default=30.0)
    p.add_argument("--sig-1d-pir-prev", type=float, default=0.0)
    p.add_argument("--sig-1d-pir-conf", type=float, default=0.0)
    p.add_argument("--buy-only-up-emas", type=float, default=0.0)
    
    # --- Config Global ---
    p.add_argument("--capital", type=float, default=100000.0)
    p.add_argument("--commission-bps", type=float, default=1.0, help="Comissao em basis points (1 bps = 0.01%%)")
    p.add_argument("--exposure", type=float, default=100.0)
    p.add_argument("--sl-signal", action='store_true')
    p.add_argument("--tbrs", action='store_true')
    p.add_argument("--short-on-stop", action='store_true')
    p.add_argument("--stops-on-candle", nargs='+', default=['1D'])

    args = p.parse_args()
    
    try:
        # 1. Carregar Dados
        # print("Carregando CSVs...")
        df1d = load_data(args.file)
        df4h = load_data(args.file_4h) if args.file_4h else None
        df8h = load_data(args.file_8h) if args.file_8h else None
        
        # (*** FIX: paridade de warm-up com sweep.py ***)
        # Ordem CORRETA: sinais e EMAs primeiro no DF COMPLETO, slice DEPOIS.
        # Antes era slice -> compute, o que zerava SMA/RSI no início do slice
        # e produzia trades diferentes do sweep principal.

        filter_ema = int(args.buy_only_up_emas)

        # 1. Sinais no DF COMPLETO
        df1d = ind.compute_1D_cluster_signals(
            df1d,
            sma_length=args.sig_1d_sma,
            trend_regime_threshold=args.sig_1d_trend_reg,
            trend_regime_tree_threshold=args.sig_1d_trend_tree,
            dist_ma_fast_threshold=args.sig_1d_dist,
            rsi_length=args.sig_1d_rsi_len,
            rsi_threshold=args.sig_1d_rsi_th,
            pir_threshold_prev=args.sig_1d_pir_prev,
            pir_threshold_confirm=args.sig_1d_pir_conf,
            filter_ema_length=filter_ema
        )

        if df4h is not None:
            df4h = ind.compute_pine_like_signals(
                df4h,
                sma_length=args.sig_4h8h_sma,
                pir_threshold=args.sig_4h8h_pir,
                filter_ema_length=filter_ema
            )

        if df8h is not None:
            df8h = ind.compute_pine_like_signals(
                df8h,
                sma_length=args.sig_4h8h_sma,
                pir_threshold=args.sig_4h8h_pir,
                filter_ema_length=filter_ema
            )

        # 2. EMA de trailing também no DF COMPLETO
        target_ema_len = args.ema_len
        df1d[f'ema_{target_ema_len}'] = ind.ema(df1d['close'], target_ema_len)
        if df4h is not None: df4h[f'ema_{target_ema_len}'] = ind.ema(df4h['close'], target_ema_len)
        if df8h is not None: df8h[f'ema_{target_ema_len}'] = ind.ema(df8h['close'], target_ema_len)

        unique_ema_lens = [target_ema_len]

        # 3. SÓ AGORA fatia por data — indicadores já vêm "aquecidos" para o início do período.
        if args.date != "ALL":
            days = int(args.date)
            df1d = du.slice_period(df1d, days)
            if df4h is not None: df4h = du.cut_matching(df4h, df1d)
            if df8h is not None: df8h = du.cut_matching(df8h, df1d)

        # 4. Preparar NumPy Data
        np_data = ce.prepare_numpy_data(df1d, df4h, df8h, unique_ema_lens)
        
        # 5. Configurar Payload para o Worker/Engine
        common_params = {
            'initial_capital': args.capital,
            'commission_bps': args.commission_bps,
            'max_exposure_pct': args.exposure,
            'stop_loss_signal': args.sl_signal,
            'stops_on_candle': args.stops_on_candle,
            'two_bar_reversal_stop': args.tbrs,
            'short_on_stop': args.short_on_stop,
            'use_sl_on_last_buy': args.sl_on_last_buy
        }
        
        payload = {
            'common_params': common_params,
            'date_list': ['SINGLE'],
            'numpy_data_slices': {'SINGLE': np_data},
            'has_multiple_signal_params': False,
            'unique_ema_lens': unique_ema_lens,
            'default_ema_len': target_ema_len,
            # B&H mockado pois calcularemos real no output se precisar, mas engine precisa da chave
            'bh_benchmarks': {'SINGLE': 0.0} 
        }
        
        # Inicializa o worker com os dados (simula comportamento do sweep)
        ce.init_worker(payload)
        
        # 6. Montar Combo Tuple
        # Ordem de parâmetros conforme run_full_track_for_combo espera (27 itens)
        combo = (
            args.sig_4h8h_sma, args.sig_4h8h_pir,
            args.sig_1d_sma, args.sig_1d_trend_reg, args.sig_1d_trend_tree,
            args.sig_1d_dist, args.sig_1d_rsi_len, args.sig_1d_rsi_th,
            args.sig_1d_pir_prev, args.sig_1d_pir_conf, args.buy_only_up_emas,
            args.buy4h, args.sell4h, args.buy8h, args.sell8h, args.buy1d, args.sell1d,
            args.sl_up_pct, args.sl_up_amount, args.sl_down_pct, args.sl_down_amount,
            args.tp_pct, args.tp_after_pct, args.tp_sell_pct,
            args.tp_ema_pct, args.tp_ema_amount, args.ema_len
        )
        
        # 7. Rodar Backtest Full Track
        # (*** FIX: usa final_equity do equity_out array (mark-to-market no último candle)
        # em vez de history[-1]['equity'] que era equity no último trade. ***)
        history, final_equity = ce.run_full_track_for_combo(combo, 'SINGLE')

        # 8. Output CSV Format (para o GUI ler)
        # Header
        print("Date,Type,Price,Qty,Reason,PnL,Equity")

        if history:
            print(f"DEBUG: Processando {len(history)} registros no historico.")
            for t in history:
                # Formata data
                d = t.get('date', '')
                if isinstance(d, (pd.Timestamp, np.datetime64)):
                    d_str = str(d)
                else:
                    d_str = str(d)
                
                # Imprime tudo
                reason_val = str(t.get('reason', ''))
                safe_reason = reason_val.replace(',', ';')
                print(f"{d_str},{t['type']},{t.get('price', 0.0):.2f},{t.get('qty', 0.0):.4f},{safe_reason},{t.get('pnl_trade', 0.0):.2f},{t.get('equity', 0.0):.2f}")
        
        # Resumo final JSON no final do output para fácil parse
        pnl_pct = (final_equity / args.capital - 1.0) * 100.0
        
        # Calc B&H
        buh_pct = 0.0
        if not df1d.empty:
            start_price = df1d['open'].iloc[0]
            end_price = df1d['close'].iloc[-1]
            if start_price > 0:
                buh_pct = (end_price - start_price) / start_price * 100.0
        
        summary = {
            "final_equity": final_equity,
            "pnl_pct": pnl_pct,
            "buy_and_hold_pct": buh_pct,
            "total_trades": len([x for x in history if x['type'] in ['SELL', 'STOP_LOSS', 'TAKE_PROFIT']])
        }
        print("---SUMMARY---")
        print(json.dumps(summary))

    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"ERROR: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
