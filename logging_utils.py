#!/usr/bin/env python3
# logging_utils.py — v1.0
# Contém: Helpers de Formatação (Log/Sweep) e Cores

import sys
import pandas as pd

# ===== ANSI Colors =====
GREEN = ""
RESET = ""
try:
    import colorama
    colorama.init()
    GREEN = "\033[92m"
    RESET = "\033[0m"
except Exception:
    GREEN = "\033[92m"
    RESET = "\033[0m"

def _print(s: str):
    try:
        print(s)
    except Exception:
        print(s.encode("utf-8", "ignore").decode("utf-8"))

# ===================== Helpers de Formatação (Log/Sweep) =====================

def format_time_delta(seconds):
    """Converte segundos em uma string HH:MM:SS"""
    try:
        s = int(seconds)
        if s < 0:
            return "00:00:00"
        m, s = divmod(s, 60)
        h, m = divmod(m, 60)
        return f"{h:02d}:{m:02d}:{s:02d}"
    except (ValueError, TypeError):
        return "--:--:--"

def get_date_range_str(df: pd.DataFrame) -> str:
    """Retorna o range de datas (min/max) de um DataFrame."""
    if df is None or df.empty:
        return "Período indisponível (DF vazio)"
    try:
        start = df.index.min().strftime('%Y-%m-%d')
        end = df.index.max().strftime('%Y-%m-%d')
        return f"{start} -> {end}"
    except Exception:
        return "Erro ao formatar datas"

def format_result_line_standard(res, label, mark):
    """Formata uma linha de resultado para o modo padrão (365d/5Y)"""
    try:
        # (*** MUDANÇA v1.6: Unpack de 16 params ***)
        b4,s4,b8,s8,b1,s1, \
        sl_up_p, sl_up_amt, sl_down_p, sl_down_amt, \
        tp_p, tp_after_p, tp_sell_p, \
        tp_ema_pct, tp_ema_amt, ema_len = res[0:16]
        
        # (*** MUDANÇA v1.5: SL Bifurcado - String de SL ***)
        sl_up_str = f"SL-UP {sl_up_p:.1f}%@{sl_up_amt:.0f}%" if sl_up_p is not None else "SL-UP None"
        sl_down_str = f"SL-DN {sl_down_p:.1f}%@{sl_down_amt:.0f}%" if sl_down_p is not None else "SL-DN None"
        sl_str = f"{sl_up_str} | {sl_down_str}"
        # (*** FIM DA MUDANÇA ***)
        
        tp_str = f"TP {tp_p:.1f}" if tp_p is not None else "TP None"
        # (*** MUDANÇA v1.6: String TP-After ***)
        tp_after_str = f"TP-Aft {tp_after_p:.1f}%@{tp_sell_p:.1f}%" if (tp_after_p is not None and tp_sell_p is not None) else "TP-Aft None"
        
        tp_ema_str = f"EMA{ema_len} {tp_ema_pct:.1f}%@{tp_ema_amt:.1f}%" if (tp_ema_pct is not None and tp_ema_amt is not None) else ""
        
        # (*** MUDANÇA v1.6: Índices de métricas ***)
        # --- Métricas 365d ---
        pct365 = res[16]      # <--- MUDANÇA DE [14]
        metrics_365 = res[17] # <--- MUDANÇA DE [15]
        
        # (*** MUDANÇA v1.9: 23 métricas ***)
        b4h_e, b8h_e, b1d_e, s4h_e, s8h_e, s1d_e, \
        b4h_s, b8h_s, b1d_s, s4h_s, s8h_s, s1d_s, \
        b4h_i, b8h_i, b1d_i, s4h_i, s8h_i, s1d_i, \
        sl, tp, tp_after, ema, tbrs = metrics_365[0:23] 
        
        metrics_365_str = (
            f"[B 4H:{b4h_e}/{b4h_s}({b4h_i}) 8H:{b8h_e}/{b8h_s}({b8h_i}) 1D:{b1d_e}/{b1d_s}({b1d_i}) | "
            f"S(Sig) 4H:{s4h_e}/{s4h_s}({s4h_i}) 8H:{s8h_e}/{s8h_s}({s8h_i}) 1D:{s1d_e}/{s1d_s}({s1d_i}) | "
            f"S(Pasv):[SL:{sl} TP:{tp} TP-A:{tp_after} EMA:{ema} TBRS:{tbrs}]" # <--- MUDANÇA (TBRS)
        )

        # --- Métricas 5Y ---
        pct5y = res[18]       # <--- MUDANÇA DE [16]
        metrics_5y = res[19]  # <--- MUDANÇA DE [17]
        # (*** FIM DA MUDANÇA ***)
        
        # (*** MUDANÇA v1.9: 23 métricas ***)
        b4h_e, b8h_e, b1d_e, s4h_e, s8h_e, s1d_e, \
        b4h_s, b8h_s, b1d_s, s4h_s, s8h_s, s1d_s, \
        b4h_i, b8h_i, b1d_i, s4h_i, s8h_i, s1d_i, \
        sl, tp, tp_after, ema, tbrs = metrics_5y[0:23]
        
        metrics_5y_str = (
            f"[B 4H:{b4h_e}/{b4h_s}({b4h_i}) 8H:{b8h_e}/{b8h_s}({b8h_i}) 1D:{b1d_e}/{b1d_s}({b1d_i}) | "
            f"S(Sig) 4H:{s4h_e}/{s4h_s}({s4h_i}) 8H:{s8h_e}/{s8h_s}({s8h_i}) 1D:{s1d_e}/{s1d_s}({s1d_i}) | "
            f"S(Pasv):[SL:{sl} TP:{tp} TP-A:{tp_after} EMA:{ema} TBRS:{tbrs}]" # <--- MUDANÇA (TBRS)
        )
        
        return (f"{mark}{label:<10} B/S (4H {b4}/{s4} | 8H {b8}/{s8} | 1D {b1}/{s1}) | {sl_str} | {tp_str} | {tp_after_str} | {tp_ema_str} | "
                f"365d {pct365:+.2f}% {metrics_365_str} | "
                f"5y {pct5y:+.2f}% {metrics_5y_str}{RESET}")
    except Exception as e:
        return f"[ERRO AO FORMATAR LINHA: {e}] {res}"

def format_result_line_custom(res, label, mark, days_label):
    """Formata uma linha de resultado para o modo custom (--date)
    Retorna uma lista com duas linhas: [linha_percentual, linha_detalhes]
    """
    try:
        # (*** MUDANÇA v1.6: Unpack de 16 params ***)
        b4,s4,b8,s8,b1,s1, \
        sl_up_p, sl_up_amt, sl_down_p, sl_down_amt, \
        tp_p, tp_after_p, tp_sell_p, \
        tp_ema_pct, tp_ema_amt, ema_len = res[0:16]
        
        # (*** MUDANÇA v1.5: SL Bifurcado - String de SL ***)
        sl_up_str = f"SL-UP {sl_up_p:.1f}%@{sl_up_amt:.0f}%" if sl_up_p is not None else "SL-UP None"
        sl_down_str = f"SL-DN {sl_down_p:.1f}%@{sl_down_amt:.0f}%" if sl_down_p is not None else "SL-DN None"
        sl_str = f"{sl_up_str} | {sl_down_str}"
        # (*** FIM DA MUDANÇA ***)
        
        tp_str = f"TP {tp_p:.1f}" if tp_p is not None else "TP None"
        # (*** MUDANÇA v1.6: String TP-After ***)
        tp_after_str = f"TP-Aft {tp_after_p:.1f}%@{tp_sell_p:.1f}%" if (tp_after_p is not None and tp_sell_p is not None) else "TP-Aft None"
        
        tp_ema_str = f"EMA{ema_len} {tp_ema_pct:.1f}%@{tp_ema_amt:.1f}%" if (tp_ema_pct is not None and tp_ema_amt is not None) else ""
        
        # (*** MUDANÇA v1.6: Índices de métricas ***)
        # --- Métricas Custom ---
        pct_custom = res[16]     # <--- MUDANÇA DE [14]
        metrics_custom = res[17] # <--- MUDANÇA DE [15]
        # (*** FIM DA MUDANÇA ***)
        
        # (*** MUDANÇA v1.9: 23 métricas ***)
        b4h_e, b8h_e, b1d_e, s4h_e, s8h_e, s1d_e, \
        b4h_s, b8h_s, b1d_s, s4h_s, s8h_s, s1d_s, \
        b4h_i, b8h_i, b1d_i, s4h_i, s8h_i, s1d_i, \
        sl, tp, tp_after, ema, tbrs = metrics_custom[0:23]

        metrics_str = (
            f"[B 4H:{b4h_e}/{b4h_s}({b4h_i}) 8H:{b8h_e}/{b8h_s}({b8h_i}) 1D:{b1d_e}/{b1d_s}({b1d_i}) | "
            f"S(Sig) 4H:{s4h_e}/{s4h_s}({s4h_i}) 8H:{s8h_e}/{s8h_s}({s8h_i}) 1D:{s1d_e}/{s1d_s}({s1d_i}) | "
            f"S(Pasv):[SL:{sl} TP:{tp} TP-A:{tp_after} EMA:{ema} TBRS:{tbrs}]" # <--- MUDANÇA (TBRS)
        )
        
        # Remove colchetes do label se presente (ex: "[BEST 1400D]" -> "BEST 1400D")
        clean_label = label.replace("[", "").replace("]", "")
        
        # Linha 1: Percentual (formato: "BEST 1400D - +100.00%")
        # O label já contém o período, então não precisa adicionar novamente
        linha1 = f"{clean_label} - {pct_custom:+.2f}%"
        
        # Linha 2: Detalhes (formato: "B/S (4H ...) | ...")
        linha2 = f"B/S (4H {b4}/{s4} | 8H {b8}/{s8} | 1D {b1}/{s1}) | {sl_str} | {tp_str} | {tp_after_str} | {tp_ema_str} | {metrics_str}"
        
        return [linha1, linha2]
    except Exception as e:
        return [f"[ERRO AO FORMATAR LINHA: {e}]", str(res)]