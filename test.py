#!/usr/bin/env python3
# test.py — v3.5 (Testa regressão com TBRS DESLIGADO)

import subprocess
import sys
import re
import numpy as np  # usado em np.isclose

# ========= 1) COMANDO MULTI-DATE =========
# (*** Flag --two-bar-reversal-stop REMOVIDO para testar o baseline ***)
COMMAND_TO_RUN = [
    "python", ".\\sweep.py",
    "--file", ".\\BTCUSDT1D.csv",
    "--file-4h", ".\\BTCUSDT240.csv",
    "--file-8h", ".\\BTCUSDT480.csv",
    "--stops-on-candle", "1D",
    "--buy4h", "10.0",
    "--buy1d", "25.0",
    "--sell4h", "0.0",
    "--buy8h", "0.0",
    "--sell8h", "0.0",
    "--sell1d", "0.0",
    "--sl-up-pct", "2.0",
    "--sl-up-amount", "100",
    "--sl-down-pct", "3.0",
    "--sl-down-amount", "100",
    "--tp-pct", "100.0",
    "--emas", "20",
    "--tp-ema-pct", "100.0",
    "--tp-ema-amount", "30.0",
    "--workers", "1",
    "--print-mode", "all",
    "--date", "1400", "400",
]

# ========= 2) EXPECTATIVAS =========

# (Parâmetros idênticos ao baseline)
EXPECTED_PARAMS_STR = (
    "B/S (4H 10.0/0.0 | 8H 0.0/0.0 | 1D 25.0/0.0) | "
    "SL-UP 2.0%@100% | SL-DN 3.0%@100% | TP 100.0 | "
    "TP-Aft None | EMA20 100.0%@30.0%"
)

# (Tolerância pequena para diferenças de arredondamento)
PNL_ATOL = 0.2

# (*** Valores ESPERADOS são os do baseline original ***)
EXPECTED_RESULTS = [
    {
        "period": "1400",
        "label_str": "[BEST 1400D]",
        "expected_pnl": 600.99,    # <--- Baseline
        "expected_sl": 41,        # <--- Baseline
        "expected_tp": 2,
        "expected_tp_after": 0,
        "expected_ema": 0,
        "expected_tbrs": 0,       # <--- Deve ser 0
    },
    {
        "period": "400",
        "label_str": "[BEST 400D]",
        "expected_pnl": 71.14,
        "expected_sl": 1,
        "expected_tp": 0,
        "expected_tp_after": 0,
        "expected_ema": 0,
        "expected_tbrs": 0,       # <--- Deve ser 0
    }
]

ANSI_RE = re.compile(r'\x1B\[[0-?]*[ -/]*[@-~]')

def clean_text(text: str) -> str:
    cleaned = ANSI_RE.sub('', text)
    return " ".join(cleaned.split())

def run_test():
    print(f"[TESTE] Regressão Multi-Date (TBRS OFF) ({', '.join([d['period']+'d' for d in EXPECTED_RESULTS])})…")
    print("Comando:\n  " + " ".join(COMMAND_TO_RUN) + "\n")

    try:
        result = subprocess.run(
            COMMAND_TO_RUN,
            capture_output=True,
            text=True,
            check=True,
            encoding="utf-8",
            errors="replace",
        )
    except subprocess.CalledProcessError as e:
        print("[FALHA] sweep.py retornou erro.")
        print("STDERR:\n" + (e.stderr or ""))
        sys.exit(1)
    except FileNotFoundError:
        print("[FALHA] 'python' ou '.\\sweep.py' não encontrado.")
        sys.exit(1)

    stdout = result.stdout or ""
    print("--- Saída do sweep.py (SUMÁRIO) ---")
    summary_part = stdout.split("===== SUMÁRIO DA EXECUÇÃO =====")[-1]
    print(summary_part.strip())
    print("----------------------------------\n")

    errors = []
    success_details = []

    for expected in EXPECTED_RESULTS:
        period = expected["period"]
        label_str = expected["label_str"]
        print(f"[TESTE] Validando Período {period}d...")

        # 1) Encontrar a linha com os parâmetros esperados
        matched_line = None
        for raw in stdout.splitlines():
            line = clean_text(raw)
            if label_str in line and EXPECTED_PARAMS_STR in line:
                matched_line = line
                break

        if not matched_line:
            errors.append(f"({period}d) Não encontrei linha de resultado com o Label '{label_str}' E os Parâmetros '{EXPECTED_PARAMS_STR}'")
            continue

        print(f"[INFO] Linha {period}d encontrada:")
        print("  " + matched_line + "\n")

        line_errors = []

        # 2) Extrair PnL do período (ex: "1400d +600.99%")
        pnl_m = re.search(rf"{expected['period']}d ([+-]\d+\.\d+)%", matched_line)
        if not pnl_m:
            line_errors.append(f"({period}d) Não foi possível extrair PnL (formato '{expected['period']}d +XX.XX%').")
        else:
            pnl_val = float(pnl_m.group(1))
            if not np.isclose(pnl_val, expected["expected_pnl"], atol=PNL_ATOL):
                line_errors.append(f"({period}d) PnL divergente! Esperado {expected['expected_pnl']}%, encontrado {pnl_val}% (±{PNL_ATOL}).")

        # 3) Extrair contagens passivas
        # (*** MUDANÇA: Regex atualizada para ler as 5 métricas ***)
        c_m = re.search(r"S\(Pasv\):\[SL:(\d+) TP:(\d+) TP-A:(\d+) EMA:(\d+) TBRS:(\d+)\]", matched_line)
        
        if not c_m:
            # (*** MUDANÇA: Mensagem de erro atualizada ***)
            line_errors.append(f"({period}d) Não foi possível extrair S(Pasv) no formato 'SL:XX TP:YY TP-A:ZZ EMA:WW TBRS:ZZ'.")
        else:
            sl_cnt = int(c_m.group(1))
            tp_cnt = int(c_m.group(2))
            tp_after_cnt = int(c_m.group(3))
            ema_cnt = int(c_m.group(4))
            tbrs_cnt = int(c_m.group(5)) # <--- Nova métrica lida
            
            if sl_cnt != expected["expected_sl"]:
                line_errors.append(f"({period}d) SL Count divergente! Esperado {expected['expected_sl']}, encontrado {sl_cnt}.")
            if tp_cnt != expected["expected_tp"]:
                line_errors.append(f"({period}d) TP Count divergente! Esperado {expected['expected_tp']}, encontrado {tp_cnt}.")
            
            if tp_after_cnt != expected["expected_tp_after"]:
                line_errors.append(f"({period}d) TP-After Count divergente! Esperado {expected['expected_tp_after']}, encontrado {tp_after_cnt}.")

            if ema_cnt != expected["expected_ema"]:
                line_errors.append(f"({period}d) EMA Count divergente! Esperado {expected['expected_ema']}, encontrado {ema_cnt}.")

            # (*** MUDANÇA: Validação da nova métrica (deve ser 0) ***)
            if tbrs_cnt != expected["expected_tbrs"]:
                line_errors.append(f"({period}d) TBRS Count divergente! Esperado {expected['expected_tbrs']}, encontrado {tbrs_cnt}.")
        
        if line_errors:
            errors.extend(line_errors)
        else:
            # (*** MUDANÇA: Log de sucesso atualizado para 5 métricas ***)
            success_details.append(
                f"  - Período {period}d: PnL {expected['expected_pnl']}% (±{PNL_ATOL}) | "
                f"S(Pasv) SL/TP/TP-A/EMA/TBRS = {expected['expected_sl']}/{expected['expected_tp']}/{expected['expected_tp_after']}/{expected['expected_ema']}/{expected['expected_tbrs']}"
            )

    # (*** MKA: Reporte Final ***)
    if errors:
        print("\n[FALHA] Regressão falhou:")
        for e in errors:
            print("  - " + e)
        sys.exit(1)
    else:
        print("\n[SUCESSO] Todos os períodos OK!")
        for s in success_details:
            print(s)
        sys.exit(0)

if __name__ == "__main__":
    run_test()