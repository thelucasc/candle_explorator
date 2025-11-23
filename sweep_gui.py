#!/usr/bin/env python3
"""
Frontend GUI para o sweep.py
Interface gráfica para configurar e executar o sweep de parâmetros
"""

import tkinter as tk
from tkinter import ttk, scrolledtext, filedialog, messagebox
import subprocess
import sys
import threading
import os
import re
import shlex
from datetime import datetime
from typing import Dict, List, Optional, Tuple


class SweepGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Sweep Parameter Configuration")
        self.root.geometry("1400x900")
        
        # Variáveis de controle
        self.process = None
        self.is_running = False
        self.showing_winners = False
        
        # Criar interface
        self.create_widgets()
        # Carrega o último comando após criar os widgets
        self.root.after(100, self.load_last_command)  # Pequeno delay para garantir que widgets estão prontos
        self.load_log_file()
        
    def create_widgets(self):
        # Frame principal com paned window para dividir em duas partes
        main_paned = ttk.PanedWindow(self.root, orient=tk.HORIZONTAL)
        main_paned.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Painel esquerdo: Parâmetros e Logs
        left_frame = ttk.Frame(main_paned)
        main_paned.add(left_frame, weight=2)
        
        # Painel direito: Log de resultados
        right_frame = ttk.Frame(main_paned)
        main_paned.add(right_frame, weight=1)
        
        # ========== PAINEL ESQUERDO ==========
        # Frame para status do teste (no topo)
        test_status_frame = ttk.Frame(left_frame)
        test_status_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Label(test_status_frame, text="Status do Teste:", font=("Arial", 9, "bold")).pack(side=tk.LEFT, padx=5)
        self.test_status_label = ttk.Label(
            test_status_frame, 
            text="Executando...", 
            font=("Arial", 9),
            foreground="blue"
        )
        self.test_status_label.pack(side=tk.LEFT, padx=5)
        
        # Executa o teste automaticamente ao iniciar o GUI
        self._run_test_on_startup()
        
        # Notebook para tabs (Parâmetros e Logs)
        left_notebook = ttk.Notebook(left_frame)
        left_notebook.pack(fill=tk.BOTH, expand=True)
        
        # Tab 1: Parâmetros
        params_frame = ttk.Frame(left_notebook)
        left_notebook.add(params_frame, text="Parâmetros")
        
        # Scrollable frame para parâmetros
        canvas = tk.Canvas(params_frame)
        scrollbar = ttk.Scrollbar(params_frame, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)
        
        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
        # Variáveis para os parâmetros
        self.vars = {}
        self.create_parameter_widgets(scrollable_frame)
        
        # Tab 2: Parâmetros dos Sinais de Compra
        signals_frame = ttk.Frame(left_notebook)
        left_notebook.add(signals_frame, text="Sinais de Compra")
        
        # Scrollable frame para parâmetros dos sinais
        canvas2 = tk.Canvas(signals_frame)
        scrollbar2 = ttk.Scrollbar(signals_frame, orient="vertical", command=canvas2.yview)
        scrollable_frame2 = ttk.Frame(canvas2)
        
        scrollable_frame2.bind(
            "<Configure>",
            lambda e: canvas2.configure(scrollregion=canvas2.bbox("all"))
        )
        
        canvas2.create_window((0, 0), window=scrollable_frame2, anchor="nw")
        canvas2.configure(yscrollcommand=scrollbar2.set)
        
        canvas2.pack(side="left", fill="both", expand=True)
        scrollbar2.pack(side="right", fill="y")
        
        self.create_signal_parameter_widgets(scrollable_frame2)
        
        # Botão de execução
        button_frame = ttk.Frame(left_frame)
        button_frame.pack(fill=tk.X, padx=5, pady=5)
        
        self.run_button = ttk.Button(
            button_frame, 
            text="▶ Executar Sweep", 
            command=self.run_sweep,
            style="Accent.TButton"
        )
        self.run_button.pack(side=tk.LEFT, padx=5)
        
        self.stop_button = ttk.Button(
            button_frame,
            text="⏹ Parar",
            command=self.stop_sweep,
            state=tk.DISABLED
        )
        self.stop_button.pack(side=tk.LEFT, padx=5)
        
        # Tab 3: Logs do Terminal
        logs_frame = ttk.Frame(left_notebook)
        left_notebook.add(logs_frame, text="Logs do Terminal")
        
        self.log_text = scrolledtext.ScrolledText(
            logs_frame,
            wrap=tk.WORD,
            font=("Consolas", 9),
            bg="#1e1e1e",
            fg="#d4d4d4",
            insertbackground="#d4d4d4"
        )
        self.log_text.pack(fill=tk.BOTH, expand=True)
        
        # ========== PAINEL DIREITO ==========
        # Frame para o log de resultados
        right_header = ttk.Frame(right_frame)
        right_header.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Label(right_header, text="Resultados (Sweep Log)", font=("Arial", 12, "bold")).pack(side=tk.LEFT)
        
        self.show_winners_btn = ttk.Button(
            right_header, 
            text="📊 Vencedores Overall", 
            command=self.toggle_view
        )
        self.show_winners_btn.pack(side=tk.RIGHT, padx=2)
        
        refresh_btn = ttk.Button(right_header, text="🔄 Atualizar", command=self.load_log_file)
        refresh_btn.pack(side=tk.RIGHT, padx=2)
        
        # Text widget para mostrar o log
        self.results_text = scrolledtext.ScrolledText(
            right_frame,
            wrap=tk.WORD,
            font=("Consolas", 9),
            bg="#1e1e1e",
            fg="#d4d4d4",
            insertbackground="#d4d4d4"
        )
        self.results_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
    def create_parameter_widgets(self, parent):
        """Cria os widgets de parâmetros"""
        row = 0
        
        # Função helper para criar linha de parâmetro
        def add_param(label, widget_type, var_name, default=None, **kwargs):
            nonlocal row
            frame = ttk.Frame(parent)
            frame.grid(row=row, column=0, sticky="ew", padx=5, pady=2)
            parent.columnconfigure(0, weight=1)
            
            ttk.Label(frame, text=label, width=25, anchor="w").pack(side=tk.LEFT, padx=5)
            
            if widget_type == "entry":
                var = tk.StringVar(value=str(default) if default is not None else "")
                widget = ttk.Entry(frame, textvariable=var, width=40)
                widget.pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)
            elif widget_type == "entry_int":
                var = tk.StringVar(value=str(default) if default is not None else "")
                widget = ttk.Entry(frame, textvariable=var, width=40)
                widget.pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)
            elif widget_type == "entry_float":
                var = tk.StringVar(value=str(default) if default is not None else "")
                widget = ttk.Entry(frame, textvariable=var, width=40)
                widget.pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)
            elif widget_type == "checkbox":
                var = tk.BooleanVar(value=default if default is not None else False)
                widget = ttk.Checkbutton(frame, variable=var)
                widget.pack(side=tk.LEFT, padx=5)
            elif widget_type == "combobox":
                var = tk.StringVar(value=default if default is not None else "")
                widget = ttk.Combobox(frame, textvariable=var, width=37, state="readonly")
                if "values" in kwargs:
                    widget["values"] = kwargs["values"]
                widget.pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)
            elif widget_type == "file":
                var = tk.StringVar(value=str(default) if default is not None else "")
                entry = ttk.Entry(frame, textvariable=var, width=30)
                entry.pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)
                btn = ttk.Button(frame, text="...", command=lambda: self.browse_file(var))
                btn.pack(side=tk.LEFT, padx=2)
                widget = entry  # Para armazenar a referência
            else:
                var = None
                widget = None
            
            self.vars[var_name] = var
            row += 1
            return var
        
        # Separador: Arquivos
        ttk.Separator(parent, orient=tk.HORIZONTAL).grid(row=row, column=0, sticky="ew", padx=5, pady=10)
        row += 1
        ttk.Label(parent, text="Arquivos CSV", font=("Arial", 10, "bold")).grid(row=row, column=0, sticky="w", padx=5)
        row += 1
        
        add_param("File (1D):", "file", "file", "BTC3D.csv")
        add_param("File 4H:", "file", "file_4h", "BTCUSDT240.csv")
        add_param("File 8H:", "file", "file_8h", "BTCUSDT480.csv")
        
        # Separador: Períodos e Configuração
        ttk.Separator(parent, orient=tk.HORIZONTAL).grid(row=row, column=0, sticky="ew", padx=5, pady=10)
        row += 1
        ttk.Label(parent, text="Períodos e Configuração", font=("Arial", 10, "bold")).grid(row=row, column=0, sticky="w", padx=5)
        row += 1
        
        add_param("Date (dias):", "entry", "date", "1400 400 1800 310 1400 590")
        add_param("Step:", "entry_int", "step", "10")
        add_param("Workers:", "entry_int", "workers", "7")
        add_param("Print Mode:", "combobox", "print_mode", "winners", values=["all", "winners", "pos"])
        add_param("Start Pct:", "entry_float", "start_pct", "0.0")
        add_param("Log File:", "entry", "log_file", "sweep_log.txt")
        
        # Separador: Grids Buy/Sell
        ttk.Separator(parent, orient=tk.HORIZONTAL).grid(row=row, column=0, sticky="ew", padx=5, pady=10)
        row += 1
        ttk.Label(parent, text="Grids Buy/Sell (valores ou 'step')", font=("Arial", 10, "bold")).grid(row=row, column=0, sticky="w", padx=5)
        row += 1
        
        add_param("Buy 1D:", "entry", "buy1d", "step")
        add_param("Sell 1D:", "entry", "sell1d", "")
        add_param("Buy 4H:", "entry", "buy4h", "")
        add_param("Sell 4H:", "entry", "sell4h", "")
        add_param("Buy 8H:", "entry", "buy8h", "")
        add_param("Sell 8H:", "entry", "sell8h", "")
        
        # Separador: Stop Loss
        ttk.Separator(parent, orient=tk.HORIZONTAL).grid(row=row, column=0, sticky="ew", padx=5, pady=10)
        row += 1
        ttk.Label(parent, text="Stop Loss", font=("Arial", 10, "bold")).grid(row=row, column=0, sticky="w", padx=5)
        row += 1
        
        add_param("SL Up Pct:", "entry", "sl_up_pct", "2.5 4")
        add_param("SL Up Amount:", "entry", "sl_up_amount", "70 100")
        add_param("SL Down Pct:", "entry", "sl_down_pct", "2.5 3 4 5 6")
        add_param("SL Down Amount:", "entry", "sl_down_amount", "20 80 100")
        
        # Separador: Take Profit
        ttk.Separator(parent, orient=tk.HORIZONTAL).grid(row=row, column=0, sticky="ew", padx=5, pady=10)
        row += 1
        ttk.Label(parent, text="Take Profit", font=("Arial", 10, "bold")).grid(row=row, column=0, sticky="w", padx=5)
        row += 1
        
        add_param("TP Pct:", "entry", "tp_pct", "50 100.0")
        add_param("TP After Pct:", "entry", "take_profit_after_percentage", "15 30 40 60")
        add_param("TP Percentage:", "entry", "take_profit_percentage", "30")
        add_param("TP EMA Pct:", "entry", "tp_ema_pct", "20 30 50")
        add_param("TP EMA Amount:", "entry", "tp_ema_amount", "10 30")
        
        # Separador: EMAs
        ttk.Separator(parent, orient=tk.HORIZONTAL).grid(row=row, column=0, sticky="ew", padx=5, pady=10)
        row += 1
        ttk.Label(parent, text="EMAs", font=("Arial", 10, "bold")).grid(row=row, column=0, sticky="w", padx=5)
        row += 1
        
        add_param("EMAs:", "entry", "emas", "20 40 50 100")
        
        # Separador: Opções Avançadas
        ttk.Separator(parent, orient=tk.HORIZONTAL).grid(row=row, column=0, sticky="ew", padx=5, pady=10)
        row += 1
        ttk.Label(parent, text="Opções Avançadas", font=("Arial", 10, "bold")).grid(row=row, column=0, sticky="w", padx=5)
        row += 1
        
        add_param("Initial Capital:", "entry_float", "initial_capital", "100000.0")
        add_param("Commission BPS:", "entry_float", "commission_bps", "1.0")
        add_param("Max Exposure Pct:", "entry_float", "max_exposure_pct", "100.0")
        add_param("Stops On Candle:", "entry", "stops_on_candle", "1D")
        add_param("Stop Loss Signal:", "checkbox", "stop_loss_signal", False)
        add_param("Two Bar Reversal Stop:", "checkbox", "two_bar_reversal_stop", False)
    
    def create_signal_parameter_widgets(self, parent):
        """Cria os widgets de parâmetros dos sinais de compra"""
        row = 0
        
        # Função helper para criar linha de parâmetro
        def add_param(label, widget_type, var_name, default=None, help_text=""):
            nonlocal row
            frame = ttk.Frame(parent)
            frame.grid(row=row, column=0, sticky="ew", padx=5, pady=2)
            parent.columnconfigure(0, weight=1)
            
            ttk.Label(frame, text=label, width=35, anchor="w").pack(side=tk.LEFT, padx=5)
            
            if widget_type == "entry":
                var = tk.StringVar(value=str(default) if default is not None else "")
                widget = ttk.Entry(frame, textvariable=var, width=40)
                widget.pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)
            elif widget_type == "entry_int":
                var = tk.StringVar(value=str(default) if default is not None else "")
                widget = ttk.Entry(frame, textvariable=var, width=40)
                widget.pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)
            elif widget_type == "entry_float":
                var = tk.StringVar(value=str(default) if default is not None else "")
                widget = ttk.Entry(frame, textvariable=var, width=40)
                widget.pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)
            else:
                var = None
                widget = None
            
            if help_text:
                help_label = ttk.Label(frame, text=help_text, font=("Arial", 8), foreground="gray")
                help_label.pack(side=tk.LEFT, padx=5)
            
            self.vars[var_name] = var
            row += 1
            return var
        
        # Separador: Sinais 4H/8H
        ttk.Separator(parent, orient=tk.HORIZONTAL).grid(row=row, column=0, sticky="ew", padx=5, pady=10)
        row += 1
        ttk.Label(parent, text="Sinais de Compra 4H/8H", font=("Arial", 10, "bold")).grid(row=row, column=0, sticky="w", padx=5)
        row += 1
        
        add_param("SMA Length (4H/8H):", "entry_int", "signal_4h8h_sma_length", "20", 
                  help_text="(default: 20)")
        add_param("PIR Threshold (4H/8H):", "entry_float", "signal_4h8h_pir_threshold", "0.85",
                  help_text="(default: 0.85)")
        
        # Separador: Sinais 1D
        ttk.Separator(parent, orient=tk.HORIZONTAL).grid(row=row, column=0, sticky="ew", padx=5, pady=10)
        row += 1
        ttk.Label(parent, text="Sinais de Compra 1D", font=("Arial", 10, "bold")).grid(row=row, column=0, sticky="w", padx=5)
        row += 1
        
        add_param("SMA Length (1D):", "entry_int", "signal_1d_sma_length", "20",
                  help_text="(default: 20)")
        add_param("Trend Regime Threshold (1D):", "entry_float", "signal_1d_trend_regime_threshold", "0.002",
                  help_text="(default: 0.002 = 0.2%)")
        add_param("Trend Regime Tree Threshold (1D):", "entry_float", "signal_1d_trend_regime_tree_threshold", "1.5",
                  help_text="(default: 1.5)")
        add_param("Distância SMA Threshold (1D):", "entry_float", "signal_1d_dist_ma_threshold", "0.03",
                  help_text="(default: 0.03 = 3%)")
        add_param("RSI Length (1D):", "entry_int", "signal_1d_rsi_length", "14",
                  help_text="(default: 14)")
        add_param("RSI Threshold (1D):", "entry_float", "signal_1d_rsi_threshold", "60",
                  help_text="(default: 60)")
        add_param("PIR Threshold Anterior (1D):", "entry_float", "signal_1d_pir_threshold_prev", "0.60",
                  help_text="(default: 0.60)")
        add_param("PIR Threshold Confirmação (1D):", "entry_float", "signal_1d_pir_threshold_confirm", "0.40",
                  help_text="(default: 0.40)")
        
    def browse_file(self, var):
        """Abre diálogo para selecionar arquivo"""
        filename = filedialog.askopenfilename(
            title="Selecionar arquivo CSV",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
        )
        if filename:
            var.set(filename)
    
    def load_last_command(self):
        """Carrega os parâmetros do último comando executado do sweep_log.txt"""
        # Verifica se self.vars existe (pode não existir se chamado antes de create_widgets)
        if not hasattr(self, 'vars') or not self.vars:
            return
        
        log_file = self.vars.get("log_file", tk.StringVar(value="sweep_log.txt")).get()
        if not log_file:
            log_file = "sweep_log.txt"
        
        if not os.path.exists(log_file):
            # Tenta o arquivo padrão se o especificado não existir
            if log_file != "sweep_log.txt" and os.path.exists("sweep_log.txt"):
                log_file = "sweep_log.txt"
            else:
                return  # Não há log para carregar
        
        try:
            with open(log_file, "r", encoding="utf-8") as f:
                lines = f.readlines()
            
            # Procura a última linha "Comando:" de trás para frente
            last_command = None
            for line in reversed(lines):
                if line.startswith("Comando:"):
                    last_command = line.replace("Comando:", "").strip()
                    break
            
            if not last_command:
                # Debug: mostra se não encontrou comando
                print(f"[DEBUG] Não encontrou linha 'Comando:' no arquivo {log_file}")
                return  # Não encontrou comando
            
            # Debug: mostra o comando encontrado (apenas no console, não no GUI)
            # print(f"[DEBUG] Comando encontrado: {last_command[:100]}...")
            
            # Parse do comando usando shlex para lidar com espaços e aspas
            # No Windows, shlex pode ter problemas com caminhos .\arquivo.csv
            # Então primeiro fazemos um pré-processamento para proteger os caminhos
            try:
                # Substitui .\ por .\\ temporariamente para proteger durante o split
                protected = last_command.replace(".\\", ".__TEMP_BACKSLASH__")
                parts = shlex.split(protected)
                # Restaura os caminhos originais
                parts = [p.replace(".__TEMP_BACKSLASH__", ".\\") for p in parts]
            except:
                # Fallback: split simples se shlex falhar
                parts = last_command.split()
            
            # Remove "python" e caminhos do sweep.py do início
            # Pode ser "python", "python3", ".\\sweep.py", "sweep.py", etc.
            while parts:
                first = parts[0].lower()
                if ("python" in first or "sweep.py" in first or 
                    first.endswith(".py") or first == "python" or first == "python3"):
                    parts.pop(0)
                else:
                    break
            
            # Parse dos argumentos
            i = 0
            while i < len(parts):
                arg = parts[i]
                
                # Remove -- se presente
                if arg.startswith("--"):
                    arg_name = arg[2:].replace("-", "_")
                    
                    # Flags booleanos (não têm valor)
                    boolean_flags = ["stop_loss_signal", "two_bar_reversal_stop"]
                    if arg_name in boolean_flags:
                        if arg_name in self.vars:
                            self.vars[arg_name].set(True)
                        i += 1
                        continue
                    
                    # Argumentos que aceitam múltiplos valores
                    multi_value_args = [
                        "date", "buy1d", "sell1d", "buy4h", "sell4h", "buy8h", "sell8h",
                        "sl_up_pct", "sl_up_amount", "sl_down_pct", "sl_down_amount",
                        "tp_pct", "take_profit_after_percentage", "take_profit_percentage",
                        "tp_ema_pct", "tp_ema_amount", "emas", "stops_on_candle",
                        "signal_4h8h_sma_length", "signal_4h8h_pir_threshold",
                        "signal_1d_sma_length", "signal_1d_trend_regime_threshold",
                        "signal_1d_trend_regime_tree_threshold", "signal_1d_dist_ma_threshold",
                        "signal_1d_rsi_length", "signal_1d_rsi_threshold",
                        "signal_1d_pir_threshold_prev", "signal_1d_pir_threshold_confirm"
                    ]
                    
                    if arg_name in multi_value_args:
                        # Coleta todos os valores até o próximo --arg
                        values = []
                        i += 1
                        while i < len(parts) and not parts[i].startswith("--"):
                            values.append(parts[i])
                            i += 1
                        i -= 1  # Ajusta para o próximo loop
                        
                        if arg_name in self.vars:
                            self.vars[arg_name].set(" ".join(values))
                        # else:
                        #     print(f"[DEBUG] Campo não encontrado (multi): {arg_name}")
                    else:
                        # Argumento com valor único
                        i += 1
                        if i < len(parts) and not parts[i].startswith("--"):
                            value = parts[i]
                            
                            # Corrige caminhos que podem ter sido parseados incorretamente
                            # Se o valor começa com . seguido de letra (sem barra), adiciona .\
                            # Isso pode acontecer se shlex.split interpretou .\arquivo.csv como .arquivo.csv
                            if value.startswith(".") and len(value) > 1:
                                if not (value.startswith(".\\") or value.startswith("./")):
                                    # Se começa com . seguido de letra/número, provavelmente é .\arquivo
                                    if value[1].isalnum() or value[1] in ['_', '-']:
                                        value = ".\\" + value[1:]
                            
                            # Usa o nome do argumento diretamente (já convertido de - para _)
                            if arg_name in self.vars:
                                self.vars[arg_name].set(value)
                            # else:
                            #     print(f"[DEBUG] Campo não encontrado (single): {arg_name} = {value}")
                            # Debug: mostra se o campo não foi encontrado
                            # else:
                            #     print(f"[DEBUG] Campo não encontrado: {arg_name}")
                i += 1
                
        except Exception as e:
            # Loga o erro para debug, mas não interrompe a execução
            # print(f"[ERRO ao carregar último comando] {e}")
            # import traceback
            # traceback.print_exc()
            pass  # Silenciosamente ignora erros de parse
    
    def build_command(self) -> List[str]:
        """Constrói o comando para executar o sweep.py"""
        cmd = ["python", "sweep.py"]
        
        # Arquivos
        if self.vars["file"].get().strip():
            cmd.extend(["--file", self.vars["file"].get().strip()])
        if self.vars["file_4h"].get().strip():
            cmd.extend(["--file-4h", self.vars["file_4h"].get().strip()])
        if self.vars["file_8h"].get().strip():
            cmd.extend(["--file-8h", self.vars["file_8h"].get().strip()])
        
        # Períodos e Configuração
        if self.vars["date"].get().strip():
            cmd.extend(["--date"] + self.vars["date"].get().strip().split())
        if self.vars["step"].get().strip():
            cmd.extend(["--step", self.vars["step"].get().strip()])
        if self.vars["workers"].get().strip():
            cmd.extend(["--workers", self.vars["workers"].get().strip()])
        if self.vars["print_mode"].get().strip():
            cmd.extend(["--print-mode", self.vars["print_mode"].get().strip()])
        if self.vars["start_pct"].get().strip():
            cmd.extend(["--start-pct", self.vars["start_pct"].get().strip()])
        if self.vars["log_file"].get().strip():
            cmd.extend(["--log-file", self.vars["log_file"].get().strip()])
        
        # Grids Buy/Sell
        if self.vars["buy1d"].get().strip():
            cmd.extend(["--buy1d"] + self.vars["buy1d"].get().strip().split())
        if self.vars["sell1d"].get().strip():
            cmd.extend(["--sell1d"] + self.vars["sell1d"].get().strip().split())
        if self.vars["buy4h"].get().strip():
            cmd.extend(["--buy4h"] + self.vars["buy4h"].get().strip().split())
        if self.vars["sell4h"].get().strip():
            cmd.extend(["--sell4h"] + self.vars["sell4h"].get().strip().split())
        if self.vars["buy8h"].get().strip():
            cmd.extend(["--buy8h"] + self.vars["buy8h"].get().strip().split())
        if self.vars["sell8h"].get().strip():
            cmd.extend(["--sell8h"] + self.vars["sell8h"].get().strip().split())
        
        # Stop Loss
        if self.vars["sl_up_pct"].get().strip():
            cmd.extend(["--sl-up-pct"] + self.vars["sl_up_pct"].get().strip().split())
        if self.vars["sl_up_amount"].get().strip():
            cmd.extend(["--sl-up-amount"] + self.vars["sl_up_amount"].get().strip().split())
        if self.vars["sl_down_pct"].get().strip():
            cmd.extend(["--sl-down-pct"] + self.vars["sl_down_pct"].get().strip().split())
        if self.vars["sl_down_amount"].get().strip():
            cmd.extend(["--sl-down-amount"] + self.vars["sl_down_amount"].get().strip().split())
        
        # Take Profit
        if self.vars["tp_pct"].get().strip():
            cmd.extend(["--tp-pct"] + self.vars["tp_pct"].get().strip().split())
        if self.vars["take_profit_after_percentage"].get().strip():
            cmd.extend(["--take-profit-after-percentage"] + self.vars["take_profit_after_percentage"].get().strip().split())
        if self.vars["take_profit_percentage"].get().strip():
            cmd.extend(["--take-profit-percentage"] + self.vars["take_profit_percentage"].get().strip().split())
        if self.vars["tp_ema_pct"].get().strip():
            cmd.extend(["--tp-ema-pct"] + self.vars["tp_ema_pct"].get().strip().split())
        if self.vars["tp_ema_amount"].get().strip():
            cmd.extend(["--tp-ema-amount"] + self.vars["tp_ema_amount"].get().strip().split())
        
        # EMAs
        if self.vars["emas"].get().strip():
            cmd.extend(["--emas"] + self.vars["emas"].get().strip().split())
        
        # Parâmetros dos Sinais de Compra (4H/8H)
        if self.vars.get("signal_4h8h_sma_length") and self.vars["signal_4h8h_sma_length"].get().strip():
            cmd.extend(["--signal-4h8h-sma-length"] + self.vars["signal_4h8h_sma_length"].get().strip().split())
        if self.vars.get("signal_4h8h_pir_threshold") and self.vars["signal_4h8h_pir_threshold"].get().strip():
            cmd.extend(["--signal-4h8h-pir-threshold"] + self.vars["signal_4h8h_pir_threshold"].get().strip().split())
        
        # Parâmetros dos Sinais de Compra (1D)
        if self.vars.get("signal_1d_sma_length") and self.vars["signal_1d_sma_length"].get().strip():
            cmd.extend(["--signal-1d-sma-length"] + self.vars["signal_1d_sma_length"].get().strip().split())
        if self.vars.get("signal_1d_trend_regime_threshold") and self.vars["signal_1d_trend_regime_threshold"].get().strip():
            cmd.extend(["--signal-1d-trend-regime-threshold"] + self.vars["signal_1d_trend_regime_threshold"].get().strip().split())
        if self.vars.get("signal_1d_trend_regime_tree_threshold") and self.vars["signal_1d_trend_regime_tree_threshold"].get().strip():
            cmd.extend(["--signal-1d-trend-regime-tree-threshold"] + self.vars["signal_1d_trend_regime_tree_threshold"].get().strip().split())
        if self.vars.get("signal_1d_dist_ma_threshold") and self.vars["signal_1d_dist_ma_threshold"].get().strip():
            cmd.extend(["--signal-1d-dist-ma-threshold"] + self.vars["signal_1d_dist_ma_threshold"].get().strip().split())
        if self.vars.get("signal_1d_rsi_length") and self.vars["signal_1d_rsi_length"].get().strip():
            cmd.extend(["--signal-1d-rsi-length"] + self.vars["signal_1d_rsi_length"].get().strip().split())
        if self.vars.get("signal_1d_rsi_threshold") and self.vars["signal_1d_rsi_threshold"].get().strip():
            cmd.extend(["--signal-1d-rsi-threshold"] + self.vars["signal_1d_rsi_threshold"].get().strip().split())
        if self.vars.get("signal_1d_pir_threshold_prev") and self.vars["signal_1d_pir_threshold_prev"].get().strip():
            cmd.extend(["--signal-1d-pir-threshold-prev"] + self.vars["signal_1d_pir_threshold_prev"].get().strip().split())
        if self.vars.get("signal_1d_pir_threshold_confirm") and self.vars["signal_1d_pir_threshold_confirm"].get().strip():
            cmd.extend(["--signal-1d-pir-threshold-confirm"] + self.vars["signal_1d_pir_threshold_confirm"].get().strip().split())
        
        # Opções Avançadas
        if self.vars["initial_capital"].get().strip():
            cmd.extend(["--initial-capital", self.vars["initial_capital"].get().strip()])
        if self.vars["commission_bps"].get().strip():
            cmd.extend(["--commission-bps", self.vars["commission_bps"].get().strip()])
        if self.vars["max_exposure_pct"].get().strip():
            cmd.extend(["--max-exposure-pct", self.vars["max_exposure_pct"].get().strip()])
        if self.vars["stops_on_candle"].get().strip():
            cmd.extend(["--stops-on-candle"] + self.vars["stops_on_candle"].get().strip().split())
        if self.vars["stop_loss_signal"].get():
            cmd.append("--stop-loss-signal")
        if self.vars["two_bar_reversal_stop"].get():
            cmd.append("--two-bar-reversal-stop")
        
        return cmd
    
    def run_sweep(self):
        """Executa o sweep em uma thread separada"""
        if self.is_running:
            messagebox.showwarning("Aviso", "Já existe uma execução em andamento!")
            return
        
        cmd = self.build_command()
        self.log_text.delete(1.0, tk.END)
        self.log_text.insert(tk.END, f"Comando: {' '.join(cmd)}\n")
        self.log_text.insert(tk.END, "=" * 80 + "\n\n")
        
        self.is_running = True
        self.run_button.config(state=tk.DISABLED)
        self.stop_button.config(state=tk.NORMAL)
        
        # Executar em thread separada
        thread = threading.Thread(target=self._run_sweep_thread, args=(cmd,), daemon=True)
        thread.start()
    
    def _run_sweep_thread(self, cmd):
        """Thread que executa o sweep"""
        try:
            self.process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                universal_newlines=True,
                bufsize=1,
                encoding='utf-8',
                errors='replace'
            )
            
            # Ler output em tempo real
            for line in iter(self.process.stdout.readline, ''):
                if not self.is_running:
                    break
                if line:  # Só processa linhas não vazias
                    # Processa linha por linha para melhor detecção
                    self.root.after(0, self._append_log, line)
            
            self.process.wait()
            
            # Verifica se houve erro no processo
            if self.process.returncode != 0 and self.is_running:
                self.root.after(0, self._append_log, f"\n[ERRO] Processo terminou com código {self.process.returncode}\n")
                self.root.after(0, self._update_test_status, "ERRO na execução", "red")
            
            if self.is_running:
                self.root.after(0, self._sweep_finished)
        except MemoryError as e:
            self.root.after(0, self._append_log, f"\n[ERRO CRÍTICO] Estouro de memória: {str(e)}\n")
            self.root.after(0, self._update_test_status, "ERRO: Estouro de memória", "red")
            self.root.after(0, self._sweep_finished)
        except Exception as e:
            self.root.after(0, self._append_log, f"\n[ERRO] {str(e)}\n")
            error_msg = str(e)[:50] if len(str(e)) > 50 else str(e)
            self.root.after(0, self._update_test_status, f"ERRO: {error_msg}", "red")
            self.root.after(0, self._sweep_finished)
    
    def _append_log(self, text):
        """Adiciona texto ao log (thread-safe)"""
        self.log_text.insert(tk.END, text)
        self.log_text.see(tk.END)
    
    def _update_test_status(self, status_text, color):
        """Atualiza o label de status do teste"""
        self.test_status_label.config(text=status_text, foreground=color)
    
    def _run_test_on_startup(self):
        """Executa o test.py automaticamente quando o GUI inicia"""
        def run_test_thread():
            test_file = "test.py"
            self.root.after(0, self._append_log, f"[TEST] Iniciando execução do teste...\n")
            
            if not os.path.exists(test_file):
                self.root.after(0, self._append_log, f"[TEST] ERRO: {test_file} não encontrado!\n")
                self.root.after(0, self._update_test_status, "test.py não encontrado", "gray")
                return
            
            self.root.after(0, self._append_log, f"[TEST] Arquivo encontrado: {test_file}\n")
            self.root.after(0, self._append_log, f"[TEST] Executando: {sys.executable} {test_file}\n")
            
            try:
                start_time = datetime.now()
                self.root.after(0, self._append_log, f"[TEST] Início: {start_time.strftime('%H:%M:%S')}\n")
                
                # Usa Popen para poder monitorar em tempo real
                process = subprocess.Popen(
                    [sys.executable, test_file],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                    bufsize=1
                )
                
                self.root.after(0, self._append_log, f"[TEST] Processo iniciado (PID: {process.pid})\n")
                
                # Monitora o processo com timeout
                import time
                timeout_seconds = 300  # 5 minutos
                start_wait = time.time()
                
                # Lê output em tempo real
                stdout_lines = []
                stderr_lines = []
                
                # Lê output linha por linha até o processo terminar
                while True:
                    # Verifica timeout
                    elapsed = time.time() - start_wait
                    if elapsed > timeout_seconds:
                        self.root.after(0, self._append_log, f"[TEST] TIMEOUT após {timeout_seconds}s! Matando processo...\n")
                        process.kill()
                        raise subprocess.TimeoutExpired(process.args, timeout_seconds)
                    
                    # Verifica se processo terminou
                    returncode = process.poll()
                    if returncode is not None:
                        # Processo terminou, lê o restante do output
                        self.root.after(0, self._append_log, f"[TEST] Processo terminou (return code: {returncode})\n")
                        break
                    
                    # Lê linhas disponíveis (não bloqueia se não houver)
                    try:
                        if process.stdout:
                            # Tenta ler uma linha (pode retornar vazio se não houver)
                            line = process.stdout.readline()
                            if line:
                                stdout_lines.append(line)
                                # Mostra todas as linhas em tempo real
                                self.root.after(0, self._append_log, f"[TEST] {line}")
                                # Verifica se encontrou indicadores de sucesso/falha
                                if "[SUCESSO]" in line or "Todos os períodos OK" in line:
                                    self.root.after(0, self._append_log, f"[TEST] Encontrou indicador de sucesso!\n")
                                elif "[FALHA]" in line or "ERRO" in line.upper():
                                    self.root.after(0, self._append_log, f"[TEST] Encontrou indicador de falha!\n")
                    except:
                        pass
                    
                    try:
                        if process.stderr:
                            line = process.stderr.readline()
                            if line:
                                stderr_lines.append(line)
                                self.root.after(0, self._append_log, f"[TEST] STDERR: {line}")
                    except:
                        pass
                    
                    time.sleep(0.1)  # Pequeno delay para não consumir CPU
                
                # Processo terminou, pega o restante do output (se houver)
                try:
                    remaining_stdout, remaining_stderr = process.communicate(timeout=2)
                    if remaining_stdout:
                        stdout_lines.append(remaining_stdout)
                        # Mostra o restante
                        for line in remaining_stdout.split('\n'):
                            if line.strip():
                                self.root.after(0, self._append_log, f"[TEST] {line}\n")
                    if remaining_stderr:
                        stderr_lines.append(remaining_stderr)
                except:
                    pass
                
                stdout = ''.join(stdout_lines)
                stderr = ''.join(stderr_lines)
                returncode = process.returncode
                
                end_time = datetime.now()
                duration = end_time - start_time
                self.root.after(0, self._append_log, f"[TEST] Término: {end_time.strftime('%H:%M:%S')} (duração: {duration})\n")
                self.root.after(0, self._append_log, f"[TEST] Return code: {returncode}\n")
                
                # Verifica se o teste passou
                combined = stdout + stderr
                
                self.root.after(0, self._append_log, f"[TEST] Tamanho stdout: {len(stdout)} caracteres\n")
                self.root.after(0, self._append_log, f"[TEST] Tamanho stderr: {len(stderr)} caracteres\n")
                
                # Mostra primeiras e últimas linhas do output
                if stdout:
                    stdout_lines = stdout.split('\n')
                    self.root.after(0, self._append_log, f"[TEST] Primeiras 3 linhas do stdout:\n")
                    for line in stdout_lines[:3]:
                        if line.strip():
                            self.root.after(0, self._append_log, f"  {line}\n")
                    self.root.after(0, self._append_log, f"[TEST] Últimas 3 linhas do stdout:\n")
                    for line in stdout_lines[-3:]:
                        if line.strip():
                            self.root.after(0, self._append_log, f"  {line}\n")
                
                if stderr:
                    self.root.after(0, self._append_log, f"[TEST] Stderr:\n{stderr[:500]}\n")
                
                test_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                
                test_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                
                if "[SUCESSO]" in combined or "Todos os períodos OK" in combined:
                    status_str = f"TEST: {test_time} - PASS"
                    self.root.after(0, self._append_log, f"[TEST] Status: PASS (encontrou [SUCESSO] ou 'Todos os períodos OK')\n")
                    self.root.after(0, self._update_test_status, status_str, "green")
                elif "[FALHA]" in combined or "ERRO" in combined.upper() or returncode != 0:
                    status_str = f"TEST: {test_time} - FAIL"
                    self.root.after(0, self._append_log, f"[TEST] Status: FAIL (encontrou [FALHA], ERRO ou return code != 0)\n")
                    self.root.after(0, self._update_test_status, status_str, "red")
                else:
                    # Se não encontrar indicadores claros, usa o return code
                    if returncode == 0:
                        status_str = f"TEST: {test_time} - PASS"
                        self.root.after(0, self._append_log, f"[TEST] Status: PASS (return code == 0, sem indicadores claros)\n")
                        self.root.after(0, self._update_test_status, status_str, "green")
                    else:
                        status_str = f"TEST: {test_time} - FAIL"
                        self.root.after(0, self._append_log, f"[TEST] Status: FAIL (return code != 0, sem indicadores claros)\n")
                        self.root.after(0, self._update_test_status, status_str, "red")
                        
            except subprocess.TimeoutExpired:
                test_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                status_str = f"TEST: {test_time} - TIMEOUT"
                self.root.after(0, self._append_log, f"[TEST] ERRO: Timeout após 5 minutos!\n")
                self.root.after(0, self._update_test_status, status_str, "red")
            except Exception as e:
                test_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                status_str = f"TEST: {test_time} - ERRO"
                self.root.after(0, self._append_log, f"[TEST] ERRO: {str(e)}\n")
                self.root.after(0, self._append_log, f"[TEST] Tipo do erro: {type(e).__name__}\n")
                import traceback
                self.root.after(0, self._append_log, f"[TEST] Traceback:\n{traceback.format_exc()}\n")
                self.root.after(0, self._update_test_status, status_str, "red")
        
        # Executa em thread separada para não travar o GUI
        thread = threading.Thread(target=run_test_thread, daemon=True)
        thread.start()
    
    def _sweep_finished(self):
        """Callback quando o sweep termina"""
        self.is_running = False
        self.run_button.config(state=tk.NORMAL)
        self.stop_button.config(state=tk.DISABLED)
        self.process = None
        self.load_log_file()  # Recarrega o log após terminar
    
    def stop_sweep(self):
        """Para a execução do sweep"""
        if self.process:
            self.process.terminate()
            self.is_running = False
            self._append_log("\n[INFO] Execução interrompida pelo usuário.\n")
            self._sweep_finished()
    
    def toggle_view(self):
        """Alterna entre conteúdo completo e vencedores overall"""
        self.showing_winners = not self.showing_winners
        if self.showing_winners:
            self.show_winners_btn.config(text="📄 Log Completo")
            self._show_overall_winners()
        else:
            self.show_winners_btn.config(text="📊 Vencedores Overall")
            self.load_log_file()
    
    def load_log_file(self):
        """Carrega o arquivo sweep_log.txt ou mostra vencedores overall"""
        if self.showing_winners:
            self._show_overall_winners()
            return
            
        log_file = self.vars.get("log_file", tk.StringVar(value="sweep_log.txt")).get()
        
        if not log_file:
            log_file = "sweep_log.txt"
        
        self.results_text.delete(1.0, tk.END)
        
        # Primeiro tenta carregar o arquivo especificado
        if os.path.exists(log_file):
            # Carrega o arquivo completo
            try:
                with open(log_file, "r", encoding="utf-8") as f:
                    content = f.read()
                self.results_text.insert(tk.END, content)
                return
            except Exception as e:
                self.results_text.insert(tk.END, f"Erro ao carregar {log_file}: {str(e)}\n")
        
        # Se não existir, tenta carregar sweep_log.txt padrão
        if log_file != "sweep_log.txt" and os.path.exists("sweep_log.txt"):
            try:
                with open("sweep_log.txt", "r", encoding="utf-8") as f:
                    content = f.read()
                self.results_text.insert(tk.END, content)
                return
            except Exception as e:
                self.results_text.insert(tk.END, f"Erro ao carregar sweep_log.txt: {str(e)}\n")
        
        # Se nenhum arquivo existir, mostra vencedores overall filtrados do sweep_log.txt se existir
        if os.path.exists("sweep_log.txt"):
            self._show_overall_winners()
        else:
            self.results_text.insert(tk.END, "Nenhum arquivo de log encontrado.\n")
            self.results_text.insert(tk.END, f"Procurando por: {log_file}\n")
            self.results_text.insert(tk.END, "Execute um sweep para gerar resultados.\n")
    
    def _show_overall_winners(self):
        """Extrai e mostra os vencedores overall do sweep_log.txt"""
        self.results_text.delete(1.0, tk.END)
        
        # Tenta carregar o arquivo especificado primeiro, senão usa o padrão
        log_file = self.vars.get("log_file", tk.StringVar(value="sweep_log.txt")).get()
        if not log_file:
            log_file = "sweep_log.txt"
        
        # Se o arquivo especificado não existir, tenta o padrão
        if not os.path.exists(log_file) and os.path.exists("sweep_log.txt"):
            log_file = "sweep_log.txt"
        
        if not os.path.exists(log_file):
            self.results_text.insert(tk.END, "Nenhum arquivo de log encontrado.\n")
            self.results_text.insert(tk.END, f"Procurando por: {log_file}\n")
            self.results_text.insert(tk.END, "Execute um sweep para gerar resultados.\n")
            return
        
        try:
            with open(log_file, "r", encoding="utf-8") as f:
                lines = f.readlines()
            
            self.results_text.insert(tk.END, "=" * 80 + "\n")
            self.results_text.insert(tk.END, f"VENCEDORES OVERALL (Filtrados de {log_file})\n")
            self.results_text.insert(tk.END, "=" * 80 + "\n\n")
            
            # Extrair todas as linhas de vencedores
            winners = []
            
            for line in lines:
                # Detecta linha de resultado [BEST ...]
                if "[BEST" in line:
                    # Extrai o período do label [BEST PERIODD] primeiro
                    best_match = re.search(r'\[BEST\s+(\d+|ALL)D?\]', line, re.IGNORECASE)
                    if best_match:
                        days = best_match.group(1)
                        # Extrai o percentual - padrão: "PERIODd +183.21%" no final da linha
                        # Procura pelo padrão específico do período seguido de percentual
                        pct_match = re.search(rf'{days}(?:d)?\s+([+-]?\d+\.\d+)%', line, re.IGNORECASE)
                        if pct_match:
                            pct = float(pct_match.group(1))
                            # Remove códigos ANSI antes de adicionar
                            clean_line = re.sub(r'\x1b\[[0-9;]*m', '', line.strip())
                            winners.append((days, pct, clean_line))
            
            # Agrupar por período e manter apenas o melhor de cada período
            best_by_period = {}
            for days, pct, full_line in winners:
                if days not in best_by_period or pct > best_by_period[days][1]:
                    best_by_period[days] = (days, pct, full_line)
            
            # Converter para lista e ordenar por percentual (maior primeiro)
            unique_winners = list(best_by_period.values())
            unique_winners.sort(key=lambda x: x[1], reverse=True)
            
            # Mostrar top 30
            top_count = min(30, len(unique_winners))
            self.results_text.insert(tk.END, f"Top {top_count} Vencedores (1 por período, ordenados por retorno):\n\n")
            
            for i, (days, pct, full_line) in enumerate(unique_winners[:top_count], 1):
                self.results_text.insert(tk.END, f"{i}. [{days}D] {pct:+.2f}%\n")
                self.results_text.insert(tk.END, f"   {full_line}\n\n")
            
            if not unique_winners:
                self.results_text.insert(tk.END, "Nenhum vencedor encontrado no log.\n")
            else:
                self.results_text.insert(tk.END, f"\nTotal de vencedores únicos (1 por período): {len(unique_winners)}\n")
                self.results_text.insert(tk.END, f"Total de entradas no log: {len(winners)}\n")
                
        except Exception as e:
            self.results_text.insert(tk.END, f"Erro ao processar sweep_log.txt: {str(e)}\n")


def main():
    root = tk.Tk()
    app = SweepGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()

