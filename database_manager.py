import sqlite3
import json
import os
import re
from datetime import datetime

DB_NAME = "sweep_results.db"

def init_db():
    """Inicializa o banco de dados e cria a tabela se não existir."""
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    
    # Tabela principal
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS results (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            asset_name TEXT,
            filename TEXT,
            period_label TEXT,
            pnl_pct REAL,
            max_drawdown REAL,
            total_trades INTEGER,
            win_rate REAL,
            sharpe_ratio REAL,
            params_json TEXT
        )
    ''')
    
    conn.commit()
    conn.close()

def extract_asset_name(filename):
    """
    Tenta extrair o nome do ativo do nome do arquivo.
    Ex: 'BTCUSDT1D.csv' -> 'BTCUSDT'
    Ex: 'ETH-PERP_4H.csv' -> 'ETH-PERP'
    """
    base = os.path.basename(filename)
    name_without_ext = os.path.splitext(base)[0]
    
    # Remove sufixos comuns de tempo (1D, 4H, 240, 15m, etc)
    # Regex busca por fim de string ignorando case
    clean_name = re.sub(r'(_?1[Dd]|[Dd]aily|_?240|_?4[Hh]|_?480|_?8[Hh]|_?60|_?1[Hh])$', '', name_without_ext)
    
    return clean_name.upper()

def save_result(filename, period_label, pnl, metrics, params_dict):
    """
    Salva um resultado no banco de dados.
    """
    try:
        conn = sqlite3.connect(DB_NAME)
        cursor = conn.cursor()
        
        asset_name = extract_asset_name(filename)
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # Serializar parametros para JSON
        params_json = json.dumps(params_dict, ensure_ascii=False)
        
        # Extrair métricas principais do dicionário de metrics (se exitir)
        # metrics ex: {'total_trades': 10, 'max_dd': -5.5, 'win_rate': 0.6, 'sharpe': 1.2 ...}
        max_dd = metrics.get('max_drawdown', 0.0)
        total_trades = metrics.get('total_trades', 0)
        win_rate = metrics.get('win_rate', 0.0)
        sharpe = metrics.get('sharpe_ratio', 0.0)

        cursor.execute('''
            INSERT INTO results (timestamp, asset_name, filename, period_label, pnl_pct, max_drawdown, total_trades, win_rate, sharpe_ratio, params_json)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (timestamp, asset_name, filename, str(period_label), pnl, max_dd, total_trades, win_rate, sharpe, params_json))
        
        conn.commit()
        conn.close()
        print(f"[DB] Resultado salvo para {asset_name} ({period_label}D)")
        
    except Exception as e:
        print(f"[DB] Erro ao salvar resultado: {e}")

def get_all_assets():
    """Retorna lista única de ativos no banco."""
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute("SELECT DISTINCT asset_name FROM results ORDER BY asset_name")
    assets = [row[0] for row in cursor.fetchall()]
    conn.close()
    return assets

def get_results(asset_filter=None):
    """Retorna resultados, opcionalmente filtrados por ativo."""
    conn = sqlite3.connect(DB_NAME)
    conn.row_factory = sqlite3.Row  # Para acessar colunas por nome
    cursor = conn.cursor()
    
    query = "SELECT * FROM results"
    args = []
    
    if asset_filter and asset_filter != "Todos":
        query += " WHERE asset_name = ?"
        args.append(asset_filter)
        
    query += " ORDER BY id DESC"
    
    cursor.execute(query, args)
    rows = cursor.fetchall()
    conn.close()
    return rows
