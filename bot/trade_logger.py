import json
import logging
import sqlite3
from datetime import datetime
from typing import Optional

from bot.signal_generator import TradingSignal
from bot.trade_constructor import SingleLegTrade

logger = logging.getLogger(__name__)
DB_PATH = "qqq_trades.db"

_SCHEMA = """
CREATE TABLE IF NOT EXISTS signals (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp       TEXT NOT NULL,
    signal          TEXT NOT NULL,
    direction       TEXT,
    confidence      INTEGER,
    spy_price       REAL,
    vix             REAL,
    iv_rank         REAL,
    breadth_pct     REAL,
    technical_score INTEGER,
    macro_score     INTEGER,
    options_score   INTEGER,
    reasons         TEXT,
    alert_sent      INTEGER DEFAULT 0
);

CREATE TABLE IF NOT EXISTS trades (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    signal_id       INTEGER REFERENCES signals(id),
    direction       TEXT,
    expiration      TEXT,
    dte_at_entry    INTEGER,
    long_strike     REAL,
    short_strike    REAL,
    debit           REAL,
    max_profit      REAL,
    max_loss        REAL,
    risk_reward     REAL,
    contracts       INTEGER,
    total_risk      REAL,
    entry_timestamp TEXT,
    exit_timestamp  TEXT,
    exit_reason     TEXT,
    exit_debit      REAL,
    pnl_dollars     REAL,
    pnl_pct         REAL,
    outcome         TEXT DEFAULT 'open'
);

CREATE VIEW IF NOT EXISTS win_rate AS
SELECT
    COUNT(*) AS total_trades,
    SUM(CASE WHEN outcome = 'win'  THEN 1 ELSE 0 END) AS wins,
    SUM(CASE WHEN outcome = 'loss' THEN 1 ELSE 0 END) AS losses,
    ROUND(100.0 * SUM(CASE WHEN outcome = 'win' THEN 1 ELSE 0 END)
          / NULLIF(COUNT(*), 0), 1) AS win_rate_pct,
    ROUND(AVG(pnl_pct), 2) AS avg_pnl_pct,
    ROUND(SUM(COALESCE(pnl_dollars, 0)), 2) AS total_pnl
FROM trades WHERE outcome != 'open';
"""


def init_db(db_path: str = DB_PATH) -> None:
    with sqlite3.connect(db_path) as conn:
        conn.executescript(_SCHEMA)
    logger.debug(f"Database initialized at {db_path}")


def log_signal(
    signal: TradingSignal,
    spy_price: float,
    vix: float,
    iv_rank: float,
    breadth_pct: float,
    alert_sent: bool = False,
    db_path: str = DB_PATH,
) -> int:
    bd = signal.confidence_breakdown
    with sqlite3.connect(db_path) as conn:
        cur = conn.execute(
            """INSERT INTO signals
               (timestamp, signal, direction, confidence, spy_price, vix,
                iv_rank, breadth_pct, technical_score, macro_score, options_score,
                reasons, alert_sent)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (
                signal.timestamp, signal.signal, signal.direction,
                signal.confidence, spy_price, vix, iv_rank, breadth_pct,
                bd.get("technical", 0), bd.get("macro", 0), bd.get("options", 0),
                json.dumps(signal.signal_reasons),
                1 if alert_sent else 0,
            ),
        )
        return cur.lastrowid


def log_trade_entry(
    signal_id: int,
    trade: SingleLegTrade,
    db_path: str = DB_PATH,
) -> int:
    with sqlite3.connect(db_path) as conn:
        cur = conn.execute(
            """INSERT INTO trades
               (signal_id, direction, expiration, dte_at_entry,
                long_strike, short_strike, debit, max_profit, max_loss,
                risk_reward, contracts, total_risk, entry_timestamp, outcome)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (
                signal_id, trade.direction, trade.expiration, trade.dte,
                trade.strike, trade.strike, trade.ask_price,
                0.0, trade.total_cost, 0.0,
                trade.contracts, trade.total_cost,
                datetime.now().isoformat(), "open",
            ),
        )
        return cur.lastrowid


def update_trade_exit(
    trade_id: int,
    exit_debit: float,
    exit_reason: str,
    db_path: str = DB_PATH,
) -> None:
    with sqlite3.connect(db_path) as conn:
        row = conn.execute(
            "SELECT debit, max_profit, contracts FROM trades WHERE id=?", (trade_id,)
        ).fetchone()
        if not row:
            logger.warning(f"Trade {trade_id} not found for exit update")
            return
        entry_debit, max_profit, contracts = row
        pnl_per_contract = exit_debit - entry_debit   # positive = profit (bought at entry_debit, exiting at exit_debit)
        pnl_dollars = round(pnl_per_contract * contracts * 100, 2)
        pnl_pct = round(pnl_per_contract / entry_debit * 100, 2) if entry_debit > 0 else 0
        outcome = "win" if pnl_dollars > 0 else ("loss" if pnl_dollars < 0 else "breakeven")
        conn.execute(
            """UPDATE trades SET exit_timestamp=?, exit_reason=?, exit_debit=?,
               pnl_dollars=?, pnl_pct=?, outcome=? WHERE id=?""",
            (datetime.now().isoformat(), exit_reason, exit_debit,
             pnl_dollars, pnl_pct, outcome, trade_id),
        )


def get_win_rate_summary(db_path: str = DB_PATH) -> dict:
    try:
        with sqlite3.connect(db_path) as conn:
            cur = conn.execute("SELECT * FROM win_rate")
            row = cur.fetchone()
            if not row:
                return {"total_trades": 0}
            cols = [d[0] for d in cur.description]
            return dict(zip(cols, row))
    except Exception:
        return {"total_trades": 0}
