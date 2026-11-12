"""Silence detection: tells you when the bot has stopped doing its job instead of failing quietly."""
from datetime import datetime, timedelta
from typing import Optional

import pytz

ET = pytz.timezone("America/New_York")


class _State:
    def __init__(self):
        self.started = datetime.now(ET)
        self.last_scan_ok: Optional[datetime] = None
        self.stale_alerted = False
        self.last_error_alert: Optional[datetime] = None
        self.monitor_fails = 0


STATE = _State()


def scan_ok(now: Optional[datetime] = None) -> Optional[str]:
    """Call after every successful scan. Returns a recovery message if we had alerted about silence."""
    STATE.last_scan_ok = now or datetime.now(ET)
    if STATE.stale_alerted:
        STATE.stale_alerted = False
        return "✅ **Bot recovered** — scans are succeeding again."
    return None


def check_stale(now: datetime, market_open_dt: datetime, threshold_min: int = 60) -> Optional[str]:
    """Run while the market is open. Returns an alert message once if no scan has succeeded recently."""
    ref = max(d for d in (STATE.last_scan_ok, market_open_dt, STATE.started) if d is not None)
    gap = (now - ref).total_seconds() / 60
    if gap > threshold_min and not STATE.stale_alerted:
        STATE.stale_alerted = True
        return (f"⚠️ **No successful scan for {int(gap)} min** while the market is open.\n"
                f"The data source may be down or the bot is stuck. Alerts and exit checks may be delayed.")
    return None


def error_alert_due(now: Optional[datetime] = None, min_gap_min: int = 60) -> bool:
    """Rate-limit error alerts to one per hour so a persistent failure doesn't spam the channel."""
    now = now or datetime.now(ET)
    if STATE.last_error_alert is None or now - STATE.last_error_alert >= timedelta(minutes=min_gap_min):
        STATE.last_error_alert = now
        return True
    return False


def monitor_result(ok: bool, fails_to_alert: int = 3) -> Optional[str]:
    """Track consecutive failures to price the open position. Alerts once when the threshold is hit."""
    if ok:
        STATE.monitor_fails = 0
        return None
    STATE.monitor_fails += 1
    if STATE.monitor_fails == fails_to_alert:
        return ("⚠️ **Can't price your open position** — the last "
                f"{fails_to_alert} checks failed. Stop-loss and exit alerts won't fire until this recovers; "
                "check the position manually.")
    return None
