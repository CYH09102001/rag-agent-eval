"""Safeguards for putting the demo on the internet.

Problem: every answer costs you API money. If the app is public, strangers (or
bots) could run up your bill. Two protections:

1. ACCESS CODE  - if APP_ACCESS_CODE is set, live Claude answers need that code.
                  Everyone else still gets the free search-only view. You put the
                  code in your resume / cover letter / email to the recruiter.
2. DAILY BUDGET - a hard cap on live answers per day across ALL visitors
                  (DAILY_QUERY_LIMIT, default 100), even for code holders.

Also set a monthly spend limit in the Anthropic console: it is the real backstop.
"""
import hmac
import os
import threading
from datetime import date


def access_code_configured() -> bool:
    return bool(os.getenv("APP_ACCESS_CODE"))


def check_code(entered: str) -> bool:
    """Constant-time comparison. False if no code is configured or input is empty."""
    expected = os.getenv("APP_ACCESS_CODE", "")
    if not expected or not entered:
        return False
    return hmac.compare_digest(entered.strip().encode(), expected.encode())


class DailyBudget:
    """Counts live queries and refuses once today's limit is reached.
    In-memory: it resets if the server restarts, which is fine for a demo."""

    def __init__(self, limit: int = None):
        self.limit = limit if limit is not None else int(os.getenv("DAILY_QUERY_LIMIT", "100"))
        self._day = date.today()
        self._used = 0
        self._lock = threading.Lock()

    def try_use(self) -> bool:
        with self._lock:
            today = date.today()
            if today != self._day:
                self._day, self._used = today, 0
            if self._used >= self.limit:
                return False
            self._used += 1
            return True

    @property
    def remaining(self) -> int:
        with self._lock:
            if date.today() != self._day:
                return self.limit
            return max(self.limit - self._used, 0)
