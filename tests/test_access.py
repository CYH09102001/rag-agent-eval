import access


def test_check_code_requires_configured_code(monkeypatch):
    monkeypatch.delenv("APP_ACCESS_CODE", raising=False)
    assert not access.access_code_configured()
    assert not access.check_code("anything")  # no code set -> nothing unlocks


def test_check_code_accepts_only_the_right_code(monkeypatch):
    monkeypatch.setenv("APP_ACCESS_CODE", "hire-me-2026")
    assert access.access_code_configured()
    assert access.check_code("hire-me-2026")
    assert access.check_code("  hire-me-2026 ")  # stray spaces from copy/paste
    assert not access.check_code("wrong")
    assert not access.check_code("")


def test_daily_budget_blocks_after_limit():
    b = access.DailyBudget(limit=3)
    assert [b.try_use() for _ in range(5)] == [True, True, True, False, False]
    assert b.remaining == 0


def test_daily_budget_resets_next_day():
    from datetime import date, timedelta

    b = access.DailyBudget(limit=1)
    assert b.try_use() and not b.try_use()
    b._day = date.today() - timedelta(days=1)  # pretend the counter is from yesterday
    assert b.remaining == 1
    assert b.try_use()
