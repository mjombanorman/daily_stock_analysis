# -*- coding: utf-8 -*-
"""Regression tests for Kenya (NSE) suffix-only market identity + boundaries.

Mirrors tests/test_jp_kr_market_support.py and tests/test_tw_market_support.py.
Unlike those markets, Kenya has NO data source wired in yet (no yfinance
coverage, no exchange-calendars entry, no internal fetcher built) -- this
file only covers market identification, LLM prompt context, and the
explicit not-supported/fail-closed boundaries that keep a ``.KE`` code from
being silently misrouted into the default A-share fetcher chain.
"""

from datetime import date

import pandas as pd
from data_provider.base import BaseFetcher, DataFetchError, DataFetcherManager, normalize_stock_code
from src.core.trading_calendar import (
    MARKET_EXCHANGE,
    MARKET_TIMEZONE,
    get_market_for_stock,
    get_market_now,
    get_next_trading_date,
    get_trading_dates,
    infer_market_phase,
    is_market_open,
    MarketPhase,
)
from src.market_context import detect_market, get_market_guidelines, get_market_role
from src.services.market_symbol_utils import get_suffix_market, is_ke_suffix_symbol


class _FakeFetcher(BaseFetcher):
    """A-share-only fake fetcher that fails loudly if called for a .KE code."""

    def __init__(self, name: str = "EfinanceFetcher"):
        self.name = name
        self.priority = 0
        self.calls = []

    def _fetch_raw_data(self, stock_code: str, start_date: str, end_date: str) -> pd.DataFrame:
        raise NotImplementedError

    def _normalize_data(self, df: pd.DataFrame, stock_code: str) -> pd.DataFrame:
        raise NotImplementedError

    def get_daily_data(self, stock_code, start_date=None, end_date=None, days=30):
        self.calls.append(stock_code)
        raise DataFetchError(f"{self.name} should never be called for {stock_code}")


def test_normalize_and_detect_ke_suffix_codes() -> None:
    assert normalize_stock_code("scom.ke") == "SCOM.KE"
    assert get_suffix_market("SCOM.KE") == "ke"
    assert get_suffix_market("KCB.KE") == "ke"
    assert get_suffix_market("EQTY.KE") == "ke"
    assert is_ke_suffix_symbol("SCOM.KE") is True
    assert is_ke_suffix_symbol("SCOM") is False

    assert detect_market("SCOM.KE") == "ke"
    # Suffix-only by design: a bare alpha code is ambiguous with the
    # existing US-ticker regex and keeps US semantics, same collision
    # precedent as the TW/JP/KR bare-code rules.
    assert detect_market("SCOM") == "us"

    assert get_market_for_stock("SCOM.KE") == "ke"


def test_market_guidelines_for_ke_use_kenya_context_not_a_share() -> None:
    role_zh = get_market_role("SCOM.KE", lang="zh")
    role_en = get_market_role("SCOM.KE", lang="en")
    assert "肯尼亚" in role_zh
    assert "Kenya" in role_en

    guidelines_en = get_market_guidelines("SCOM.KE", lang="en")
    guidelines_zh = get_market_guidelines("SCOM.KE", lang="zh")
    assert "NSE" in guidelines_en
    assert "VWAP" in guidelines_en
    assert "肯尼亚" in guidelines_zh
    # Guideline explicitly names these as excluded A-share-specific concepts
    # (same pattern as the existing jp/kr/tw guidelines), so the substring
    # IS expected to appear here -- it is naming what not to apply, not
    # applying it.
    assert "北向资金" in guidelines_zh
    assert "龙虎榜" in guidelines_zh


def test_ke_has_no_fabricated_exchange_calendar() -> None:
    # exchange-calendars has no African exchange coverage; ke must stay out
    # of MARKET_EXCHANGE rather than pointing at an invented calendar code.
    assert "ke" not in MARKET_EXCHANGE
    assert MARKET_TIMEZONE["ke"] == "Africa/Nairobi"

    now = get_market_now("ke")
    assert "Nairobi" in str(now.tzinfo)

    # No calendar registered -> existing fail-open/fail-closed semantics,
    # not a crash and not a fabricated "always open" guarantee dressed up
    # as real calendar data.
    assert is_market_open("ke", date(2026, 11, 12)) is True
    assert get_trading_dates("ke", date(2026, 11, 1), date(2026, 11, 12)) is None
    assert get_next_trading_date("ke", date(2026, 11, 12)) is None
    assert infer_market_phase("ke") == MarketPhase.UNKNOWN


def test_ke_realtime_quote_fails_closed_without_misrouting_to_cn() -> None:
    fake = _FakeFetcher()
    manager = DataFetcherManager(fetchers=[fake])

    quote = manager.get_realtime_quote("SCOM.KE", log_final_failure=False)

    assert quote is None
    assert fake.calls == []


def test_ke_daily_history_fails_closed_without_misrouting_to_cn() -> None:
    fake = _FakeFetcher()
    manager = DataFetcherManager(fetchers=[fake])

    try:
        manager.get_daily_data("SCOM.KE", days=30)
        raised = False
    except DataFetchError:
        raised = True

    assert raised is True
    assert fake.calls == []


def test_ke_fundamental_context_returns_not_supported() -> None:
    manager = DataFetcherManager(fetchers=[_FakeFetcher()])

    context = manager.get_fundamental_context("SCOM.KE")

    assert context.get("market") == "ke"
    assert context.get("status") == "not_supported"
