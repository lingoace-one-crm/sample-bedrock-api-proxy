"""Usage stats pricing behavior."""

from decimal import Decimal
from unittest.mock import MagicMock

from app.db.dynamodb import UsageStatsManager


def test_aggregate_usage_prices_cached_tokens_as_subset_of_input_tokens():
    manager = UsageStatsManager.__new__(UsageStatsManager)
    manager.usage_table = MagicMock()
    manager.usage_table.query.return_value = {
        "Items": [
            {
                "timestamp": "1000",
                "model": "openai.gpt-oss-120b",
                "input_tokens": 100,
                "output_tokens": 50,
                "cached_tokens": 30,
                "cache_write_input_tokens": 0,
                "metadata": {"input_tokens_include_cached_tokens": True},
            }
        ]
    }

    result = manager.aggregate_usage_for_key(
        "sk-test",
        pricing_cache={
            "openai.gpt-oss-120b": {
                "input_price": Decimal("2.00"),
                "output_price": Decimal("8.00"),
                "cache_read_price": Decimal("0.20"),
                "cache_write_price": Decimal("2.50"),
            }
        },
    )

    expected_cost = ((70 * 2.00) + (50 * 8.00) + (30 * 0.20)) / 1_000_000
    # input_tokens is reported cache-inclusive by OpenAI APIs (flag set), so the
    # displayed total must subtract the cached subset to match the Anthropic
    # convention used elsewhere — otherwise cached is counted twice (once inside
    # input, once as cached). 100 - 30 cached = 70.
    assert result["total_input_tokens"] == 70
    assert result["total_cached_tokens"] == 30
    assert abs(result["total_cost"] - expected_cost) < 1e-12


def test_aggregate_usage_input_tokens_exclusive_when_flag_absent():
    """Native Anthropic records (no flag) keep input_tokens as-is (already cache-exclusive)."""
    manager = UsageStatsManager.__new__(UsageStatsManager)
    manager.usage_table = MagicMock()
    manager.usage_table.query.return_value = {
        "Items": [
            {
                "timestamp": "1000",
                "model": "claude-sonnet-4-5",
                "input_tokens": 100,
                "output_tokens": 50,
                "cached_tokens": 30,
                "cache_write_input_tokens": 0,
                "metadata": {},
            }
        ]
    }

    result = manager.aggregate_usage_for_key("sk-test")

    assert result["total_input_tokens"] == 100
    assert result["total_cached_tokens"] == 30


def test_aggregate_usage_tolerates_null_cache_write_tokens():
    """A record with cache_write_input_tokens stored as DynamoDB NULL (Python
    None) must not crash the whole aggregation run — it should be treated as
    zero. Regression test for a bug where a single stale/malformed record
    (missing this field, e.g. from an older code path) raised
    ``TypeError: int() argument ... not 'NoneType'`` and aborted
    ``aggregate_all_usage`` for every API key processed after it, silently
    freezing budget_used/budget_used_mtd at 0 despite real, billable usage.
    """
    manager = UsageStatsManager.__new__(UsageStatsManager)
    manager.usage_table = MagicMock()
    manager.usage_table.query.return_value = {
        "Items": [
            {
                "timestamp": "1000",
                "model": "openai.gpt-5.6-luna",
                "input_tokens": 100,
                "output_tokens": 20,
                "cached_tokens": None,
                "cache_write_input_tokens": None,
                "metadata": None,
            }
        ]
    }

    # Must not raise TypeError.
    result = manager.aggregate_usage_for_key("sk-test")

    assert result["total_requests"] == 1
    assert result["total_input_tokens"] == 100
    assert result["total_output_tokens"] == 20
    assert result["total_cached_tokens"] == 0
    assert result["total_cache_write_tokens"] == 0


def test_aggregate_usage_null_record_does_not_abort_later_records():
    """A malformed record must be tolerated, not just non-fatal for itself —
    subsequent records in the same query page must still be aggregated.
    """
    manager = UsageStatsManager.__new__(UsageStatsManager)
    manager.usage_table = MagicMock()
    manager.usage_table.query.return_value = {
        "Items": [
            {
                "timestamp": "1000",
                "model": "openai.gpt-5.6-luna",
                "input_tokens": 100,
                "output_tokens": 20,
                "cached_tokens": None,
                "cache_write_input_tokens": None,
                "metadata": None,
            },
            {
                "timestamp": "2000",
                "model": "openai.gpt-5.6-luna",
                "input_tokens": 50,
                "output_tokens": 10,
                "cached_tokens": 0,
                "cache_write_input_tokens": 0,
                "metadata": {},
            },
        ]
    }

    result = manager.aggregate_usage_for_key("sk-test")

    assert result["total_requests"] == 2
    assert result["total_input_tokens"] == 150
    assert result["total_output_tokens"] == 30
