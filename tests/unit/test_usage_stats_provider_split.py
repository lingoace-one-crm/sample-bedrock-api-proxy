"""Provider-split (Anthropic vs. non-Anthropic) cache stats behavior.

Covers:
- `_is_anthropic_model`: pricing-table-driven provider attribution, with a
  model-id prefix fallback when no pricing row exists.
- `UsageStatsManager.aggregate_usage_for_key`: the anthropic_* subset fields
  returned alongside the existing totals.
- `UsageStatsManager.update_stats` / `increment_stats`: persisting the
  anthropic_* fields (first-write and incremental paths).

This split exists because Anthropic models report a genuine
cache_creation_input_tokens ("write") signal that non-Anthropic
(OpenAI-compatible) models via Bedrock Mantle don't have an equivalent
for — see the frontend's anthropicCacheHitRate/otherCacheHitRate formulas
for how the two are displayed differently.
"""

from unittest.mock import MagicMock

from app.db.dynamodb import UsageStatsManager, _is_anthropic_model


class TestIsAnthropicModel:
    def test_pricing_provider_anthropic(self):
        assert _is_anthropic_model("global.anthropic.claude-sonnet-4-6", {"provider": "Anthropic"}) is True

    def test_pricing_provider_openai(self):
        assert _is_anthropic_model("openai.gpt-5.6-luna", {"provider": "OpenAI"}) is False

    def test_pricing_provider_case_insensitive(self):
        assert _is_anthropic_model("some-model", {"provider": "ANTHROPIC"}) is True

    def test_pricing_provider_other(self):
        assert _is_anthropic_model("qwen.qwen3-32b-v1:0", {"provider": "Qwen"}) is False

    def test_no_pricing_falls_back_to_model_id_prefix_anthropic(self):
        assert _is_anthropic_model("global.anthropic.claude-opus-4-8", None) is True
        assert _is_anthropic_model("claude-sonnet-5", None) is True

    def test_no_pricing_falls_back_to_model_id_prefix_non_anthropic(self):
        assert _is_anthropic_model("openai.gpt-5.6-luna", None) is False
        assert _is_anthropic_model("qwen.qwen3-32b-v1:0", None) is False

    def test_pricing_row_without_provider_field_falls_back_to_model_id(self):
        # A pricing row exists but has no provider set — must not be treated
        # as a truthy-but-wrong provider string; fall back to the model id.
        assert _is_anthropic_model("claude-sonnet-5", {}) is True
        assert _is_anthropic_model("openai.gpt-5.6-luna", {}) is False


class TestAggregateUsageProviderSplit:
    def test_mixed_anthropic_and_openai_records_split_correctly(self):
        manager = UsageStatsManager.__new__(UsageStatsManager)
        manager.usage_table = MagicMock()
        manager.usage_table.query.return_value = {
            "Items": [
                {
                    "timestamp": "1000",
                    "model": "openai.gpt-5.6-luna",
                    "input_tokens": 500,
                    "output_tokens": 10,
                    "cached_tokens": 400,
                    "cache_write_input_tokens": 0,
                    "metadata": {},
                },
                {
                    "timestamp": "2000",
                    "model": "claude-sonnet-5",
                    "input_tokens": 100,
                    "output_tokens": 20,
                    "cached_tokens": 30,
                    "cache_write_input_tokens": 15,
                    "metadata": {},
                },
            ]
        }

        result = manager.aggregate_usage_for_key(
            "sk-test",
            pricing_cache={
                "openai.gpt-5.6-luna": {"provider": "OpenAI"},
                "claude-sonnet-5": {"provider": "Anthropic"},
            },
        )

        # Combined totals (existing behavior) are unaffected.
        assert result["total_input_tokens"] == 600
        assert result["total_cached_tokens"] == 430
        assert result["total_cache_write_tokens"] == 15
        assert result["total_requests"] == 2

        # Anthropic-only subset covers just the claude-sonnet-5 record.
        assert result["anthropic_input_tokens"] == 100
        assert result["anthropic_cached_tokens"] == 30
        assert result["anthropic_cache_write_tokens"] == 15

    def test_all_openai_records_have_zero_anthropic_subset(self):
        manager = UsageStatsManager.__new__(UsageStatsManager)
        manager.usage_table = MagicMock()
        manager.usage_table.query.return_value = {
            "Items": [
                {
                    "timestamp": "1000",
                    "model": "openai.gpt-5.6-luna",
                    "input_tokens": 500,
                    "output_tokens": 10,
                    "cached_tokens": 400,
                    "cache_write_input_tokens": 0,
                    "metadata": {},
                },
            ]
        }

        result = manager.aggregate_usage_for_key(
            "sk-test",
            pricing_cache={"openai.gpt-5.6-luna": {"provider": "OpenAI"}},
        )

        assert result["total_input_tokens"] == 500
        assert result["anthropic_input_tokens"] == 0
        assert result["anthropic_cached_tokens"] == 0
        assert result["anthropic_cache_write_tokens"] == 0

    def test_no_pricing_cache_falls_back_to_model_id_for_split(self):
        """Even without a pricing_cache at all, the split must still work
        via the model-id prefix fallback in _is_anthropic_model — this
        mirrors real usage records for models with no pricing row yet.
        """
        manager = UsageStatsManager.__new__(UsageStatsManager)
        manager.usage_table = MagicMock()
        manager.usage_table.query.return_value = {
            "Items": [
                {
                    "timestamp": "1000",
                    "model": "global.anthropic.claude-opus-4-8",
                    "input_tokens": 100,
                    "output_tokens": 5,
                    "cached_tokens": 20,
                    "cache_write_input_tokens": 10,
                    "metadata": {},
                },
                {
                    "timestamp": "2000",
                    "model": "openai.gpt-5.6-luna",
                    "input_tokens": 200,
                    "output_tokens": 5,
                    "cached_tokens": 150,
                    "cache_write_input_tokens": 0,
                    "metadata": {},
                },
            ]
        }

        result = manager.aggregate_usage_for_key("sk-test", pricing_cache=None)

        assert result["anthropic_input_tokens"] == 100
        assert result["anthropic_cached_tokens"] == 20
        assert result["anthropic_cache_write_tokens"] == 10
        # Total input/cached still cover both records.
        assert result["total_input_tokens"] == 300
        assert result["total_cached_tokens"] == 170

    def test_record_missing_model_with_pricing_cache_does_not_crash(self):
        """回归：pricing_cache 非空但某条记录缺 model 字段时，provider 归属判断
        曾因 bedrock_model_id 未定义抛 UnboundLocalError，中断整个聚合循环并让
        budget 静默冻结。该组合必须被容忍：缺 model 的记录按非 Anthropic 归属，
        且不影响同页其他记录。
        """
        manager = UsageStatsManager.__new__(UsageStatsManager)
        manager.usage_table = MagicMock()
        manager.usage_table.query.return_value = {
            "Items": [
                {
                    # 关键组合：有 usage、无 model 字段
                    "timestamp": "1000",
                    "input_tokens": 100,
                    "output_tokens": 20,
                    "cached_tokens": 10,
                    "cache_write_input_tokens": 5,
                    "metadata": {},
                },
                {
                    "timestamp": "2000",
                    "model": "claude-sonnet-5",
                    "input_tokens": 50,
                    "output_tokens": 10,
                    "cached_tokens": 30,
                    "cache_write_input_tokens": 15,
                    "metadata": {},
                },
            ]
        }

        # pricing_cache 非空是触发条件；必须不抛异常。
        result = manager.aggregate_usage_for_key(
            "sk-test",
            pricing_cache={"claude-sonnet-5": {"provider": "Anthropic"}},
        )

        # 两条记录都被聚合（缺 model 的那条没有中断循环）。
        assert result["total_requests"] == 2
        assert result["total_input_tokens"] == 150
        # 只有 claude-sonnet-5 归入 Anthropic；缺 model 的记录归入非 Anthropic。
        assert result["anthropic_input_tokens"] == 50
        assert result["anthropic_cached_tokens"] == 30
        assert result["anthropic_cache_write_tokens"] == 15


class TestUpdateAndIncrementStatsPersistAnthropicSplit:
    def test_update_stats_writes_anthropic_fields(self):
        manager = UsageStatsManager.__new__(UsageStatsManager)
        manager.table = MagicMock()

        manager.update_stats(
            api_key="sk-test",
            input_tokens=600,
            output_tokens=30,
            cached_tokens=430,
            cache_write_tokens=15,
            request_count=2,
            last_aggregated_timestamp=2000,
            anthropic_input_tokens=100,
            anthropic_cached_tokens=30,
            anthropic_cache_write_tokens=15,
        )

        put_item_call = manager.table.put_item.call_args
        item = put_item_call.kwargs["Item"]
        assert item["anthropic_input_tokens"] == 100
        assert item["anthropic_cached_tokens"] == 30
        assert item["anthropic_cache_write_tokens"] == 15
        # Existing combined fields still present and correct.
        assert item["total_input_tokens"] == 600
        assert item["total_cache_write_tokens"] == 15

    def test_update_stats_defaults_anthropic_fields_to_zero(self):
        """Callers that don't pass the new kwargs (e.g. older code paths)
        must not crash, and should persist zero rather than omitting the
        fields — omission would make get_stats()'s .get(..., 0) fallback
        do the same thing, but explicit zero keeps the schema consistent.
        """
        manager = UsageStatsManager.__new__(UsageStatsManager)
        manager.table = MagicMock()

        manager.update_stats(
            api_key="sk-test",
            input_tokens=100,
            output_tokens=10,
            cached_tokens=5,
            cache_write_tokens=0,
            request_count=1,
        )

        item = manager.table.put_item.call_args.kwargs["Item"]
        assert item["anthropic_input_tokens"] == 0
        assert item["anthropic_cached_tokens"] == 0
        assert item["anthropic_cache_write_tokens"] == 0

    def test_increment_stats_passes_anthropic_deltas_to_update_expression(self):
        manager = UsageStatsManager.__new__(UsageStatsManager)
        manager.table = MagicMock()

        manager.increment_stats(
            api_key="sk-test",
            delta_input_tokens=200,
            delta_output_tokens=5,
            delta_cached_tokens=150,
            delta_cache_write_tokens=0,
            delta_request_count=1,
            last_aggregated_timestamp=3000,
            delta_anthropic_input_tokens=0,
            delta_anthropic_cached_tokens=0,
            delta_anthropic_cache_write_tokens=0,
        )

        update_call = manager.table.update_item.call_args
        expr = update_call.kwargs["UpdateExpression"]
        values = update_call.kwargs["ExpressionAttributeValues"]

        assert "anthropic_input_tokens" in expr
        assert "anthropic_cached_tokens" in expr
        assert "anthropic_cache_write_tokens" in expr
        assert values[":anthropic_input_tokens"] == 0
        assert values[":anthropic_cached_tokens"] == 0
        assert values[":anthropic_cache_write_tokens"] == 0
