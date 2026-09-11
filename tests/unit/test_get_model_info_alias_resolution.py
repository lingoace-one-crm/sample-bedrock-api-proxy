"""Unit tests for BedrockService.get_model_info()'s alias resolution fix.

Bug: get_model_info() passed the caller-supplied model_id straight to
Bedrock's get_foundation_model() control-plane call without resolving
proxy-facing aliases (e.g. "gpt-5.5", "claude-sonnet-5",
"openai.gpt-5.6-luna") to their real Bedrock model id first. Bedrock's
control plane raises ValidationException (not ResourceNotFoundException)
for an unrecognized alias, which fell through to a bare `raise Exception`
and surfaced as an opaque 500 to callers (e.g. Claude Code CLI's
CLAUDE_CODE_ENABLE_GATEWAY_MODEL_DISCOVERY=1 model-discovery probe).

Fix: resolve the alias via the same mapping list_available_models() already
uses (settings.default_model_mapping + DynamoDB ModelMappingTable) before
calling Bedrock, and treat ValidationException the same as
ResourceNotFoundException (return None -> 404 to the HTTP caller) instead of
raising.
"""
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from botocore.exceptions import ClientError

sys.path.insert(0, str(Path(__file__).parent.parent))

from app.services.bedrock_service import BedrockService


def _make_service():
    with patch("boto3.client"):
        return BedrockService()


def _client_error(code: str, message: str = "boom") -> ClientError:
    return ClientError(
        error_response={"Error": {"Code": code, "Message": message}},
        operation_name="GetFoundationModel",
    )


class TestResolveModelAlias:
    def test_known_alias_resolves_to_real_bedrock_id(self):
        svc = _make_service()
        with patch("app.core.config.settings.default_model_mapping", {
            "gpt-5.5": "openai.gpt-5.5",
        }):
            assert svc._resolve_model_alias("gpt-5.5") == "openai.gpt-5.5"

    def test_ddb_mapping_overrides_default(self):
        svc = _make_service()
        mock_db = MagicMock()
        mock_mapping_manager = MagicMock()
        mock_mapping_manager.list_mappings.return_value = [
            {"anthropic_model_id": "gpt-5.5", "bedrock_model_id": "openai.gpt-5.5-ddb-override"},
        ]
        with patch("app.core.config.settings.default_model_mapping", {"gpt-5.5": "openai.gpt-5.5"}), \
             patch("app.db.dynamodb.DynamoDBClient", return_value=mock_db), \
             patch("app.db.dynamodb.ModelMappingManager", return_value=mock_mapping_manager):
            assert svc._resolve_model_alias("gpt-5.5") == "openai.gpt-5.5-ddb-override"

    def test_unmapped_id_passes_through_unchanged(self):
        svc = _make_service()
        with patch("app.core.config.settings.default_model_mapping", {}):
            assert svc._resolve_model_alias("global.anthropic.claude-opus-4-8") == "global.anthropic.claude-opus-4-8"

    def test_ddb_failure_falls_back_to_defaults_only(self):
        svc = _make_service()
        with patch("app.core.config.settings.default_model_mapping", {"gpt-5.5": "openai.gpt-5.5"}), \
             patch("app.db.dynamodb.DynamoDBClient", side_effect=RuntimeError("DDB down")):
            assert svc._resolve_model_alias("gpt-5.5") == "openai.gpt-5.5"


class TestGetModelInfo:
    def test_alias_is_resolved_before_calling_bedrock(self):
        """get_model_info must call get_foundation_model with the RESOLVED
        model id, not the raw alias the caller passed in."""
        svc = _make_service()
        mock_bedrock_client = MagicMock()
        mock_bedrock_client.get_foundation_model.return_value = {
            "modelDetails": {
                "modelId": "openai.gpt-5.5",
                "modelName": "GPT-5.5",
                "providerName": "OpenAI",
                "inputModalities": ["TEXT"],
                "outputModalities": ["TEXT"],
                "responseStreamingSupported": True,
                "customizationsSupported": [],
            }
        }

        with patch("app.core.config.settings.default_model_mapping", {"gpt-5.5": "openai.gpt-5.5"}), \
             patch("app.db.dynamodb.DynamoDBClient", side_effect=RuntimeError("no ddb in test")), \
             patch("boto3.client", return_value=mock_bedrock_client):
            result = svc.get_model_info("gpt-5.5")

        mock_bedrock_client.get_foundation_model.assert_called_once_with(
            modelIdentifier="openai.gpt-5.5"
        )
        assert result["id"] == "openai.gpt-5.5"
        assert result["provider"] == "OpenAI"

    def test_validation_exception_for_unmapped_alias_returns_none_not_500(self):
        """The core bug fix: an alias with no mapping entry must surface as
        a clean 404 (None), never as an unhandled 500."""
        svc = _make_service()
        mock_bedrock_client = MagicMock()
        mock_bedrock_client.get_foundation_model.side_effect = _client_error(
            "ValidationException", "The provided model identifier is invalid."
        )

        with patch("app.core.config.settings.default_model_mapping", {}), \
             patch("app.db.dynamodb.DynamoDBClient", side_effect=RuntimeError("no ddb in test")), \
             patch("boto3.client", return_value=mock_bedrock_client):
            result = svc.get_model_info("totally-unknown-alias")

        assert result is None

    def test_resource_not_found_still_returns_none(self):
        """Pre-existing behavior for a real-but-nonexistent model id must
        be unaffected by the fix. Neither get_foundation_model nor the
        get_inference_profile fallback finds it, so the result is None."""
        svc = _make_service()
        mock_bedrock_client = MagicMock()
        mock_bedrock_client.get_foundation_model.side_effect = _client_error(
            "ResourceNotFoundException"
        )
        mock_bedrock_client.get_inference_profile.side_effect = _client_error(
            "ResourceNotFoundException"
        )

        with patch("app.core.config.settings.default_model_mapping", {}), \
             patch("app.db.dynamodb.DynamoDBClient", side_effect=RuntimeError("no ddb in test")), \
             patch("boto3.client", return_value=mock_bedrock_client):
            result = svc.get_model_info("global.anthropic.claude-nonexistent-v1:0")

        assert result is None

    def test_other_client_errors_still_raise(self):
        """Non-alias-related ClientErrors (throttling, access denied, etc.)
        must still propagate as errors, not be silently swallowed."""
        svc = _make_service()
        mock_bedrock_client = MagicMock()
        mock_bedrock_client.get_foundation_model.side_effect = _client_error(
            "AccessDeniedException"
        )

        with patch("app.core.config.settings.default_model_mapping", {}), \
             patch("app.db.dynamodb.DynamoDBClient", side_effect=RuntimeError("no ddb in test")), \
             patch("boto3.client", return_value=mock_bedrock_client):
            with pytest.raises(Exception, match="Failed to get model info"):
                svc.get_model_info("some-model")


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
