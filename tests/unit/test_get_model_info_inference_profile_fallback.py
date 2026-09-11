"""Unit tests for BedrockService.get_model_info()'s Inference Profile fallback
and the ModelMappingManager instantiation fix.

Second-round fix, discovered after the initial alias-resolution fix:

1. `_resolve_model_alias` / `list_available_models` both instantiated the
   DynamoDB model-mapping lookup as `DynamoDBClient().model_mapping_manager`
   — an attribute that does not exist on `DynamoDBClient`. This silently
   failed (caught by a broad `except Exception`), so DDB-sourced custom
   mappings were never actually merged in; only `settings.default_model_mapping`
   ever took effect. Fixed to `ModelMappingManager(DynamoDBClient())`, the
   correct constructor signature used everywhere else in the codebase.

2. Even with alias resolution fixed, `get_model_info()` still 404'd for
   correctly-resolved ids like "global.anthropic.claude-sonnet-5", because
   that id is a Bedrock *inference profile*, not a foundation model, and
   `get_foundation_model()` raises ResourceNotFoundException for inference
   profile ids even though the profile is valid and ACTIVE. Fixed by
   falling back to `get_inference_profile()` on ResourceNotFoundException
   before giving up.
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


class TestModelMappingManagerInstantiation:
    def test_resolve_model_alias_uses_correct_constructor(self):
        """ModelMappingManager must be constructed as
        ModelMappingManager(dynamodb_client), never as an attribute lookup
        on the DynamoDBClient instance."""
        svc = _make_service()
        mock_db_client = MagicMock()
        mock_mapping_manager = MagicMock()
        mock_mapping_manager.list_mappings.return_value = [
            {"anthropic_model_id": "gpt-5.5", "bedrock_model_id": "openai.gpt-5.5-from-ddb"},
        ]

        with patch("app.core.config.settings.default_model_mapping", {"gpt-5.5": "openai.gpt-5.5"}), \
             patch("app.db.dynamodb.DynamoDBClient", return_value=mock_db_client), \
             patch("app.db.dynamodb.ModelMappingManager", return_value=mock_mapping_manager) as MockMgr:
            result = svc._resolve_model_alias("gpt-5.5")

        MockMgr.assert_called_once_with(mock_db_client)
        assert result == "openai.gpt-5.5-from-ddb"

    def test_list_available_models_uses_correct_constructor(self):
        svc = _make_service()
        mock_db_client = MagicMock()
        mock_mapping_manager = MagicMock()
        mock_mapping_manager.list_mappings.return_value = []

        with patch("app.core.config.settings.default_model_mapping", {"gpt-5.5": "openai.gpt-5.5"}), \
             patch("app.db.dynamodb.DynamoDBClient", return_value=mock_db_client), \
             patch("app.db.dynamodb.ModelMappingManager", return_value=mock_mapping_manager) as MockMgr:
            svc.list_available_models()

        MockMgr.assert_called_once_with(mock_db_client)


class TestGetModelInfoInferenceProfileFallback:
    def test_inference_profile_id_falls_back_after_foundation_model_404(self):
        """A resolved id that is really an inference profile (e.g.
        "global.anthropic.claude-sonnet-5") must be looked up via
        get_inference_profile after get_foundation_model 404s, not
        surfaced as a 404 to the caller."""
        svc = _make_service()
        mock_bedrock_client = MagicMock()
        mock_bedrock_client.get_foundation_model.side_effect = _client_error(
            "ResourceNotFoundException", "Model not found."
        )
        mock_bedrock_client.get_inference_profile.return_value = {
            "inferenceProfileId": "global.anthropic.claude-sonnet-5",
            "inferenceProfileName": "Global Anthropic Claude Sonnet 5",
            "status": "ACTIVE",
        }

        with patch("app.core.config.settings.default_model_mapping", {
            "claude-sonnet-5": "global.anthropic.claude-sonnet-5",
        }), \
             patch("app.db.dynamodb.DynamoDBClient", side_effect=RuntimeError("no ddb in test")), \
             patch("boto3.client", return_value=mock_bedrock_client):
            result = svc.get_model_info("claude-sonnet-5")

        mock_bedrock_client.get_foundation_model.assert_called_once_with(
            modelIdentifier="global.anthropic.claude-sonnet-5"
        )
        mock_bedrock_client.get_inference_profile.assert_called_once_with(
            inferenceProfileIdentifier="global.anthropic.claude-sonnet-5"
        )
        assert result["id"] == "global.anthropic.claude-sonnet-5"
        assert result["name"] == "Global Anthropic Claude Sonnet 5"

    def test_neither_foundation_model_nor_inference_profile_found_returns_none(self):
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
            result = svc.get_model_info("nonexistent-both-ways")

        assert result is None

    def test_inference_profile_validation_exception_returns_none(self):
        """A genuinely invalid identifier (e.g. an unmapped alias) 404s the
        same way whether get_inference_profile reports it as NotFound or
        Validation."""
        svc = _make_service()
        mock_bedrock_client = MagicMock()
        mock_bedrock_client.get_foundation_model.side_effect = _client_error(
            "ResourceNotFoundException"
        )
        mock_bedrock_client.get_inference_profile.side_effect = _client_error(
            "ValidationException"
        )

        with patch("app.core.config.settings.default_model_mapping", {}), \
             patch("app.db.dynamodb.DynamoDBClient", side_effect=RuntimeError("no ddb in test")), \
             patch("boto3.client", return_value=mock_bedrock_client):
            result = svc.get_model_info("garbage-id")

        assert result is None

    def test_foundation_model_hit_does_not_call_inference_profile(self):
        """When get_foundation_model succeeds outright (a real foundation
        model id), get_inference_profile must never be called."""
        svc = _make_service()
        mock_bedrock_client = MagicMock()
        mock_bedrock_client.get_foundation_model.return_value = {
            "modelDetails": {
                "modelId": "anthropic.claude-sonnet-5",
                "modelName": "Claude Sonnet 5",
                "providerName": "Anthropic",
                "inputModalities": ["TEXT"],
                "outputModalities": ["TEXT"],
                "responseStreamingSupported": True,
                "customizationsSupported": [],
            }
        }

        with patch("app.core.config.settings.default_model_mapping", {}), \
             patch("app.db.dynamodb.DynamoDBClient", side_effect=RuntimeError("no ddb in test")), \
             patch("boto3.client", return_value=mock_bedrock_client):
            result = svc.get_model_info("anthropic.claude-sonnet-5")

        mock_bedrock_client.get_inference_profile.assert_not_called()
        assert result["id"] == "anthropic.claude-sonnet-5"

    def test_other_inference_profile_client_errors_still_raise(self):
        """A non-not-found error from get_inference_profile (e.g. throttling)
        must still propagate, not be swallowed as a 404."""
        svc = _make_service()
        mock_bedrock_client = MagicMock()
        mock_bedrock_client.get_foundation_model.side_effect = _client_error(
            "ResourceNotFoundException"
        )
        mock_bedrock_client.get_inference_profile.side_effect = _client_error(
            "ThrottlingException"
        )

        with patch("app.core.config.settings.default_model_mapping", {}), \
             patch("app.db.dynamodb.DynamoDBClient", side_effect=RuntimeError("no ddb in test")), \
             patch("boto3.client", return_value=mock_bedrock_client):
            with pytest.raises(Exception, match="Failed to get model info"):
                svc.get_model_info("some-throttled-id")


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
