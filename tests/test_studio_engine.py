import os
import sys
import pytest
from unittest.mock import patch, MagicMock

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
if os.path.basename(ROOT_DIR) in ["src", "tests"]:
    ROOT_DIR = os.path.dirname(ROOT_DIR)
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from src.studio_engine import StudioEngine, get_studio_engine


def test_studio_engine_initialization_with_key():
    engine = StudioEngine(api_key="test_api_key", default_model="gemini-3.1-pro-preview")
    assert engine.default_model == "gemini-3.1-pro-preview"
    assert engine.api_key == "test_api_key"
    assert engine.total_calls == 0


def test_studio_engine_missing_key_raises():
    with patch.dict(os.environ, {"GEMINI_API_KEY": "", "GOOGLE_API_KEY": ""}, clear=True):
        with pytest.raises(ValueError, match="API Key"):
            StudioEngine(api_key=None)


def test_studio_engine_generate_text_mock():
    engine = StudioEngine(api_key="test_key")
    mock_response = MagicMock()
    mock_response.text = "Mathematical Proof OK"
    mock_response.usage_metadata.prompt_token_count = 10
    mock_response.usage_metadata.candidates_token_count = 5

    with patch.object(engine.client.models, "generate_content", return_value=mock_response):
        out = engine.generate_text("test prompt")
        assert out == "Mathematical Proof OK"
        assert engine.total_calls == 1
        assert engine.total_prompt_tokens == 10
        assert engine.total_candidate_tokens == 5


def test_studio_engine_generate_json_mock():
    engine = StudioEngine(api_key="test_key")
    mock_response = MagicMock()
    mock_response.text = '```json\n{"candidates": [{"name": "compulsive_ratio", "code": "def f(): pass"}]}\n```'
    mock_response.usage_metadata.prompt_token_count = 50
    mock_response.usage_metadata.candidates_token_count = 20

    with patch.object(engine.client.models, "generate_content", return_value=mock_response):
        data = engine.generate_json("generate features")
        assert "candidates" in data
        assert len(data["candidates"]) == 1
        assert data["candidates"][0]["name"] == "compulsive_ratio"


def test_studio_engine_usage_summary():
    engine = StudioEngine(api_key="test_key")
    engine.total_calls = 3
    engine.total_prompt_tokens = 1500
    engine.total_candidate_tokens = 500
    summary = engine.get_usage_summary()
    assert "Calls: 3" in summary
    assert "2,000" in summary
