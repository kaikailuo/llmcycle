from __future__ import annotations

import json
import tempfile
import unittest
from datetime import date
from pathlib import Path
from unittest.mock import patch

from src.runner import run
from tests.test_openai_provider import MARKDOWN


class RunnerTests(unittest.TestCase):
    def test_candidate_is_complete_and_change_log_is_human_readable(self) -> None:
        document = {
            "models": [
                {
                    "provider": "OpenAI",
                    "model_api_id": "exact-model",
                    "model_region": "Global",
                    "input": [
                        {"context_min": 0, "context_max": 272000, "per_million": 99.0},
                        {"context_min": 272000, "context_max": None, "per_million": 99.0},
                    ],
                    "output": [
                        {"context_min": 0, "context_max": 272000, "per_million": 99.0},
                        {"context_min": 272000, "context_max": None, "per_million": 99.0},
                    ],
                },
                {
                    "provider": "Anthropic",
                    "model_api_id": "other-model",
                    "model_region": "Global",
                    "input": [
                        {"context_min": 0, "context_max": None, "per_million": 3.0}
                    ],
                },
            ]
        }

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "models.json").write_text(json.dumps(document), encoding="utf-8")
            with patch("src.providers.openai.OpenAIProvider.fetch", return_value=MARKDOWN):
                summary = run(root, run_date=date(2026, 9, 30))

            candidate = json.loads(summary.candidate_path.read_text(encoding="utf-8"))
            log = summary.change_log_path.read_text(encoding="utf-8")

        self.assertEqual(2, len(candidate["models"]))
        self.assertEqual(document["models"][1], candidate["models"][1])
        self.assertEqual(1.0, candidate["models"][0]["input"][0]["per_million"])
        self.assertIn("[PRICE]\nOpenAI / exact-model", log)
        self.assertIn("input[0-272000]: 99 -> 1", log)
        self.assertNotIn("Anthropic", log)

    def test_one_missing_model_warns_without_blocking_other_models(self) -> None:
        document = {
            "models": [
                {
                    "provider": "OpenAI",
                    "model_api_id": "missing-model",
                    "model_region": "Global",
                    "input": [
                        {"context_min": 0, "context_max": 272000, "per_million": 9.0},
                        {"context_min": 272000, "context_max": None, "per_million": 9.0},
                    ],
                },
                {
                    "provider": "OpenAI",
                    "model_api_id": "exact-model",
                    "model_region": "Global",
                    "input": [
                        {"context_min": 0, "context_max": 272000, "per_million": 9.0},
                        {"context_min": 272000, "context_max": None, "per_million": 9.0},
                    ],
                },
            ]
        }

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "models.json").write_text(json.dumps(document), encoding="utf-8")
            with patch("src.providers.openai.OpenAIProvider.fetch", return_value=MARKDOWN):
                summary = run(root, run_date=date(2026, 9, 30))

            candidate = json.loads(summary.candidate_path.read_text(encoding="utf-8"))
            log = summary.change_log_path.read_text(encoding="utf-8")

        self.assertEqual(9.0, candidate["models"][0]["input"][0]["per_million"])
        self.assertEqual(1.0, candidate["models"][1]["input"][0]["per_million"])
        self.assertIn("[WARNING]\nOpenAI / missing-model\nofficial price not found", log)
        self.assertEqual(1, summary.warning_count)

    def test_source_failure_still_writes_unchanged_candidate(self) -> None:
        document = {
            "models": [
                {
                    "provider": "OpenAI",
                    "model_api_id": "exact-model",
                    "model_region": "Global",
                    "input": [
                        {"context_min": 0, "context_max": None, "per_million": 9.0}
                    ],
                }
            ]
        }

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "models.json").write_text(json.dumps(document), encoding="utf-8")
            with patch(
                "src.providers.openai.OpenAIProvider.fetch", side_effect=OSError("offline")
            ):
                summary = run(root, run_date=date(2026, 9, 30))

            candidate = json.loads(summary.candidate_path.read_text(encoding="utf-8"))
            log = summary.change_log_path.read_text(encoding="utf-8")

        self.assertEqual(document, candidate)
        self.assertIn("official pricing source failed", log)
        self.assertIn("retained all 1 existing models", log)
        self.assertEqual(1, summary.warning_count)


if __name__ == "__main__":
    unittest.main()
