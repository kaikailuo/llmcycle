from __future__ import annotations

import unittest
from pathlib import Path
from unittest.mock import patch

import main
from src.runner import RunSummary


class MainTests(unittest.TestCase):
    @patch("main.run")
    def test_no_provider_arguments_runs_all_registered_providers(self, run_mock) -> None:
        run_mock.return_value = RunSummary(
            candidate_path=Path("candidate.json"),
            change_log_path=Path("change.log"),
            model_count=0,
            changed_model_count=0,
            warning_count=0,
        )

        with patch("sys.argv", ["main.py"]):
            self.assertEqual(0, main.main())

        self.assertEqual(list(main.PROVIDERS), run_mock.call_args.args[1])


if __name__ == "__main__":
    unittest.main()
