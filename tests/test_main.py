from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import main
from src.runner import RunSummary


class MainTests(unittest.TestCase):
    @patch("main.load_dotenv")
    @patch("main.run")
    def test_no_provider_arguments_runs_all_registered_providers(
        self, run_mock, load_dotenv_mock
    ) -> None:
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
        load_dotenv_mock.assert_called_once_with(main.PROJECT_ROOT / ".env")

    @patch("main.run")
    def test_loads_project_env_without_overriding_existing_environment(
        self, run_mock
    ) -> None:
        run_mock.return_value = RunSummary(
            candidate_path=Path("candidate.json"),
            change_log_path=Path("change.log"),
            model_count=0,
            changed_model_count=0,
            warning_count=0,
        )
        with tempfile.TemporaryDirectory() as directory:
            project_root = Path(directory)
            (project_root / ".env").write_text(
                "VOLCENGINE_ACCESS_KEY_ID=from-dotenv\n"
                "VOLCENGINE_SECRET_ACCESS_KEY=from-dotenv\n",
                encoding="utf-8",
            )
            with patch.object(main, "PROJECT_ROOT", project_root), patch.dict(
                os.environ,
                {"VOLCENGINE_ACCESS_KEY_ID": "from-environment"},
                clear=True,
            ), patch("sys.argv", ["main.py", "volcengine"]):
                self.assertEqual(0, main.main())
                self.assertEqual(
                    "from-environment", os.environ["VOLCENGINE_ACCESS_KEY_ID"]
                )
                self.assertEqual(
                    "from-dotenv", os.environ["VOLCENGINE_SECRET_ACCESS_KEY"]
                )

        run_mock.assert_called_once_with(project_root, ["volcengine"])


if __name__ == "__main__":
    unittest.main()
