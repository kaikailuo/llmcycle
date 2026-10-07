from __future__ import annotations

import argparse
from pathlib import Path

from dotenv import load_dotenv

from src.registry import PROVIDERS
from src.runner import run


PROJECT_ROOT = Path(__file__).resolve().parent


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate model pricing candidates")
    parser.add_argument(
        "providers",
        nargs="*",
        default=list(PROVIDERS),
        help="provider registry keys (default: all registered providers)",
    )
    args = parser.parse_args()

    load_dotenv(PROJECT_ROOT / ".env")
    try:
        summary = run(PROJECT_ROOT, args.providers)
    except (OSError, ValueError) as exc:
        parser.exit(1, f"error: {exc}\n")

    print(f"Candidate: {summary.candidate_path}")
    print(f"Change log: {summary.change_log_path}")
    print(
        f"Models: {summary.model_count}; changed: {summary.changed_model_count}; "
        f"warnings: {summary.warning_count}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
