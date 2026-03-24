#!/usr/bin/env python
from __future__ import annotations

import argparse
import os
from pathlib import Path
import json

import requests


ROOT = Path(__file__).resolve().parents[1]
SUBMISSION_URL = "https://www.ariel-datachallenge.space/api/score/calculate/"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Submit ExoHack prediction files.")
    parser.add_argument(
        "mu_path",
        nargs="?",
        type=Path,
        default=ROOT / "final_mu.csv",
        help="Path to the mean prediction CSV.",
    )
    parser.add_argument(
        "std_path",
        nargs="?",
        type=Path,
        default=ROOT / "final_std.csv",
        help="Path to the standard deviation prediction CSV.",
    )
    return parser.parse_args()


def load_credentials() -> dict[str, str]:
    credentials = {
        "secret_key": os.environ.get("EXOHACK_SECRET_KEY", "").strip(),
        "team_name": os.environ.get("EXOHACK_TEAM_NAME", "").strip(),
        "team_no": os.environ.get("EXOHACK_TEAM_NO", "").strip(),
    }
    missing = [key for key, value in credentials.items() if not value]
    if missing:
        missing_keys = ", ".join(missing)
        raise SystemExit(
            f"Missing submission credentials in environment: {missing_keys}. "
            "Set EXOHACK_SECRET_KEY, EXOHACK_TEAM_NAME, and EXOHACK_TEAM_NO."
        )
    return credentials


def main() -> None:
    args = parse_args()
    mu_path = args.mu_path.resolve()
    std_path = args.std_path.resolve()
    credentials = load_credentials()

    mu_text = mu_path.read_text(encoding="utf-8")
    std_text = std_path.read_text(encoding="utf-8")

    submission_files = {
        "mu_file": (mu_path.name, mu_text, "text/csv"),
        "std_file": (std_path.name, std_text, "text/csv"),
    }

    response = requests.post(
        SUBMISSION_URL,
        data=credentials,
        files=submission_files,
        timeout=120,
    )

    print(f"HTTP status code: {response.status_code}")
    print("Full response body:")
    print(response.text)

    if response.ok:
        try:
            payload = response.json()
        except json.JSONDecodeError:
            payload = {}
        if "score" in payload:
            print(f"Returned score: {payload['score']}")


if __name__ == "__main__":
    main()
