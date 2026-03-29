#!/usr/bin/env python3
"""Verify a benchmark submission JSON file against the WSDP schema.

Uses manual validation (no external dependencies required).

Usage:
    python scripts/verify_submission.py benchmarks/submissions/widar_ResNet1D_john.json
"""

import json
import re
import sys
from pathlib import Path

VALID_DATASETS = ["widar", "gait", "xrf55", "elderAL", "zte"]
REQUIRED_FIELDS = ["model", "dataset", "accuracy_mean", "accuracy_std", "seeds", "submitter", "date"]
DATE_PATTERN = re.compile(r"^\d{4}-\d{2}-\d{2}$")
URI_PATTERN = re.compile(r"^https?://\S+$")


def validate_submission(filepath: str) -> list[str]:
    """Validate a submission JSON file. Returns a list of error messages (empty = valid)."""
    errors = []
    path = Path(filepath)

    # Check file exists and is valid JSON
    if not path.exists():
        return [f"File not found: {filepath}"]

    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        return [f"Invalid JSON: {e}"]

    if not isinstance(data, dict):
        return ["Top-level value must be a JSON object"]

    # Check required fields
    for field in REQUIRED_FIELDS:
        if field not in data:
            errors.append(f"Missing required field: '{field}'")

    # If critical fields are missing, return early
    if errors:
        return errors

    # Validate model
    if not isinstance(data["model"], str) or not data["model"].strip():
        errors.append("'model' must be a non-empty string")

    # Validate dataset
    if data["dataset"] not in VALID_DATASETS:
        errors.append(f"'dataset' must be one of {VALID_DATASETS}, got '{data['dataset']}'")

    # Validate accuracy_mean
    if not isinstance(data["accuracy_mean"], (int, float)):
        errors.append("'accuracy_mean' must be a number")
    elif not (0 <= data["accuracy_mean"] <= 1):
        errors.append(f"'accuracy_mean' must be between 0 and 1, got {data['accuracy_mean']}")

    # Validate accuracy_std
    if not isinstance(data["accuracy_std"], (int, float)):
        errors.append("'accuracy_std' must be a number")
    elif data["accuracy_std"] < 0:
        errors.append(f"'accuracy_std' must be >= 0, got {data['accuracy_std']}")

    # Validate seeds
    if not isinstance(data["seeds"], list):
        errors.append("'seeds' must be an array")
    elif len(data["seeds"]) < 3:
        errors.append(f"'seeds' must have at least 3 entries, got {len(data['seeds'])}")
    else:
        for i, seed in enumerate(data["seeds"]):
            if not isinstance(seed, int):
                errors.append(f"'seeds[{i}]' must be an integer, got {type(seed).__name__}")
                break

    # Validate submitter
    if not isinstance(data["submitter"], str) or not data["submitter"].strip():
        errors.append("'submitter' must be a non-empty string")

    # Validate date format
    if not isinstance(data["date"], str):
        errors.append("'date' must be a string")
    elif not DATE_PATTERN.match(data["date"]):
        errors.append(f"'date' must be in YYYY-MM-DD format, got '{data['date']}'")

    # Validate optional fields
    if "params_M" in data:
        if not isinstance(data["params_M"], (int, float)):
            errors.append("'params_M' must be a number")
        elif data["params_M"] < 0:
            errors.append(f"'params_M' must be >= 0, got {data['params_M']}")

    if "training_config" in data:
        if not isinstance(data["training_config"], dict):
            errors.append("'training_config' must be an object")

    if "paper_url" in data:
        if not isinstance(data["paper_url"], str) or not URI_PATTERN.match(data["paper_url"]):
            errors.append(f"'paper_url' must be a valid URI, got '{data.get('paper_url')}'")

    if "code_url" in data:
        if not isinstance(data["code_url"], str) or not URI_PATTERN.match(data["code_url"]):
            errors.append(f"'code_url' must be a valid URI, got '{data.get('code_url')}'")

    return errors


def main():
    if len(sys.argv) < 2:
        print("Usage: python verify_submission.py <submission.json>")
        sys.exit(1)

    filepath = sys.argv[1]
    print(f"Verifying: {filepath}")
    print("-" * 60)

    errors = validate_submission(filepath)

    if errors:
        print("FAIL - Validation errors found:")
        for err in errors:
            print(f"  - {err}")
        sys.exit(1)
    else:
        print("PASS - Submission is valid")
        sys.exit(0)


if __name__ == "__main__":
    main()
