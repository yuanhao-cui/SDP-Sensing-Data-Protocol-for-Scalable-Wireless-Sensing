#!/usr/bin/env python3
"""Generate the leaderboard markdown tables from submission JSON files.

Reads all JSON files from benchmarks/submissions/, groups by dataset,
sorts by accuracy descending, and updates docs/leaderboard.md.

Usage:
    python scripts/generate_leaderboard.py
"""

import json
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SUBMISSIONS_DIR = REPO_ROOT / "benchmarks" / "submissions"
LEADERBOARD_PATH = REPO_ROOT / "docs" / "leaderboard.md"

DATASETS = ["widar", "gait", "xrf55", "elderAL", "zte"]

# Pattern to match leaderboard table sections
SECTION_PATTERN = re.compile(
    r"(<!-- LEADERBOARD_START:(\w+) -->).*?(<!-- LEADERBOARD_END:\2 -->)",
    re.DOTALL,
)


def load_submissions() -> dict[str, list[dict]]:
    """Load all submission JSON files, grouped by dataset."""
    grouped: dict[str, list[dict]] = {ds: [] for ds in DATASETS}

    if not SUBMISSIONS_DIR.exists():
        return grouped

    for path in SUBMISSIONS_DIR.glob("*.json"):
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except (json.JSONDecodeError, OSError) as e:
            print(f"Warning: skipping {path.name}: {e}", file=sys.stderr)
            continue

        dataset = data.get("dataset")
        if dataset in grouped:
            data["_filename"] = path.name
            grouped[dataset].append(data)

    # Sort each dataset by accuracy descending
    for ds in grouped:
        grouped[ds].sort(key=lambda x: x.get("accuracy_mean", 0), reverse=True)

    return grouped


def format_table(entries: list[dict]) -> str:
    """Generate a markdown table for a list of submissions."""
    lines = [
        "| Rank | Model | Accuracy (mean +/- std) | Params (M) | Seeds | Source |",
        "|------|-------|------------------------|------------|-------|--------|",
    ]

    if not entries:
        lines.append("| - | *No submissions yet* | - | - | - | - |")
        return "\n".join(lines)

    for rank, entry in enumerate(entries, 1):
        model = entry.get("model", "Unknown")
        mean = entry.get("accuracy_mean", 0)
        std = entry.get("accuracy_std", 0)
        accuracy = f"{mean:.3f} +/- {std:.3f}"
        params = f"{entry['params_M']:.1f}" if "params_M" in entry else "-"
        seeds = str(len(entry.get("seeds", [])))

        # Build source link
        source_parts = []
        if entry.get("paper_url"):
            source_parts.append(f"[Paper]({entry['paper_url']})")
        if entry.get("code_url"):
            source_parts.append(f"[Code]({entry['code_url']})")
        if not source_parts:
            source = entry.get("submitter", "-")
        else:
            source = " / ".join(source_parts)

        lines.append(f"| {rank} | {model} | {accuracy} | {params} | {seeds} | {source} |")

    return "\n".join(lines)


def update_leaderboard(grouped: dict[str, list[dict]]) -> None:
    """Update the leaderboard markdown file with generated tables."""
    if not LEADERBOARD_PATH.exists():
        print(f"Error: {LEADERBOARD_PATH} not found", file=sys.stderr)
        sys.exit(1)

    content = LEADERBOARD_PATH.read_text(encoding="utf-8")

    def replace_section(match: re.Match) -> str:
        start_tag = match.group(1)
        dataset = match.group(2)
        end_tag = match.group(3)

        entries = grouped.get(dataset, [])
        table = format_table(entries)
        return f"{start_tag}\n{table}\n{end_tag}"

    updated = SECTION_PATTERN.sub(replace_section, content)

    LEADERBOARD_PATH.write_text(updated, encoding="utf-8")
    print(f"Updated {LEADERBOARD_PATH}")

    # Summary
    total = sum(len(v) for v in grouped.values())
    print(f"Total submissions: {total}")
    for ds in DATASETS:
        count = len(grouped[ds])
        if count > 0:
            print(f"  {ds}: {count} entries")


def main():
    print("Loading submissions...")
    grouped = load_submissions()
    print("Updating leaderboard...")
    update_leaderboard(grouped)
    print("Done.")


if __name__ == "__main__":
    main()
