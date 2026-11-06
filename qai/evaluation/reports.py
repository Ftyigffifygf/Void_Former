"""Structured reporting to JSON, CSV, and Markdown."""

from __future__ import annotations
import os
import json
import csv
from typing import Dict, Any, List


class BenchmarkReporter:
    """Exports benchmark and validation reports in machine- and human-readable formats."""

    @staticmethod
    def export_json(data: Any, filepath: str) -> None:
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    @staticmethod
    def export_csv(rows: List[Dict[str, Any]], filepath: str) -> None:
        if not rows:
            return
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        keys = list(rows[0].keys())
        with open(filepath, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(rows)

    @staticmethod
    def generate_markdown_table(rows: List[Dict[str, Any]]) -> str:
        if not rows:
            return ""
        keys = list(rows[0].keys())
        header = "| " + " | ".join(keys) + " |"
        sep = "| " + " | ".join(["---"] * len(keys)) + " |"
        body = []
        for row in rows:
            body.append("| " + " | ".join(str(row.get(k, "")) for k in keys) + " |")
        return "\n".join([header, sep] + body)
