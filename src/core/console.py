from __future__ import annotations
from pathlib import Path
from rich.console import Console

console = Console(record=True)


def save_report(output_dir: Path, stem: str = "pipeline_report") -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / f"{stem}.html").write_text(console.export_html())
