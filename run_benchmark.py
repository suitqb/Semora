from __future__ import annotations
import argparse
from pathlib import Path

from dotenv import load_dotenv


load_dotenv()  # charge .env avant tout import de backend

from src.core.pipeline import run # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="VLM Benchmark — TITAN")
    parser.add_argument("--models",    type=Path, default=Path("configs/models.yaml"))
    parser.add_argument("--clips",     type=Path, default=Path("configs/clips.yaml"))
    parser.add_argument("--benchmark", type=Path, default=Path("configs/benchmark.yaml"))
    args = parser.parse_args()

    run(
        models_cfg_path=args.models,
        clips_cfg_path=args.clips,
        benchmark_cfg_path=args.benchmark,
    )


if __name__ == "__main__":
    main()
