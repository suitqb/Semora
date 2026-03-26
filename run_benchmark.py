from __future__ import annotations
import argparse
import sys
from pathlib import Path
from dotenv import load_dotenv

from rich.console import Console
from rich.prompt import Prompt
from rich.panel import Panel
from rich.table import Table

load_dotenv()

from src.core.pipeline import run # noqa: E402

console = Console()

def get_available_models(models_cfg_path: Path) -> list[str]:
    import yaml
    with open(models_cfg_path) as f:
        cfg = yaml.safe_load(f)
    return list(cfg.get("models", {}).keys())

import termios
import tty

def get_key() -> str:
    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    try:
        tty.setraw(sys.stdin.fileno())
        ch = sys.stdin.read(1)
        if ch == '\x1b':
            ch += sys.stdin.read(2)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
    return ch

def interactive_model_selection(available_models: list[str]) -> list[str]:
    from rich.live import Live
    
    selected_mask = [False] * len(available_models)
    current_index = 0
    
    def generate_table() -> Table:
        table = Table(title="[bold cyan]Model Selection[/bold cyan]\n[dim](Space: Toggle, Enter: Validate, A: Select All, N: Select None)[/dim]", box=None)
        table.add_column("Status", justify="center", width=10)
        table.add_column("Model", style="green")
        
        for i, model in enumerate(available_models):
            prefix = "> " if i == current_index else "  "
            status = "[bold green][X][/bold green]" if selected_mask[i] else "[ ]"
            style = "bold white on blue" if i == current_index else ""
            table.add_row(f"{prefix}{status}", model, style=style)
        return table

    with Live(generate_table(), auto_refresh=False, console=console) as live:
        while True:
            live.update(generate_table(), refresh=True)
            key = get_key()
            
            if key == '\x1b[A': # Up arrow
                current_index = (current_index - 1) % len(available_models)
            elif key == '\x1b[B': # Down arrow
                current_index = (current_index + 1) % len(available_models)
            elif key == ' ': # Space bar
                selected_mask[current_index] = not selected_mask[current_index]
            elif key.lower() == 'a': # Select all
                selected_mask = [True] * len(available_models)
            elif key.lower() == 'n': # Deselect all
                selected_mask = [False] * len(available_models)
            elif key == '\r' or key == '\n': # Enter key
                break
            elif key == '\x03': # Ctrl+C to exit
                sys.exit(0)

    selected_models = [model for i, model in enumerate(available_models) if selected_mask[i]]
    if not selected_models:
        console.print("[yellow]No model selected, using all models by default.[/yellow]")
        return available_models
    return selected_models

def main() -> None:
    parser = argparse.ArgumentParser(description="VLM Benchmark — Semora")
    parser.add_argument("--models",    type=str, help="List of models to run (comma separated)")
    parser.add_argument("--models-cfg", type=Path, default=Path("configs/models.yaml"), help="Path to models.yaml")
    parser.add_argument("--clips-cfg",  type=Path, default=Path("configs/clips.yaml"), help="Path to clips.yaml")
    parser.add_argument("--benchmark-cfg", type=Path, default=Path("configs/benchmark.yaml"), help="Path to benchmark.yaml")
    parser.add_argument("--non-interactive", action="store_true", help="Disable interactivity (uses --models argument or all available models)")
    parser.add_argument("-c", "--use-config", action="store_true", help="Use only 'enabled: true' models from models.yaml (skips interactive selection)")
    
    args = parser.parse_args()

    available_models = get_available_models(args.models_cfg)
    selected_models = None

    if args.models:
        selected_models = [m.strip() for m in args.models.split(",")]
        # Basic verification of model names
        invalid = [m for m in selected_models if m not in available_models]
        if invalid:
            console.print(f"[bold red]Warning: Ignoring unknown models: {invalid}[/bold red]")
            selected_models = [m for m in selected_models if m in available_models]

    # Interactive mode default if:
    # 1. No models passed via --models
    # 2. Config usage was not forced via --use-config / -c
    # 3. Not in --non-interactive mode
    if not selected_models and not args.use_config and not args.non_interactive:
        console.print(Panel("[bold magenta]Welcome to Semora Benchmark[/bold magenta]\n"
                            "Modular evaluation system for VLMs", 
                            border_style="blue"))
        selected_models = interactive_model_selection(available_models)

    # If in --non-interactive mode but no --models provided, default to all models
    if not selected_models and args.non_interactive and not args.use_config:
        selected_models = available_models

    run(
        models_cfg_path=args.models_cfg,
        clips_cfg_path=args.clips_cfg,
        benchmark_cfg_path=args.benchmark_cfg,
        selected_models=selected_models
    )

if __name__ == "__main__":
    main()
