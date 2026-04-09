from __future__ import annotations
import argparse
import sys
from pathlib import Path
from dotenv import load_dotenv

from rich.prompt import Prompt
from rich.panel import Panel
from rich.table import Table

load_dotenv()

from src.core.console import console  # noqa: E402
from src.core.pipeline import run  # noqa: E402

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

def interactive_mode_selection() -> str:
    from rich.live import Live

    modes = [
        ("extraction", "Plans 1 & 2 — Extraction quality & temporal tracking"),
        ("complexity", "Plan 3  — Complexity ladder (scene density scaling)"),
    ]
    current_index = 0

    def generate_table() -> Table:
        table = Table(
            title="[bold cyan]Benchmark Mode[/bold cyan]\n[dim](↑↓: Navigate, Enter: Select)[/dim]",
            box=None,
        )
        table.add_column("", justify="center", width=3)
        table.add_column("Mode",        style="green", width=14)
        table.add_column("Description", style="dim")

        for i, (mode, desc) in enumerate(modes):
            prefix = ">" if i == current_index else " "
            style  = "bold white on blue" if i == current_index else ""
            table.add_row(prefix, mode, desc, style=style)
        return table

    with Live(generate_table(), auto_refresh=False, console=console) as live:
        while True:
            live.update(generate_table(), refresh=True)
            key = get_key()
            if key == "\x1b[A":
                current_index = (current_index - 1) % len(modes)
            elif key == "\x1b[B":
                current_index = (current_index + 1) % len(modes)
            elif key in ("\r", "\n"):
                break
            elif key == "\x03":
                sys.exit(0)

    return modes[current_index][0]


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
    parser.add_argument("mode", nargs="?", choices=["extraction", "complexity"],
                        default=None,
                        help="Benchmark mode: extraction (Plans 1 & 2) or complexity (Plan 3). "
                             "Prompted interactively if omitted.")
    parser.add_argument("--models",       type=str,  help="Comma-separated list of models to run")
    parser.add_argument("--models-cfg",   type=Path, default=Path("configs/models.yaml"))
    parser.add_argument("--clips-cfg",    type=Path, default=None,
                        help="Path to clips config. Defaults to clips.yaml (extraction) "
                             "or clips_complexity.yaml (complexity) based on mode.")
    parser.add_argument("--benchmark-cfg",type=Path, default=Path("configs/benchmark.yaml"))
    parser.add_argument("--non-interactive", action="store_true",
                        help="Disable all prompts (uses --models or all models, defaults to extraction mode)")
    parser.add_argument("-c", "--use-config", action="store_true",
                        help="Use only 'enabled: true' models from models.yaml")
    parser.add_argument("--debug", action="store_true",
                        help="Enable debug logging (per-frame inference details)")

    args = parser.parse_args()

    if args.debug:
        import src.core.utils as _utils
        _utils.DEBUG = True
        console.print("[dim cyan][DBG] Debug mode enabled[/dim cyan]")

    console.print(Panel(
        "[bold magenta]Welcome to Semora Benchmark[/bold magenta]\n"
        "Modular evaluation system for VLMs",
        border_style="blue",
    ))

    # ── Determine mode ────────────────────────────────────────────────────────
    mode = args.mode
    if mode is None:
        mode = "extraction" if args.non_interactive else interactive_mode_selection()

    # ── Resolve clips config based on mode (--clips-cfg always wins) ────────
    if args.clips_cfg is None:
        if mode == "complexity":
            args.clips_cfg = Path("configs/clips_complexity.yaml")
        else:
            args.clips_cfg = Path("configs/clips.yaml")

    # ── Mode: extraction (Plans 1 & 2) ────────────────────────────────────────
    available_models = get_available_models(args.models_cfg)
    selected_models = None

    if args.models:
        selected_models = [m.strip() for m in args.models.split(",")]
        invalid = [m for m in selected_models if m not in available_models]
        if invalid:
            console.print(f"[bold red]Warning: Ignoring unknown models: {invalid}[/bold red]")
            selected_models = [m for m in selected_models if m in available_models]

    if not selected_models and not args.use_config and not args.non_interactive:
        selected_models = interactive_model_selection(available_models)

    if not selected_models and not args.use_config:
        selected_models = available_models

    # ── Run benchmark ─────────────────────────────────────────────────────────
    if mode == "complexity":
        from src.core.pipeline_complexity import run_complexity
        results_dir = run_complexity(
            models_cfg_path=args.models_cfg,
            clips_cfg_path=args.clips_cfg,
            benchmark_cfg_path=args.benchmark_cfg,
            selected_models=selected_models,
        )
    else:
        results_dir = run(
            models_cfg_path=args.models_cfg,
            clips_cfg_path=args.clips_cfg,
            benchmark_cfg_path=args.benchmark_cfg,
            selected_models=selected_models,
        )

    # ── Post-run: always generate full report ─────────────────────────────────
    report_dir = results_dir / "report"

    if mode == "complexity":
        from src.analysis import analyze_complexity as _ac
        from src.analysis import plot_complexity as _pc

        console.print("\n[bold cyan]── Complexity Analysis ───────────────────────[/bold cyan]")
        _ac.run_complexity_analysis(results_dir, report_dir)

        console.print("\n[bold cyan]── Complexity Plots ──────────────────────────[/bold cyan]")
        _pc.run_complexity_plots(results_dir, report_dir)

        _ac.save_report(report_dir)
        console.print(f"[dim]reports saved → {report_dir}/[/dim]")

    else:
        from src.analysis import analyze as _analyze
        from src.analysis import plot as _plot

        console.print("\n[bold cyan]── Analysis ──────────────────────────────────[/bold cyan]")
        _analyze.print_scores_table(results_dir)
        _analyze.run_temporal_analysis(results_dir, report_dir)
        _analyze.save_field_guide(report_dir)
        console.print(f"[dim]Field guide → {report_dir}/field_guide.html[/dim]")

        console.print("\n[bold cyan]── Plots ─────────────────────────────────────[/bold cyan]")
        _plot.run_all_plots(results_dir, report_dir)

        _analyze.save_report(report_dir)
        console.print(f"[dim]reports saved → {report_dir}/[/dim]")

if __name__ == "__main__":
    main()
