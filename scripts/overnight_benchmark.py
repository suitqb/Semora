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


def interactive_tracking_selection(current: bool, mode: str = "extraction") -> bool:
    from rich.live import Live

    options = [False, True]
    current_index = 1 if current else 0

    if mode == "complexity":
        title = "[bold cyan]Detection Context (YOLO — single frame)[/bold cyan]\n[dim](↑↓: Navigate, Enter: Select)[/dim]"
        rows = [
            (False, "[red]OFF[/red]", "Standard prompt — no YOLO pre-detection"),
            (True,  "[green]ON[/green]",  "Inject YOLO bbox detections into the prompt before inference"),
        ]
    else:
        title = "[bold cyan]Tracking Context (YOLO+ByteTrack)[/bold cyan]\n[dim](↑↓: Navigate, Enter: Select)[/dim]"
        rows = [
            (False, "[red]OFF[/red]", "Standard prompt — no tracking context"),
            (True,  "[green]ON[/green]",  "Inject YOLO+ByteTrack track IDs into the prompt (live, per window)"),
        ]

    def generate_table() -> Table:
        table = Table(title=title, box=None)
        table.add_column("", justify="center", width=3)
        table.add_column("", width=6)
        table.add_column("Description", style="dim")

        for i, (_, label, desc) in enumerate(rows):
            prefix = ">" if i == current_index else " "
            style  = "bold white on blue" if i == current_index else ""
            table.add_row(prefix, label, desc, style=style)
        return table

    with Live(generate_table(), auto_refresh=False, console=console) as live:
        while True:
            live.update(generate_table(), refresh=True)
            key = get_key()
            if key == "\x1b[A":
                current_index = (current_index - 1) % len(options)
            elif key == "\x1b[B":
                current_index = (current_index + 1) % len(options)
            elif key in ("\r", "\n"):
                break
            elif key == "\x03":
                sys.exit(0)

    return options[current_index]


def interactive_multi_crop_selection(current: bool) -> bool:
    from rich.live import Live

    options = [False, True]
    current_index = 1 if current else 0

    rows = [
        (False, "[red]OFF[/red]", "Standard — full frame only"),
        (True,  "[green]ON[/green]",  "Append entity crops after full frame (YOLO bboxes + padding)"),
    ]
    title = "[bold cyan]Multi-Crop Context[/bold cyan]\n[dim](↑↓: Navigate, Enter: Select)[/dim]"

    def generate_table() -> Table:
        table = Table(title=title, box=None)
        table.add_column("", justify="center", width=3)
        table.add_column("", width=6)
        table.add_column("Description", style="dim")

        for i, (_, label, desc) in enumerate(rows):
            prefix = ">" if i == current_index else " "
            style  = "bold white on blue" if i == current_index else ""
            table.add_row(prefix, label, desc, style=style)
        return table

    with Live(generate_table(), auto_refresh=False, console=console) as live:
        while True:
            live.update(generate_table(), refresh=True)
            key = get_key()
            if key == "\x1b[A":
                current_index = (current_index - 1) % len(options)
            elif key == "\x1b[B":
                current_index = (current_index + 1) % len(options)
            elif key in ("\r", "\n"):
                break
            elif key == "\x03":
                sys.exit(0)

    return options[current_index]


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
                        help="Path to clips config. Defaults to clips_extraction.yaml (extraction) "
                             "or clips_complexity.yaml (complexity) based on mode.")
    parser.add_argument("--benchmark-cfg",type=Path, default=Path("configs/benchmark.yaml"))
    parser.add_argument("--non-interactive", action="store_true",
                        help="Disable all prompts (uses --models or all models, defaults to extraction mode)")
    parser.add_argument("-c", "--use-config", action="store_true",
                        help="Use only 'enabled: true' models from models.yaml")
    parser.add_argument("--run-id",      type=str, default=None,
                        help="Human-readable run identifier (used as directory name under runs/). "
                             "Defaults to YYYYMMDD_HHMMSS timestamp.")
    parser.add_argument("--tracking",   type=lambda x: x.lower() == "true", default=None,
                        metavar="true|false",
                        help="Override tracking feature (bypasses interactive prompt and benchmark.yaml)")
    parser.add_argument("--multi-crop", type=lambda x: x.lower() == "true", default=None,
                        metavar="true|false",
                        help="Override multi_crop feature (bypasses interactive prompt and benchmark.yaml)")
    parser.add_argument("--max-resolution", type=str, default=None,
                        metavar="WxH",
                        help="Override max resolution (e.g. 640x480). Defaults to clips config value.")
    parser.add_argument("--runs", type=str, default=None,
                        metavar="N-M|N,M,...",
                        help="Run a batch of predefined conditions (e.g. --runs 1-4 or --runs 1,3,5). "
                             "Use --list-runs to see all conditions.")
    parser.add_argument("--list-runs", action="store_true",
                        help="List all predefined run conditions and exit.")
    parser.add_argument("--debug", action="store_true",
                        help="Enable debug logging (per-frame inference details)")

    args = parser.parse_args()

    # ── Predefined conditions ─────────────────────────────────────────────────
    # Each entry: (label, mode, tracking, multi_crop, resolution_or_None)
    _CONDITIONS = [
        ( 1, "extraction | baseline      | 1280x720", "extraction", False, False, "1280x720"),
        ( 2, "extraction | tracking      | 1280x720", "extraction", True,  False, "1280x720"),
        ( 3, "extraction | crop          | 1280x720", "extraction", False, True,  "1280x720"),
        ( 4, "extraction | tracking+crop | 1280x720", "extraction", True,  True,  "1280x720"),
        ( 5, "extraction | baseline      | 640x480",  "extraction", False, False, "640x480"),
        ( 6, "extraction | tracking      | 640x480",  "extraction", True,  False, "640x480"),
        ( 7, "extraction | crop          | 640x480",  "extraction", False, True,  "640x480"),
        ( 8, "extraction | tracking+crop | 640x480",  "extraction", True,  True,  "640x480"),
        ( 9, "complexity | baseline",                 "complexity", False, False, None),
        (10, "complexity | tracking",                 "complexity", True,  False, None),
        (11, "complexity | crop",                     "complexity", False, True,  None),
        (12, "complexity | tracking+crop",            "complexity", True,  True,  None),
    ]

    if args.list_runs:
        from rich.table import Table as _Table
        t = _Table(title="Predefined run conditions", box=None, header_style="bold cyan")
        t.add_column("#",           justify="right",  width=4,  no_wrap=True)
        t.add_column("Label",       style="green",    width=36, no_wrap=True)
        t.add_column("Mode",        width=12,         no_wrap=True)
        t.add_column("Tracking",    justify="center", width=9,  no_wrap=True)
        t.add_column("Crop",        justify="center", width=6,  no_wrap=True)
        t.add_column("Resolution",  justify="center", width=10, no_wrap=True)
        for idx, label, mode_, trk, crp, res in _CONDITIONS:
            t.add_row(str(idx), label, mode_,
                      "[green]ON[/green]" if trk else "[red]OFF[/red]",
                      "[green]ON[/green]" if crp else "[red]OFF[/red]",
                      res or "default")
        console.print(t)
        sys.exit(0)

    if args.runs:
        # Parse "all", "1-4" or "1,3,5-7" into a set of indices
        enabled = set()
        if args.runs.strip().lower() == "all":
            enabled = {c[0] for c in _CONDITIONS}
        else:
            for part in args.runs.split(","):
                if "-" in part:
                    a, b = part.split("-", 1)
                    enabled.update(range(int(a), int(b) + 1))
                else:
                    enabled.add(int(part))

        selected_conditions = [c for c in _CONDITIONS if c[0] in enabled]
        if not selected_conditions:
            console.print("[bold red]No valid conditions found for --runs value.[/bold red]")
            sys.exit(1)

        if args.debug:
            import src.core.utils as _utils
            _utils.DEBUG = True

        # Resolve models once
        selected_models_batch = None
        if args.models:
            available = get_available_models(args.models_cfg)
            selected_models_batch = [m.strip() for m in args.models.split(",") if m.strip() in available]
        # (if not set, -c / use_config will be honoured by each run call)

        console.print(Panel(
            "[bold magenta]Welcome to Semora Benchmark[/bold magenta]\n"
            "Modular evaluation system for VLMs",
            border_style="blue",
        ))
        console.print(f"[bold cyan]Batch mode — running conditions: {sorted(enabled)}[/bold cyan]\n")

        for idx, label, mode_, trk, crp, res_str in selected_conditions:
            res = None
            if res_str:
                w, h = res_str.split("x")
                res = (int(w), int(h))

            clips_cfg = Path("configs/clips_complexity.yaml") if mode_ == "complexity" \
                        else Path("configs/clips_extraction.yaml")

            from datetime import datetime as _dt
            run_id = f"{label.replace(' ', '').replace('|','_')}_{_dt.now().strftime('%Y%m%d_%H%M%S')}"

            console.print(f"\n[bold yellow]▶ [{idx:2d}] {label}[/bold yellow]")

            try:
                if mode_ == "complexity":
                    from src.core.pipeline_complexity import run_complexity
                    results_dir = run_complexity(
                        models_cfg_path=args.models_cfg,
                        clips_cfg_path=clips_cfg,
                        benchmark_cfg_path=args.benchmark_cfg,
                        selected_models=selected_models_batch,
                        tracking=trk,
                        run_id=run_id,
                        max_resolution=res,
                    )
                else:
                    results_dir = run(
                        models_cfg_path=args.models_cfg,
                        clips_cfg_path=clips_cfg,
                        benchmark_cfg_path=args.benchmark_cfg,
                        selected_models=selected_models_batch,
                        tracking=trk,
                        multi_crop=crp,
                        run_id=run_id,
                        max_resolution=res,
                    )
                console.print(f"[bold green]✔ [{idx:2d}] Done → {results_dir / 'raw'}[/bold green]")
            except Exception as e:
                console.print(f"[bold red]✗ [{idx:2d}] Failed: {e}[/bold red]")

        sys.exit(0)

    max_resolution = None
    if args.max_resolution:
        try:
            w, h = args.max_resolution.lower().split("x")
            max_resolution = (int(w), int(h))
        except ValueError:
            console.print(f"[bold red]Invalid --max-resolution '{args.max_resolution}', expected WxH (e.g. 640x480)[/bold red]")
            sys.exit(1)

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
            args.clips_cfg = Path("configs/clips_extraction.yaml")

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

    # ── Tracking / detection + multi-crop selection ──────────────────────────────
    # CLI flags (--tracking / --multi-crop) take priority over interactive prompts
    tracking_override:   bool | None = args.tracking
    multi_crop_override: bool | None = args.multi_crop

    if not args.non_interactive and tracking_override is None:
        import yaml as _yaml
        with open(args.benchmark_cfg) as _f:
            _bcfg = _yaml.safe_load(_f)
        _features = _bcfg.get("benchmark", {}).get("features", {})
        _current_tracking   = _features.get("tracking",   False)
        _current_multi_crop = _features.get("multi_crop", False)
        tracking_override = interactive_tracking_selection(_current_tracking, mode=mode)
        if mode != "complexity" and multi_crop_override is None:
            multi_crop_override = interactive_multi_crop_selection(_current_multi_crop)

    # ── Run benchmark ─────────────────────────────────────────────────────────
    if mode == "complexity":
        from src.core.pipeline_complexity import run_complexity
        results_dir = run_complexity(
            models_cfg_path=args.models_cfg,
            clips_cfg_path=args.clips_cfg,
            benchmark_cfg_path=args.benchmark_cfg,
            selected_models=selected_models,
            tracking=tracking_override,
            run_id=args.run_id,
            max_resolution=max_resolution,
        )
    else:
        results_dir = run(
            models_cfg_path=args.models_cfg,
            clips_cfg_path=args.clips_cfg,
            benchmark_cfg_path=args.benchmark_cfg,
            selected_models=selected_models,
            tracking=tracking_override,
            multi_crop=multi_crop_override,
            run_id=args.run_id,
            max_resolution=max_resolution,
        )

    console.print(f"\n[bold green]✓ Raw results saved → {results_dir / 'raw'}[/bold green]")
    console.print("[dim]Run [bold]streamlit run app.py[/bold] to explore results.[/dim]")

if __name__ == "__main__":
    main()
