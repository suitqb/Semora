from __future__ import annotations
import argparse
import sys
from pathlib import Path
from dotenv import load_dotenv

from rich.console import Console
from rich.prompt import Prompt, IntPrompt
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
        table = Table(title="[bold cyan]Sélection des Modèles[/bold cyan]\n[dim](Espace: Cocher, Entrée: Valider, A: Tout cocher, N: Tout décocher)[/dim]", box=None)
        table.add_column("Statut", justify="center", width=10)
        table.add_column("Modèle", style="green")
        
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
            
            if key == '\x1b[A': # Up
                current_index = (current_index - 1) % len(available_models)
            elif key == '\x1b[B': # Down
                current_index = (current_index + 1) % len(available_models)
            elif key == ' ': # Space
                selected_mask[current_index] = not selected_mask[current_index]
            elif key.lower() == 'a': # All
                selected_mask = [True] * len(available_models)
            elif key.lower() == 'n': # None
                selected_mask = [False] * len(available_models)
            elif key == '\r' or key == '\n': # Enter
                break
            elif key == '\x03': # Ctrl+C
                sys.exit(0)

    selected_models = [model for i, model in enumerate(available_models) if selected_mask[i]]
    if not selected_models:
        console.print("[yellow]Aucun modèle sélectionné, utilisation de tous les modèles par défaut.[/yellow]")
        return available_models
    return selected_models

def main() -> None:
    parser = argparse.ArgumentParser(description="VLM Benchmark — Semora")
    parser.add_argument("--models",    type=str, help="Liste de modèles à faire tourner (séparés par des virgules)")
    parser.add_argument("--models-cfg", type=Path, default=Path("configs/models.yaml"), help="Chemin vers models.yaml")
    parser.add_argument("--clips-cfg",  type=Path, default=Path("configs/clips.yaml"), help="Chemin vers clips.yaml")
    parser.add_argument("--benchmark-cfg", type=Path, default=Path("configs/benchmark.yaml"), help="Chemin vers benchmark.yaml")
    parser.add_argument("--non-interactive", action="store_true", help="Désactive l'interactivité (utilise les modèles de l'argument --models ou sinon tous)")
    parser.add_argument("-c", "--use-config", action="store_true", help="Utilise uniquement la configuration 'enabled' de models.yaml (skip l'interactif)")
    
    args = parser.parse_args()

    available_models = get_available_models(args.models_cfg)
    selected_models = None

    if args.models:
        selected_models = [m.strip() for m in args.models.split(",")]
        # Vérification sommaire
        invalid = [m for m in selected_models if m not in available_models]
        if invalid:
            console.print(f"[bold red]Attention : Modèles inconnus ignorés : {invalid}[/bold red]")
            selected_models = [m for m in selected_models if m in available_models]

    # Mode interactif par défaut si :
    # 1. Aucun modèle n'est passé via --models
    # 2. On n'a pas forcé l'usage de la config via --use-config / -c
    # 3. On n'est pas en mode --non-interactive
    if not selected_models and not args.use_config and not args.non_interactive:
        console.print(Panel("[bold magenta]Bienvenue dans Semora Benchmark[/bold magenta]\n"
                            "Système d'évaluation modulaire pour VLMs", 
                            border_style="blue"))
        selected_models = interactive_model_selection(available_models)

    # Si on est en --non-interactive mais sans --models, par défaut on prend tout
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
