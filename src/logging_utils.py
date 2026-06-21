"""Shared logging and terminal-presentation utilities.

Provides a single `rich`-backed `Console` instance and a configured
`logging.Logger` so every pipeline script (Random Forest, XGBoost, CNN)
prints with a consistent, readable visual identity instead of ad-hoc
`print()` calls.

Typical usage::

    from logging_utils import get_logger, console, section

    log = get_logger(__name__)
    section("Loading data")
    log.info("Loaded %d rows", len(df))
"""

from __future__ import annotations

import logging

from rich.console import Console
from rich.logging import RichHandler
from rich.table import Table

console: Console = Console()

_CONFIGURED = False


def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """Return a `rich`-formatted logger shared across the project.

    Configures the root logging handler exactly once (idempotent across
    repeated calls/imports, which matters in notebooks where cells may
    re-run), so colorized, leveled, timestamped logs replace raw
    ``print()`` statements throughout the codebase.

    Args:
        name: Logger name, conventionally ``__name__`` of the caller.
        level: Minimum severity to emit. Defaults to ``logging.INFO``.

    Returns:
        A configured `logging.Logger` instance.
    """
    global _CONFIGURED
    if not _CONFIGURED:
        logging.basicConfig(
            level=level,
            format="%(message)s",
            datefmt="[%X]",
            handlers=[RichHandler(console=console, rich_tracebacks=True, markup=True)],
        )
        _CONFIGURED = True
    return logging.getLogger(name)


def section(title: str) -> None:
    """Print a styled section banner to delineate pipeline stages.

    Args:
        title: Short label for the stage about to run, e.g. ``"Training"``.
    """
    console.rule(f"[bold cyan]{title}[/bold cyan]", style="cyan")


def metrics_table(title: str, metrics: dict[str, float], precision: int = 4) -> Table:
    """Build a `rich.Table` rendering a metric-name/value pair set.

    Args:
        title: Table title shown above the rendered rows.
        metrics: Mapping of metric name to scalar value.
        precision: Number of decimal places used to format each value.

    Returns:
        A `rich.table.Table` ready to be passed to `console.print`.
    """
    table = Table(title=title, header_style="bold magenta")
    table.add_column("Metric", style="cyan", no_wrap=True)
    table.add_column("Value", style="green", justify="right")
    for key, value in metrics.items():
        table.add_row(str(key), f"{value:.{precision}f}")
    return table
