"""CLI for julio."""

# Authors: Synchon Mandal <s.mandal@fz-juelich.de>
# License: AGPL

import logging
import logging.config
import pathlib
import sys
from pathlib import Path

import click
import datalad
import structlog

from . import _functions as cli_func
from . import _utils as cli_utils


__all__ = ["cli", "create"]

# Common processors for stdlib and structlog
_timestamper = structlog.processors.TimeStamper(fmt="%Y-%m-%d %H:%M:%S")


def _remove_datalad_message(_, __, event_dict):
    """Clean datalad records."""
    if "message" in event_dict:
        event_dict.pop("message")
    if "dlm_progress" in event_dict:
        event_dict.pop("dlm_progress")
    if "dlm_progress_noninteractive_level" in event_dict:
        event_dict.pop("dlm_progress_noninteractive_level")
    if "dlm_progress_update" in event_dict:
        event_dict.pop("dlm_progress_update")
    if "dlm_progress_label" in event_dict:
        event_dict.pop("dlm_progress_label")
    if "dlm_progress_unit" in event_dict:
        event_dict.pop("dlm_progress_unit")
    if "dlm_progress_total" in event_dict:
        event_dict.pop("dlm_progress_total")
    return event_dict


_pre_chain = [
    structlog.stdlib.add_log_level,
    structlog.stdlib.add_logger_name,
    structlog.stdlib.ExtraAdder(),
    _timestamper,
    _remove_datalad_message,
]


def _set_log_config(verbose: int) -> None:
    """Set logging config.

    Parameters
    ----------
    verbose : int
        Verbosity.

    """
    # Configure logger based on verbosity
    if verbose == 0:
        level = logging.WARNING
    elif verbose == 1:
        level = logging.INFO
    else:
        level = logging.DEBUG
    # Configure stdlib
    logging.config.dictConfig(
        {
            "version": 1,
            "disable_existing_loggers": False,
            "formatters": {
                "console": {
                    "()": structlog.stdlib.ProcessorFormatter,
                    "processors": [
                        structlog.stdlib.add_logger_name,
                        structlog.stdlib.ProcessorFormatter.remove_processors_meta,
                        structlog.dev.ConsoleRenderer(
                            colors=sys.stdout.isatty() and sys.stderr.isatty()
                        ),
                    ],
                    "foreign_pre_chain": _pre_chain,
                },
            },
            "handlers": {
                "default": {
                    "level": level,
                    "class": "logging.StreamHandler",
                    "formatter": "console",
                },
            },
            "loggers": {
                "": {
                    "handlers": ["default"],
                    "level": level,
                    "propagate": True,
                },
            },
        }
    )
    # Configure structlog
    structlog.configure(
        processors=[
            structlog.stdlib.add_log_level,
            structlog.stdlib.PositionalArgumentsFormatter(),
            structlog.processors.TimeStamper(fmt="%Y-%m-%d %H:%M:%S"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
        ],
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.make_filtering_bound_logger(level),
        cache_logger_on_first_use=True,
    )
    # Remove datalad logger handlers to avoid duplicate logging
    _datalad_lgr_hdlrs = datalad.log.lgr.handlers
    for h in _datalad_lgr_hdlrs:
        datalad.log.lgr.removeHandler(h)
    datalad.log.lgr.setLevel(level)


@click.group
@click.version_option(prog_name="julio")
@click.help_option()
def cli() -> None:
    """julio CLI."""  # noqa: D403


@cli.command
@click.argument(
    "registry_path",
    type=click.Path(
        exists=False,
        readable=True,
        writable=True,
        file_okay=False,
        path_type=pathlib.Path,
    ),
    metavar="<registry>",
)
@click.option("-v", "--verbose", count=True, type=int)
def create(
    registry_path: click.Path,
    verbose: int,
) -> None:
    """Create registry."""
    _set_log_config(verbose)
    try:
        cli_func.create(registry_path)
    except RuntimeError as err:
        click.echo(f"{err}", err=True)
    else:
        click.echo("Success")


@cli.command
@click.argument(
    "yaml_path",
    type=click.Path(
        exists=True,
        readable=True,
        writable=True,
        dir_okay=False,
        path_type=pathlib.Path,
    ),
    metavar="<yaml>",
)
@click.option(
    "-r",
    "--registry",
    default=None,
    type=cli_utils.PathOrURL,
    metavar="<registry>",
    help="Path to registry; if not passed, use current directory",
)
@click.option("-v", "--verbose", count=True, type=int)
def add(
    yaml_path: click.Path,
    registry: click.Path,
    verbose: int,
) -> None:
    """Add feature(s) registry."""
    _set_log_config(verbose)
    try:
        cli_func.add(
            yaml_path, registry if registry is not None else Path(".")
        )
    except RuntimeError as err:
        click.echo(f"{err}", err=True)
    else:
        click.echo("Success")
