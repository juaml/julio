"""Functions for julio."""

# Authors: Synchon Mandal <s.mandal@fz-juelich.de>
# License: AGPL

import tempfile
from pathlib import Path

import datalad.api as dl
import structlog
from datalad.support.exceptions import IncompleteResultsError

from ._utils import is_julio_registry, process_features


__all__ = ["add", "create"]


logger = structlog.get_logger()


def create(registry_path: Path):
    """Create a registry at `registry_path`.

    Parameters
    ----------
    registry_path : pathlib.Path
        Path to the registry.

    Raises
    ------
    RuntimeError
        If there is a problem creating the registry.

    """
    try:
        ds = dl.create(
            path=registry_path,
            cfg_proc="text2git",
            on_failure="stop",
            result_renderer="disabled",
        )
    except IncompleteResultsError as e:
        raise RuntimeError(f"Failed to create dataset: {e.failed}") from e
    else:
        logger.debug(
            "Registry created successfully",
            cmd="create",
            path=str(registry_path.resolve()),
        )
    # Add config file
    conf_path = Path(ds.path) / "registry-config.yml"
    conf_path.touch()
    ds.save(
        conf_path,
        message="[julio] add registry configuration",
        on_failure="stop",
        result_renderer="disabled",
    )


def add(yaml_path: Path, registry_path: str | Path) -> None:
    """Add feature(s) from `yaml_path` to the registry at `registry_path`.

    Parameters
    ----------
    yaml_path : pathlib.Path
        Path to the junifer YAML.
    registry_path : str or pathlib.Path
        Path to the existing julio registry.

    Raises
    ------
    RuntimeError
        If there is a problem cloning a remote registry or
        if the dataset is not a julio registry.

    """
    log = logger.bind(
        cmd="create",
        path=registry_path if str else str(registry_path.resolve()),
    )
    if isinstance(registry_path, str):
        log.debug("Cloning remote registry")
        with tempfile.TemporaryDirectory() as tmpdir:
            log.debug(f"Temporary directory created at {tmpdir}")
            try:
                ds = dl.clone(
                    source=registry_path,
                    path=tmpdir,
                    on_failure="stop",
                    result_renderer="disabled",
                )
            except IncompleteResultsError as e:
                raise RuntimeError(
                    f"Failed to clone dataset: {e.failed}"
                ) from e
            else:
                log.debug("Remote registry cloned successfully")
            if not is_julio_registry(ds):
                raise RuntimeError(
                    f"Dataset at {ds.path} is not a julio registry"
                )
            process_features(yaml_path, ds)
            # TODO: push changes to remote
    else:
        ds = dl.Dataset(registry_path)
        if not is_julio_registry(ds):
            raise RuntimeError(f"Dataset at {ds.path} is not a julio registry")
        process_features(yaml_path, ds)
