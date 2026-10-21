"""Functions for julio."""

# Authors: Synchon Mandal <s.mandal@fz-juelich.de>
# License: AGPL

import shutil
import tempfile
from pathlib import Path

import datalad.api as dl
import structlog
from datalad.support.exceptions import CommandError, IncompleteResultsError

from ._utils import build_site, is_julio_registry, process_features


__all__ = ["add", "build", "create", "publish"]


logger = structlog.get_logger()


def create(registry_path: Path) -> None:
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
    conf_path = ds.pathobj / "registry-config.yaml"
    shutil.copy(
        src=Path(__file__).parent / "registry-config.yaml",
        dst=conf_path,
    )
    ds.save(
        conf_path,
        message="[julio] add registry configuration",
        on_failure="stop",
        result_renderer="disabled",
    )


def add(
    yaml_path: Path,
    registry_path: str | Path,
) -> None:
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
        cmd="add",
        path=registry_path
        if isinstance(registry_path, str)
        else str(registry_path.resolve()),
    )
    if isinstance(registry_path, str):
        log.debug("Cloning remote registry")
        with tempfile.TemporaryDirectory() as tmpdir:
            log.debug(f"Temporary directory created at {tmpdir}")
            # Clone the remote registry
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
            # Add features
            process_features(yaml_path, ds)
            # Push changes to remote registry
            try:
                ds.push(
                    to="origin",
                    on_failure="stop",
                    result_renderer="disabled",
                )
            except IncompleteResultsError as e:
                raise RuntimeError(
                    f"Failed to push dataset: {e.failed}"
                ) from e
            else:
                log.debug("Pushed changes to remote registry successfully")
    else:
        ds = dl.Dataset(registry_path)
        if not is_julio_registry(ds):
            raise RuntimeError(f"Dataset at {ds.path} is not a julio registry")
        process_features(yaml_path, ds)


def build(output: Path, registry_path: str | Path) -> None:
    """Build static site at `output` for registry at `registry_path`.

    Parameters
    ----------
    output : pathlib.Path
        Path to the output directory.
    registry_path : str or pathlib.Path
        Path to the existing julio registry.

    Raises
    ------
    RuntimeError
        If there is a problem building the static site or
        if the dataset is not a julio registry.

    """
    log = logger.bind(
        cmd="build",
        path=registry_path
        if isinstance(registry_path, str)
        else str(registry_path.resolve()),
    )
    if isinstance(registry_path, str):
        log.debug("Cloning remote registry")
        with tempfile.TemporaryDirectory() as tmpdir:
            log.debug(f"Temporary directory created at {tmpdir}")
            # Clone the remote registry
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
            build_site(output, ds)
    else:
        ds = dl.Dataset(registry_path)
        if not is_julio_registry(ds):
            raise RuntimeError(f"Dataset at {ds.path} is not a julio registry")
        build_site(output, ds)


def publish(
    registry_path: Path,
    remote: str,
    name: str | None = None,
) -> str:
    """Publish the registry at `registry_path` to `remote`.

    Parameters
    ----------
    registry_path : pathlib.Path
        Path to the existing julio registry.
    remote : str
        Name of a sibling of the registry or URL to publish to. If the URL is
        not of a sibling, a sibling is added: a RIA sibling with its storage
        sibling for ``ria+`` URLs, or a git sibling for the others.
    name : str or None, optional
        Name of the sibling to add, if ``remote`` is a new URL. If None,
        ``origin`` is used, if the registry does not have it (default None).

    Returns
    -------
    str
        Name of the sibling the registry was published to.

    Raises
    ------
    RuntimeError
        If the dataset is not a julio registry,
        if the sibling cannot be added or
        if there is a problem pushing to it.

    """
    log = logger.bind(cmd="publish", path=str(registry_path.resolve()))
    ds = dl.Dataset(registry_path)
    if not is_julio_registry(ds):
        raise RuntimeError(f"Dataset at {ds.path} is not a julio registry")
    info = {s["name"]: s for s in ds.siblings(result_renderer="disabled")}
    siblings = {k: v.get("url") for k, v in info.items()}
    # The siblings with the URL, including the RIA siblings, whose RIA URL
    # is the one of their storage sibling
    by_url = [
        k
        for k, v in info.items()
        if v.get("url") == remote
        or info.get(v.get("datalad-publish-depends"), {}).get("annex-url")
        == remote
    ]
    if remote in siblings:
        sibling = remote
    elif by_url:
        sibling = by_url[0]
    else:
        sibling = name if name is not None else "origin"
        if sibling in siblings:
            raise RuntimeError(
                f"Sibling `{sibling}` already exists with URL "
                f"{siblings[sibling]}: pass a name for the sibling of "
                f"{remote}"
            )
        log.debug(f"Adding sibling `{sibling}` for {remote}")
        try:
            if remote.startswith("ria+"):
                ds.create_sibling_ria(
                    url=remote,
                    name=sibling,
                    new_store_ok=True,
                    on_failure="stop",
                    result_renderer="disabled",
                )
            else:
                ds.siblings(
                    action="add",
                    name=sibling,
                    url=remote,
                    on_failure="stop",
                    result_renderer="disabled",
                )
        except IncompleteResultsError as e:
            raise RuntimeError(f"Failed to add sibling: {e.failed}") from e
    log.debug(f"Publishing registry to `{sibling}`")
    try:
        ds.push(
            to=sibling,
            data="anything",
            on_failure="stop",
            result_renderer="disabled",
        )
    except IncompleteResultsError as e:
        raise RuntimeError(f"Failed to push dataset: {e.failed}") from e
    except CommandError as e:
        raise RuntimeError(f"Failed to push dataset: {e}") from e
    log.debug("Published registry")
    return sibling
