"""Utilities for julio."""

# Authors: Synchon Mandal <s.mandal@fz-juelich.de>
# License: AGPL

import pathlib
import re
from pathlib import Path
from typing import Any

import click
import datalad.api as dl
import structlog
from h5io import read_hdf5, write_hdf5
from tqdm import tqdm

from ._yaml import yaml


__all__ = ["PathOrURL", "is_julio_registry", "process_features"]

logger = structlog.get_logger()

# The following is taken from:
# https://validators.readthedocs.io/en/latest/_modules/validators/url.html#url
# and based on:
# https://gist.github.com/dperini/729294
ip_middle_octet = r"(?:\.(?:1?\d{1,2}|2[0-4]\d|25[0-5]))"
ip_last_octet = r"(?:\.(?:[1-9]\d?|1\d\d|2[0-4]\d|25[0-4]))"

regex = re.compile(
    "^"
    # protocol identifier
    "(?:(?:https?|ftp)://)"
    # user:pass authentication
    r"(?:\S+(?::\S*)?@)?"
    "(?:"
    "(?P<private_ip>"
    # IP address exclusion
    # private & local networks
    "(?:(?:10|127)" + ip_middle_octet + "{2}" + ip_last_octet + ")|"
    r"(?:(?:169\.254|192\.168)" + ip_middle_octet + ip_last_octet + ")|"
    r"(?:172\.(?:1[6-9]|2\d|3[0-1])" + ip_middle_octet + ip_last_octet + "))"
    "|"
    # IP address dotted notation octets
    # excludes loopback network 0.0.0.0
    # excludes reserved space >= 224.0.0.0
    # excludes network & broadcast addresses
    # (first & last IP address of each class)
    "(?P<public_ip>"
    r"(?:[1-9]\d?|1\d\d|2[01]\d|22[0-3])"
    "" + ip_middle_octet + "{2}"
    "" + ip_last_octet + ")"
    "|"
    # host name
    "(?:(?:[a-z\u00a1-\uffff0-9]-?)*[a-z\u00a1-\uffff0-9]+)"
    # domain name
    r"(?:\.(?:[a-z\u00a1-\uffff0-9]-?)*[a-z\u00a1-\uffff0-9]+)*"
    # TLD identifier
    r"(?:\.(?:[a-z\u00a1-\uffff]{2,}))"
    ")"
    # port number
    r"(?::\d{2,5})?"
    # resource path
    r"(?:/\S*)?"
    # query string
    r"(?:\?\S*)?"
    "$",
    re.UNICODE | re.IGNORECASE,
)

pattern = re.compile(regex)


class PathOrURLParamType(click.ParamType):
    name = "path_or_url"

    def convert(self, value, param, ctx):
        # URL
        if value.startswith("http"):
            if pattern.match(value) is not None:
                return value
            else:
                self.fail(f"{value!r} is not a valid url.", param, ctx)
        # Path
        try:
            p = click.Path(
                exists=True,
                readable=True,
                writable=True,
                file_okay=False,
                path_type=pathlib.Path,
            ).convert(value, param, ctx)
        except click.BadParameter as e:
            self.fail(f"{e}", param, ctx)
        else:
            return p


PathOrURL = PathOrURLParamType()


def is_julio_registry(ds: dl.Dataset) -> bool:
    """Check if the dataset is a julio registry.

    Parameters
    ----------
    ds : dl.Dataset
        Dataset to check.

    Returns
    -------
    bool
        True if the dataset is a julio registry, False otherwise.

    """
    if (ds.pathobj / "registry-config.yml").is_file():
        return True
    return False


def _make_absolute_path(path: str, base_path: Path) -> str:
    """Make a path absolute.

    Parameters
    ----------
    path : str
        Path to make absolute.
    base_path : Path
        Base path to use.

    Returns
    -------
    str
        Absolute path.

    """
    log = logger.bind(cmd="make_absolute_path", path=str(path))
    log.debug("Making path absolute")
    path = Path(path)
    if not path.is_absolute():
        path = base_path / path
    log.debug("Made path absolute")
    return str(path.resolve())


def _adjust_paths(data: dict, yaml_path: Path) -> dict:
    """Adjust paths in the data dictionary.

    Parameters
    ----------
    data : dict
        Data dictionary to adjust.
    yaml_path : Path
        Path to the YAML file.

    Returns
    -------
    dict
        Adjusted data dictionary.

    """
    log = logger.bind(cmd="adjust_paths", path=str(yaml_path.resolve()))
    log.debug("Adjusting paths")
    if "workdir" in data:
        if isinstance(data["workdir"], str):
            data["workdir"] = _make_absolute_path(
                data["workdir"], yaml_path.parent
            )
        else:
            data["workdir"]["path"] = _make_absolute_path(
                data["workdir"]["path"], yaml_path.parent
            )
    if "with" in data:
        if not isinstance(data["with"], list):
            data["with"] = list(data["with"])
        mods = []
        for w in data["with"]:
            if w.endswith(".py"):
                mods.append(_make_absolute_path(w, yaml_path.parent))
            else:
                mods.append(w)
        data["with"] = mods
    data["storage"]["uri"] = _make_absolute_path(
        data["storage"]["uri"], yaml_path.parent
    )
    log.debug("Adjusted paths")
    return data


def _parse_yaml(yaml_path: Path) -> dict:
    """Parse the junifer YAML.

    Parameters
    ----------
    yaml_path : Path
        Path to the junifer YAML.

    Returns
    -------
    dict
        Parsed YAML content.

    Raises
    ------
    RuntimeError
        If the YAML file is invalid or
        storage file does not exist.

    """
    log = logger.bind(cmd="parse_yaml", path=str(yaml_path.resolve()))
    log.debug("Parsing junifer YAML")
    contents = yaml.load(yaml_path)
    # Validate mandatory sections
    mandatory = ("workdir", "datagrabber", "markers", "storage")
    for s in mandatory:
        if s not in contents.keys():
            raise RuntimeError(
                f"`{s}` section not defined in {yaml_path.resolve()!s}"
            )
    # Validate optional sections
    optional = ("with", "preprocess", "queue")
    for k in contents.keys():
        if k not in mandatory + optional:
            raise RuntimeError(
                f"Unknown section `{k}` in {yaml_path.resolve()!s}"
            )
    # Validate storage
    if "uri" not in contents["storage"]:
        raise RuntimeError(
            f"`uri` missing from `storage` section in {yaml_path.resolve()!s}"
        )
    # Replace relative file paths with absolute
    contents = _adjust_paths(contents, yaml_path)
    # Validate storage file exists
    if not Path(contents["storage"]["uri"]).exists():
        raise RuntimeError(
            f"Storage file does not exist: {contents['storage']['uri']}"
        )
    # Remove elements key if empty
    if "elements" in contents:
        if contents["elements"] is None:
            _ = contents.pop("elements")
    log.debug("Parsed junifer YAML")
    return contents


def _feature_yaml_from_meta(data: dict, feature: str) -> dict:
    """Generate a feature YAML from metadata.

    Parameters
    ----------
    data : dict
        Dictionary containing the HDF5 metadata.
    feature : str
        MD5 hash of the feature.

    Returns
    -------
    dict
        Feature YAML.

    """
    y: dict[str, Any] = {}
    y["workdir"] = ""
    if "with" in data:
        y["with"] = data["with"].copy()
    # Set datagrabber
    y["datagrabber"] = data["datagrabber"].copy()
    a = y["datagrabber"].pop("class")
    y["datagrabber"]["kind"] = a
    if a not in ("PatternDataGrabber", "PatternDataladDataGrabber"):
        _ = y["datagrabber"].pop("uri")
        _ = y["datagrabber"].pop("rootdir")
        _ = y["datagrabber"].pop("patterns")
        _ = y["datagrabber"].pop("replacements")
        _ = y["datagrabber"].pop("confounds_format")
        _ = y["datagrabber"].pop("partial_pattern_ok")
        for k in data[
            "datagrabber"
        ].keys():  # use data instead of y to avoid .copy()
            if k.startswith("datalad"):
                _ = y["datagrabber"].pop(k)
    # Set preprocess
    if "preprocess" in data:
        y["preprocess"] = data["preprocess"].copy()
        b = y["preprocess"].pop("class")
        y["preprocess"]["kind"] = b
    # Set markers
    y["markers"] = []
    y["markers"].append(data["marker"].copy())
    c = y["markers"][0].pop("class")
    y["markers"][0]["kind"] = c
    if y["markers"][0]["masks"] is None:
        _ = y["markers"][0].pop("masks")
    # Set storage
    y["storage"] = {
        "kind": "HDF5FeatureStorage",
        "uri": "",
    }
    # Set queue
    y["queue"] = {
        "jobname": data["name"],
        "kind": "",
    }
    return y


def _process_hdf5(data: dict, ds: dl.Dataset) -> dl.Dataset:
    """Read and write HDF5 data.

    Parameters
    ----------
    data : dict
        Dictionary containing the HDF5 data.
    ds : dl.Dataset
        Dataset to add features to.

    Returns
    -------
    dl.Dataset
        Dataset with features added.

    """
    log = logger.bind(cmd="process_hdf5", path=data["storage"]["uri"])
    log.debug("Processing HDF5 file")
    metadata = read_hdf5(
        fname=data["storage"]["uri"],
        title="meta",
        slash="ignore",
    )
    feature_dir = ds.pathobj / "features"
    feature_dir.mkdir(exist_ok=True)
    for k, v in tqdm(metadata.items(), desc="Processing features"):
        # Metadata
        meta_path = feature_dir / f"feature-{k}-meta.yaml"
        yaml.dump(v, stream=meta_path.open("w"))
        # YAML
        yaml_data = _feature_yaml_from_meta(v, k)
        yaml_path = feature_dir / f"feature-{k}.yml"
        yaml.dump(yaml_data, stream=yaml_path.open("w"))
        # Data
        data = read_hdf5(fname=data["storage"]["uri"], title=k, slash="ignore")
        data_path = feature_dir / f"feature-{k}.h5"
        write_hdf5(
            fname=str(data_path.resolve()),
            data=data,
            overwrite=True,
            title=k,
            slash="error",
            use_json=False,
        )
    log.debug("Processed HDF5 file")
    return ds


def process_features(yaml_path: Path, ds: dl.Dataset) -> dl.Dataset:
    """Parse the junifer YAML and add features to the dataset.

    Parameters
    ----------
    yaml_path : Path
        Path to the junifer YAML.
    ds : dl.Dataset
        Dataset to add features to.

    Returns
    -------
    dl.Dataset
        Dataset with features added.

    Raises
    ------
    RuntimeError
        If the storage file does not exist or
        is not a valid format.

    """
    log = logger.bind(cmd="process_features", path=str(yaml_path.resolve()))
    log.debug("Processing features")
    data = _parse_yaml(yaml_path)
    if data["storage"]["uri"].endswith(".hdf5"):
        ds = _process_hdf5(data, ds)
    elif data["storage"]["uri"].endswith(".sqlite"):
        # TODO(synchon): add support for SQLite
        pass
    else:
        raise RuntimeError(
            "Unsupported storage format extension: "
            f"{Path(data['storage']['uri']).suffix}"
        )
    ds.save(
        message=f"[julio] add features for {yaml_path.name}",
        on_failure="stop",
        result_renderer="disabled",
    )
    log.debug("Processed features")
    return ds
