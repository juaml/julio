"""Utilities for julio."""

# Authors: Synchon Mandal <s.mandal@fz-juelich.de>
# License: AGPL

import pathlib
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import click
import datalad.api as dl
import structlog
from h5io import read_hdf5, write_hdf5
from junifer.api import generate_yaml, parse_yaml
from junifer.utils import yaml
from tqdm import tqdm


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


class PathOrURLParamType(click.ParamType):  # pragma: no cover
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
    contents = parse_yaml(yaml_path)
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
    # Validate storage file exists
    if not Path(contents["storage"]["uri"]).exists():
        raise RuntimeError(
            f"Storage file does not exist: {contents['storage']['uri']}"
        )
    log.debug("Parsed junifer YAML")
    return contents


def _generate_meta_yaml(
    meta: dict,
    data: dict,
    md5: str,
    mappings: dict,
    dataset_display_name: str | None,
) -> dict:
    """Generate a feature meta YAML from data.

    Parameters
    ----------
    meta : dict
        Feature metadata as dictionary.
    data : dict
        Feature data as dictionary.
    md5 : str
        Feature MD5.
    mappings : dict
        Registry mappings as dictionary.
    dataset_display_name : str or None
        Dataset display name.

    Returns
    -------
    dict
        Feature meta YAML.

    """
    y: dict[str, Any] = {}
    y["md5"] = md5
    y["name"] = meta["name"]
    y["added_on"] = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
    ndata = data["data"]
    y["data"] = {
        "shape": ndata.shape,
        "size": ndata.size,
        "dtype": str(ndata.dtype),
        "nbytes": sys.getsizeof(ndata),
    }
    y["samples"] = len(data["element"])
    mm = {}
    for k, v in meta["marker"].items():
        if k not in ("class", "on", "name"):
            mm[k] = v
    y["marker_meta"] = mm
    sm = {}
    for k, v in data.items():
        if k not in ("data", "element"):
            sm[k] = v
    y["storage_meta"] = sm
    tags = []
    for i in mappings["datagrabbers"]:
        if meta["datagrabber"]["class"] == i["class"]:
            tags.extend(i["tags"])
            y["dataset"] = {
                "name": i["class"],
                "description": i["description"],
                "display_name": i["display_name"]
                if dataset_display_name is None
                else dataset_display_name,
            }
    for i in mappings["markers"]:
        if meta["marker"]["class"] == i["class"]:
            tags.extend(i["tags"])
    y["tags"] = list(set(tags))
    return y


def _process_hdf5(
    data: dict,
    ds: dl.Dataset,
    dataset_display_name: str | None,
) -> dl.Dataset:
    """Read and write HDF5 data.

    Parameters
    ----------
    data : dict
        Parsed YAML as dictionary.
    ds : dl.Dataset
        Dataset to add features to.
    dataset_display_name : str or None
        Dataset display name.

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
        # Add "with" and "queue" section if present in data
        if "with" in data:
            v["with"] = data["with"].copy()
        if "queue" in data:
            v["queue"] = data["queue"].copy()
        # YAML
        yaml_data = generate_yaml(v)
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
        # Metadata
        config_path = ds.pathobj / "registry-config.yml"
        meta_data = _generate_meta_yaml(
            meta=v,
            data=data,
            md5=k,
            mappings=yaml.load(stream=config_path.open("r"))["mappings"],
            dataset_display_name=dataset_display_name,
        )
        meta_path = feature_dir / f"feature-{k}-meta.yml"
        yaml.dump(meta_data, stream=meta_path.open("w"))
    log.debug("Processed HDF5 file")
    return ds


def process_features(
    yaml_path: Path,
    ds: dl.Dataset,
    dataset_display_name: str | None,
) -> dl.Dataset:
    """Parse the junifer YAML and add features to the dataset.

    Parameters
    ----------
    yaml_path : Path
        Path to the junifer YAML.
    ds : dl.Dataset
        Dataset to add features to.
    dataset_display_name : str or None
        Dataset display name.

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
        ds = _process_hdf5(
            data=data,
            ds=ds,
            dataset_display_name=dataset_display_name,
        )
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
