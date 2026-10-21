"""Utilities for julio."""

# Authors: Synchon Mandal <s.mandal@fz-juelich.de>
# License: AGPL

import datetime as dt
import glob
import pathlib
import re
import shutil
from importlib.metadata import version
from pathlib import Path
from typing import TYPE_CHECKING, Any

import click
import datalad.api as dl
import structlog
from h5io import write_hdf5
from jinja2 import Environment, PackageLoader, select_autoescape
from junifer.api import generate_yaml, parse_yaml
from junifer.storage import HDF5FeatureStorage
from junifer.utils import yaml
from tqdm import tqdm


if TYPE_CHECKING:
    import numpy as np


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

# Remote URLs that are not HTTP(S), e.g., ssh://host/path, git@host:path or
# ria+ssh://host/path
remote_pattern = re.compile(
    r"^(?:[a-z][a-z0-9+.-]*://\S+|[\w.-]+@[\w.-]+:\S*)$"
)


class PathOrURLParamType(click.ParamType):  # pragma: no cover
    name = "path_or_url"

    def convert(self, value, param, ctx):
        # URL
        if value.startswith("http"):
            if pattern.match(value) is not None:
                return value
            else:
                self.fail(f"{value!r} is not a valid url.", param, ctx)
        if remote_pattern.match(value) is not None:
            return value
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
    if (ds.pathobj / "registry-config.yaml").is_file():
        return True
    return False


def _get_storage_path(yaml_path: Path) -> Path:
    """Get the path of the HDF5 storage of a junifer YAML.

    Parameters
    ----------
    yaml_path : Path
        Path to the junifer YAML.

    Returns
    -------
    Path
        Path to the HDF5 storage, with the features collected.

    Raises
    ------
    RuntimeError
        If the storage is not an existing ``HDF5FeatureStorage``.

    """
    log = logger.bind(cmd="parse_yaml", path=str(yaml_path.resolve()))
    log.debug("Parsing junifer YAML")
    # junifer resolves the storage URI relative to the YAML
    contents = parse_yaml(yaml_path)
    storage = contents.get("storage", {})
    if storage.get("kind") != "HDF5FeatureStorage":
        raise RuntimeError(
            f"Unsupported storage kind `{storage.get('kind')}` in "
            f"{yaml_path.resolve()!s}: only `HDF5FeatureStorage` is supported"
        )
    if "uri" not in storage:
        raise RuntimeError(
            f"`uri` missing from `storage` section in {yaml_path.resolve()!s}"
        )
    storage_path = Path(storage["uri"])
    if not storage_path.is_file():
        raise RuntimeError(
            f"Storage file does not exist: {storage_path!s}. If the "
            "features were stored with `single_output: false`, run "
            "`junifer collect` first."
        )
    log.debug("Parsed junifer YAML")
    return storage_path


def _data_info(data: "np.ndarray | list[np.ndarray]") -> dict[str, Any]:
    """Get the shape, size, data type and number of bytes of feature data.

    Parameters
    ----------
    data : np.ndarray or list of np.ndarray
        The feature data: an array for vectors and matrices or a list of
        arrays (one per element) for the others (e.g., timeseries).

    Returns
    -------
    dict
        The information of the data.

    """
    arrays = data if isinstance(data, list) else [data]
    if isinstance(data, list):
        # The shape of the elements, if they are all the same
        shapes = {a.shape for a in arrays}
        shape = [len(arrays), *next(iter(shapes))] if len(shapes) == 1 else []
    else:
        shape = list(data.shape)
    return {
        "shape": [int(x) for x in shape],
        "size": int(sum(a.size for a in arrays)),
        "dtype": str(arrays[0].dtype) if arrays else "",
        "nbytes": int(sum(a.nbytes for a in arrays)),
    }


def _generate_meta_yaml(
    feature_md5: str,
    meta: dict,
    data: dict,
    feature_yaml: dict,
) -> dict:
    """Generate a feature meta YAML from data.

    The meta YAML is a cache of the HDF5 file of the feature, to build the
    site without reading them, so it only has information from it: it can be
    generated again from the HDF5 file (e.g., it does not have the mappings
    of the registry or when the feature was added).

    Parameters
    ----------
    feature_md5 : str
        Feature MD5.
    meta : dict
        Feature metadata as dictionary.
    data : dict
        Feature data as dictionary.
    feature_yaml : dict
        The junifer YAML to compute the feature.

    Returns
    -------
    dict
        Feature meta YAML.

    """
    y: dict[str, Any] = {}
    y["md5"] = feature_md5
    y["name"] = meta["name"]
    y["data"] = _data_info(data["data"])
    y["samples"] = len(data["element"])
    # The parameters set in the YAML, as the defaults are not stored
    marker = feature_yaml["markers"][0]
    y["marker_meta"] = {
        k: v for k, v in marker.items() if k not in ("kind", "name")
    }
    y["storage_meta"] = {
        k: v for k, v in data.items() if k not in ("data", "element")
    }
    y["dependencies"] = dict(meta.get("dependencies", {}))
    # The classes, to get the human-readable names from the mappings of the
    # registry when building the site
    y["dataset"] = {"name": meta["datagrabber"]["class"]}
    y["marker"] = {"name": meta["marker"]["class"]}
    return y


def _write_feature_hdf5(
    path: Path, feature_md5: str, meta: dict, data: dict
) -> None:
    """Write a feature to its own HDF5 file, as junifer stores it.

    The file can be read with :class:`junifer.storage.HDF5FeatureStorage`.

    Parameters
    ----------
    path : Path
        Path to the HDF5 file.
    feature_md5 : str
        Feature MD5.
    meta : dict
        Feature metadata as dictionary.
    data : dict
        Feature data as dictionary.

    """
    write_hdf5(
        fname=str(path),
        data={feature_md5: meta},
        overwrite=True,
        title="meta",
        slash="error",
        use_json=True,
    )
    write_hdf5(
        fname=str(path),
        data=data,
        overwrite="update",
        title=feature_md5,
        slash="error",
        use_json=True,
    )


def _process_hdf5(
    storage_path: Path,
    ds: dl.Dataset,
) -> list[str]:
    """Add the features of an HDF5 storage to the dataset.

    Parameters
    ----------
    storage_path : Path
        Path to the HDF5 storage.
    ds : dl.Dataset
        Dataset to add features to.

    Returns
    -------
    list of str
        The MD5 of the features added.

    Raises
    ------
    RuntimeError
        If the YAML cannot be generated for a feature (e.g., it was computed
        with junifer <= 0.0.8).

    """
    log = logger.bind(cmd="process_hdf5", path=str(storage_path))
    log.debug("Processing HDF5 file")
    storage = HDF5FeatureStorage(uri=storage_path)
    features = storage.list_features()
    # Generate all the YAMLs first, to not add any feature if one fails
    feature_yamls = {}
    for feature_md5, meta in features.items():
        try:
            feature_yamls[feature_md5] = generate_yaml(meta)
        except ValueError as e:
            raise RuntimeError(
                f"Cannot add the features in {storage_path!s}: {e}"
            ) from e
    feature_dir = ds.pathobj / "features"
    feature_dir.mkdir(exist_ok=True)
    added = []
    for feature_md5, meta in tqdm(
        features.items(), desc="Processing features"
    ):
        meta_path = feature_dir / f"feature-{feature_md5}-meta.yaml"
        if meta_path.exists():
            log.info(f"Feature {feature_md5} already in registry, skipping")
            continue
        # YAML
        with (feature_dir / f"feature-{feature_md5}.yaml").open("w") as f:
            yaml.dump(feature_yamls[feature_md5], stream=f)
        # Data
        data = storage.read(feature_md5=feature_md5)
        _write_feature_hdf5(
            path=feature_dir / f"feature-{feature_md5}.hdf5",
            feature_md5=feature_md5,
            meta=meta,
            data=data,
        )
        # Metadata
        meta_data = _generate_meta_yaml(
            feature_md5=feature_md5,
            meta=meta,
            data=data,
            feature_yaml=feature_yamls[feature_md5],
        )
        with meta_path.open("w") as f:
            yaml.dump(meta_data, stream=f)
        added.append(feature_md5)
    log.debug("Processed HDF5 file")
    return added


def process_features(
    yaml_path: Path,
    ds: dl.Dataset,
) -> dl.Dataset:
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
    storage_path = _get_storage_path(yaml_path)
    added = _process_hdf5(
        storage_path=storage_path,
        ds=ds,
    )
    if added:
        ds.save(
            path=ds.pathobj / "features",
            message=f"[julio] add features for {yaml_path.name}",
            on_failure="stop",
            result_renderer="disabled",
        )
    log.debug("Processed features", added=len(added))
    return ds


def _find_mapping(mappings: list[dict], klass: str) -> dict | None:
    """Find the mapping of a pipeline component.

    Parameters
    ----------
    mappings : list of dict
        The mappings of a kind of pipeline component.
    klass : str
        The class of the pipeline component.

    Returns
    -------
    dict or None
        The mapping of the component, if any.

    """
    for mapping in mappings:
        if mapping["class"] == klass:
            return mapping
    return None


def _apply_mappings(meta: dict, mappings: dict) -> dict:
    """Add the human-readable names and tags to a feature meta YAML.

    The components that are not in the mappings keep their class name.

    Parameters
    ----------
    meta : dict
        Feature meta YAML.
    mappings : dict
        Registry mappings as dictionary.

    Returns
    -------
    dict
        Feature meta YAML with the names, descriptions and tags.

    """
    out = dict(meta)
    tags = []
    for key, kind in (("dataset", "datagrabbers"), ("marker", "markers")):
        info = dict(meta[key])
        mapping = _find_mapping(mappings.get(kind) or [], info["name"])
        if mapping is None:
            mapping = {"display_name": info["name"], "description": ""}
        info["display_name"] = mapping["display_name"]
        info["description"] = mapping["description"]
        tags.extend(mapping.get("tags") or [])
        out[key] = info
    out["tags"] = sorted(set(tags))
    return out


JINJA_ENV = Environment(
    loader=PackageLoader("julio", package_path=""),
    autoescape=select_autoescape(),
)


def _added_on(ds: dl.Dataset) -> dict[str, str]:
    """Get when the features were added to the registry, from git.

    Parameters
    ----------
    ds : dl.Dataset
        Dataset of the registry.

    Returns
    -------
    dict
        When each feature was added (MD5 -> date in UTC), for the features
        that are committed.

    """
    out = {}
    added = None
    # The newest commits first, with the files they added
    for line in ds.repo.call_git(
        [
            "log",
            "--diff-filter=A",
            "--format=@%at",
            "--name-only",
            "--",
            "features",
        ]
    ).splitlines():
        if line.startswith("@"):
            added = dt.datetime.fromtimestamp(int(line[1:]), tz=dt.UTC)
        elif line.endswith("-meta.yaml") and added is not None:
            md5 = (
                Path(line)
                .name.removeprefix("feature-")
                .removesuffix("-meta.yaml")
            )
            out.setdefault(md5, added.strftime("%Y-%m-%d %H:%M:%S"))
    return out


def build_site(output: Path, ds: dl.Dataset) -> None:
    """Build the static site.

    Parameters
    ----------
    output : Path
        Path to the output directory.
    ds : dl.Dataset
        Dataset of the site.

    """
    log = logger.bind(
        cmd="build_site",
        registry=str(ds.pathobj.resolve()),
        output=str(output.resolve()),
    )
    log.debug("Building static site")
    feature_dir = ds.pathobj / "features"
    # The current mappings of the registry
    with (ds.pathobj / "registry-config.yaml").open("r") as stream:
        mappings = yaml.load(stream)["mappings"]
    # Gather available meta files
    fs = []
    for f in sorted(glob.iglob(str((feature_dir / "*-meta.yaml").resolve()))):
        with Path(f).open("r") as stream:
            fs.append(_apply_mappings(yaml.load(stream), mappings))
    added_on = _added_on(ds)
    for f in fs:
        f["added_on"] = added_on.get(f["md5"], "")
    # Load templates
    tem_idx = JINJA_ENV.get_template("index.html")
    tem_feature = JINJA_ENV.get_template("feature.html")
    # The URL to clone the registry: the one of origin, if any
    clone_url = None
    for s in dl.siblings(dataset=ds, result_renderer="disabled"):
        if s["name"] == "origin":
            clone_url = s["url"]
            break
    if clone_url is None:
        log.warning(
            "Registry has no `origin` sibling, using its local path to clone "
            "it in the download instructions"
        )
        clone_url = str(ds.pathobj.resolve())
    julio_version = version("julio")
    # Write HTML files
    output.mkdir(parents=True, exist_ok=True)
    (output / "index.html").write_text(
        tem_idx.render(features=fs, version=julio_version),
    )
    for f in fs:
        md5 = f["md5"]
        # As written by generate_yaml, with the comments
        feature_yaml = (feature_dir / f"feature-{md5}.yaml").read_text()
        tem_render = tem_feature.render(
            feature=f,
            yaml=feature_yaml,
            version=julio_version,
            dataset=clone_url,
        )
        (output / f"{md5}.html").write_text(tem_render)
    # Copy JS
    shutil.copy(Path(__file__).parent / "index.js", output / "index.js")
    shutil.copy(Path(__file__).parent / "feature.js", output / "feature.js")
    # Copy CSS
    shutil.copy(Path(__file__).parent / "julio.css", output / "julio.css")
    # Copy assets
    shutil.copy(Path(__file__).parent / "logo.png", output / "logo.png")
    log.debug("Built static site")
