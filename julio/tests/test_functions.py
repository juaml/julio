"""Test for functions."""

# Authors: Synchon Mandal <s.mandal@fz-juelich.de>
# License: AGPL

import html
import io
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import datalad.api as dl
import numpy as np
import pandas as pd
import pytest
from h5io import write_hdf5
from junifer.api import generate_yaml
from junifer.storage import HDF5FeatureStorage
from junifer.utils import yaml

from julio import add, build, create, publish
from julio._utils import _apply_mappings, _data_info, _generate_meta_yaml


TESTS_DIR = Path(__file__).parent


def test_create(tmp_path: Path) -> None:
    """Test registry creation.

    Parameters
    ----------
    tmp_path : Path
        Pytest fixture that provides a temporary directory.

    """
    registry_path = tmp_path / "test_registry"
    create(registry_path)
    config_path = registry_path / "registry-config.yaml"
    assert config_path.is_file()
    with pytest.raises(RuntimeError):
        create(registry_path)
    shutil.rmtree(registry_path)


def _n_commits(path: Path) -> int:
    """Count the commits of a git repository.

    Parameters
    ----------
    path : Path
        Path to the repository.

    Returns
    -------
    int
        The number of commits.

    """
    out = subprocess.run(
        ["git", "rev-list", "--count", "HEAD"],
        cwd=path,
        capture_output=True,
        check=True,
        text=True,
    )
    return int(out.stdout)


def test_add(tmp_path: Path) -> None:
    """Test feature addition.

    Parameters
    ----------
    tmp_path : Path
        Pytest fixture that provides a temporary directory.

    """
    registry_path = tmp_path / "test_registry"
    create(registry_path)
    n_commits = _n_commits(registry_path)
    add(
        yaml_path=TESTS_DIR / "valid.yaml",
        registry_path=registry_path,
    )
    assert _n_commits(registry_path) == n_commits + 1
    assert not dl.Dataset(registry_path).repo.dirty

    source = HDF5FeatureStorage(uri=TESTS_DIR / "store.hdf5")
    source_features = source.list_features()
    f_dir = registry_path / "features"
    assert len(list(f_dir.glob("*-meta.yaml"))) == len(source_features) == 2
    for md5, source_meta in source_features.items():
        # The data, readable with junifer as in the original storage
        feature = HDF5FeatureStorage(uri=f_dir / f"feature-{md5}.hdf5")
        assert list(feature.list_features()) == [md5]
        pd.testing.assert_frame_equal(
            feature.read_df(feature_md5=md5),
            source.read_df(feature_md5=md5),
        )
        # The YAML, to compute the feature again
        with (f_dir / f"feature-{md5}.yaml").open("r") as f:
            feature_yaml = yaml.load(f)
        expected = generate_yaml(source_meta)
        assert feature_yaml["markers"] == expected["markers"]
        assert feature_yaml["datagrabber"] == expected["datagrabber"]
        assert feature_yaml["with"] == ["junifer.testing.registry"]
        # The metadata
        with (f_dir / f"feature-{md5}-meta.yaml").open("r") as f:
            meta = yaml.load(f)
        assert meta["md5"] == md5
        assert meta["name"] == source_meta["name"]
        assert meta["samples"] == 2
        assert meta["data"]["nbytes"] > 0
        assert not any(k.startswith("_") for k in meta["marker_meta"])
        assert meta["marker_meta"]["parcellation"] == ["Schaefer100x7"]
        # The classes, the mappings are used when building the site
        assert meta["dataset"] == {"name": "PartlyCloudyTestingDataGrabber"}
        assert meta["marker"]["name"] == source_meta["marker"]["class"]
        assert "tags" not in meta
        assert meta["dependencies"]["junifer"] is not None
    # Adding the same features again does nothing
    add(
        yaml_path=TESTS_DIR / "valid.yaml",
        registry_path=registry_path,
    )
    assert _n_commits(registry_path) == n_commits + 1


def test_add_meta_from_hdf5(tmp_path: Path) -> None:
    """Test that the meta YAML can be generated again from the HDF5 file.

    Parameters
    ----------
    tmp_path : Path
        Pytest fixture that provides a temporary directory.

    """
    registry_path = tmp_path / "test_registry"
    create(registry_path)
    add(yaml_path=TESTS_DIR / "valid.yaml", registry_path=registry_path)
    f_dir = registry_path / "features"
    for meta_path in f_dir.glob("*-meta.yaml"):
        with meta_path.open("r") as f:
            stored = yaml.load(f)
        md5 = stored["md5"]
        # Only from the HDF5 file of the feature in the registry
        storage = HDF5FeatureStorage(uri=f_dir / f"feature-{md5}.hdf5")
        meta = storage.list_features()[md5]
        generated = _generate_meta_yaml(
            feature_md5=md5,
            meta=meta,
            data=storage.read(feature_md5=md5),
            feature_yaml=generate_yaml(meta),
        )
        # As stored in the YAML
        stream = io.StringIO()
        yaml.dump(generated, stream=stream)
        stream.seek(0)
        assert yaml.load(stream) == stored
        if stored["marker"]["name"] == "FunctionalConnectivityParcels":
            # The matrix has the shape of the array, with the elements last
            assert stored["data"]["shape"] == [100, 100, 2]
            assert stored["data"]["dtype"] == "float32"


@pytest.mark.parametrize(
    "yml",
    [
        "invalid_store.yaml",
        "missing_storage_uri.yaml",
        "missing_storage_file.yaml",
    ],
)
def test_add_invalid_yaml(tmp_path: Path, yml: str) -> None:
    """Test feature addition with invalid YAMLs.

    Parameters
    ----------
    tmp_path : Path
        Pytest fixture that provides a temporary directory.
    yml : str
        The parametrized YAML file name.

    """
    registry_path = tmp_path / "test_registry"
    create(registry_path)
    with pytest.raises(RuntimeError):
        add(
            yaml_path=TESTS_DIR / yml,
            registry_path=registry_path,
        )
    assert not (registry_path / "features").exists()


def test_add_not_registry(tmp_path: Path) -> None:
    """Test feature addition to a dataset that is not a registry.

    Parameters
    ----------
    tmp_path : Path
        Pytest fixture that provides a temporary directory.

    """
    with pytest.raises(RuntimeError, match="not a julio registry"):
        add(
            yaml_path=TESTS_DIR / "valid.yaml",
            registry_path=tmp_path,
        )


def test_add_old_junifer(tmp_path: Path) -> None:
    """Test feature addition of features computed with junifer <= 0.0.8.

    Parameters
    ----------
    tmp_path : Path
        Pytest fixture that provides a temporary directory.

    """
    # The storage without the parameters set to create the components, as
    # stored by junifer <= 0.0.8
    storage_path = tmp_path / "store.hdf5"
    shutil.copy(TESTS_DIR / "store.hdf5", storage_path)
    metas = HDF5FeatureStorage(uri=storage_path).list_features()
    for meta in metas.values():
        for step in ("datagrabber", "marker"):
            meta[step].pop("_fields_set")
    write_hdf5(
        fname=str(storage_path),
        data=metas,
        overwrite="update",
        title="meta",
        slash="error",
        use_json=True,
    )
    yaml_path = tmp_path / "old.yaml"
    shutil.copy(TESTS_DIR / "valid.yaml", yaml_path)
    registry_path = tmp_path / "test_registry"
    create(registry_path)
    with pytest.raises(RuntimeError, match=r"junifer > 0\.0\.8"):
        add(
            yaml_path=yaml_path,
            registry_path=registry_path,
        )
    assert not (registry_path / "features").exists()


def test_add_remote(tmp_path: Path) -> None:
    """Test feature addition to a remote registry.

    Parameters
    ----------
    tmp_path : Path
        Pytest fixture that provides a temporary directory.

    """
    registry_path = tmp_path / "test_registry"
    create(registry_path)
    # Allow pushing to the checked out branch of the "remote"
    subprocess.run(
        ["git", "config", "receive.denyCurrentBranch", "updateInstead"],
        cwd=registry_path,
        check=True,
    )
    n_commits = _n_commits(registry_path)
    add(
        yaml_path=TESTS_DIR / "valid.yaml",
        registry_path=str(registry_path),
    )
    assert _n_commits(registry_path) == n_commits + 1
    assert len(list((registry_path / "features").glob("*-meta.yaml"))) == 2


def test_add_remote_invalid(tmp_path: Path) -> None:
    """Test feature addition to an invalid remote registry.

    Parameters
    ----------
    tmp_path : Path
        Pytest fixture that provides a temporary directory.

    """
    with pytest.raises(RuntimeError, match="Failed to clone"):
        add(
            yaml_path=TESTS_DIR / "valid.yaml",
            registry_path=str(tmp_path / "missing"),
        )
    dl.create(path=tmp_path / "dataset", result_renderer="disabled")
    with pytest.raises(RuntimeError, match="not a julio registry"):
        add(
            yaml_path=TESTS_DIR / "valid.yaml",
            registry_path=str(tmp_path / "dataset"),
        )


def test_data_info() -> None:
    """Test the information of the data of the features."""
    info = _data_info(np.zeros((3, 4, 2), dtype="float32"))
    assert info == {
        "shape": [3, 4, 2],
        "size": 24,
        "dtype": "float32",
        "nbytes": 96,
    }
    # One array per element, e.g., timeseries
    info = _data_info([np.zeros((5, 3)), np.zeros((5, 3))])
    assert info["shape"] == [2, 5, 3]
    assert info["size"] == 30
    assert info["nbytes"] == 240
    # With different number of timepoints
    info = _data_info([np.zeros((5, 3)), np.zeros((4, 3))])
    assert info["shape"] == []
    assert info["size"] == 27


def _registry_with_features(tmp_path: Path) -> Path:
    """Create a registry with the features of the test storage.

    Parameters
    ----------
    tmp_path : Path
        Pytest fixture that provides a temporary directory.

    Returns
    -------
    Path
        Path to the registry.

    """
    registry_path = tmp_path / "test_registry"
    create(registry_path)
    add(
        yaml_path=TESTS_DIR / "valid.yaml",
        registry_path=registry_path,
    )
    return registry_path


def test_build(tmp_path: Path) -> None:
    """Test static site building.

    Parameters
    ----------
    tmp_path : Path
        Pytest fixture that provides a temporary directory.

    """
    registry_path = _registry_with_features(tmp_path)
    output = tmp_path / "site" / "build"
    build(output=output, registry_path=registry_path)
    for asset in ("index.js", "feature.js", "julio.css", "logo.png"):
        assert (output / asset).is_file()
    md5s = HDF5FeatureStorage(uri=TESTS_DIR / "store.hdf5").list_features()
    index = (output / "index.html").read_text()
    for md5 in md5s:
        assert f'href="./{md5}.html"' in index
        page = (output / f"{md5}.html").read_text()
        # The YAML of the feature, as written by generate_yaml
        feature_yaml = (
            registry_path / "features" / f"feature-{md5}.yaml"
        ).read_text()
        yaml_block = re.search(r'id="yaml">(.*?)</code>', page, re.DOTALL)
        assert yaml_block is not None
        assert html.unescape(yaml_block.group(1)) == feature_yaml
        # Without origin, the local path is used to clone
        assert f"datalad clone {registry_path.resolve()!s} registry" in page
        assert f"datalad get features/feature-{md5}.hdf5" in page
    # When the features were added, from git
    added_on = subprocess.run(
        [
            "git",
            "log",
            "-1",
            "--format=%ad",
            "--date=format-local:%Y-%m-%d %H:%M:%S",
        ],
        cwd=registry_path,
        capture_output=True,
        check=True,
        text=True,
        env={**os.environ, "TZ": "UTC"},
    ).stdout.strip()
    assert index.count(f"<td>{added_on}</td>") == len(md5s)
    # The registry is not changed
    assert not dl.Dataset(registry_path).repo.dirty


def test_build_remote(tmp_path: Path) -> None:
    """Test static site building for a remote registry.

    The download instructions of the site are followed to read the features.

    Parameters
    ----------
    tmp_path : Path
        Pytest fixture that provides a temporary directory.

    """
    registry_path = _registry_with_features(tmp_path)
    output = tmp_path / "build"
    build(output=output, registry_path=str(registry_path))
    source = HDF5FeatureStorage(uri=TESTS_DIR / "store.hdf5")
    for md5 in source.list_features():
        page = html.unescape((output / f"{md5}.html").read_text())
        commands = re.search(
            r"language-bash[^>]*>(.*?)</code>", page, re.DOTALL
        )
        assert commands is not None
        user_dir = tmp_path / f"user-{md5}"
        user_dir.mkdir()
        cwd = user_dir
        for command in commands.group(1).splitlines():
            args = command.removeprefix("$ ").split()
            if args[0] == "cd":
                cwd = cwd / args[1]
            else:
                # datalad of the environment, which might not be in PATH
                args[0] = str(Path(sys.executable).parent / args[0])
                subprocess.run(args, cwd=cwd, check=True, capture_output=True)
        code = re.search(r"language-python[^>]*>(.*?)</code>", page, re.DOTALL)
        assert code is not None
        assert f'feature_md5="{md5}"' in code.group(1)
        storage = HDF5FeatureStorage(
            uri=cwd / "features" / f"feature-{md5}.hdf5"
        )
        pd.testing.assert_frame_equal(
            storage.read_df(feature_md5=md5),
            source.read_df(feature_md5=md5),
        )


def test_build_not_registry(tmp_path: Path) -> None:
    """Test static site building for a dataset that is not a registry.

    Parameters
    ----------
    tmp_path : Path
        Pytest fixture that provides a temporary directory.

    """
    with pytest.raises(RuntimeError, match="not a julio registry"):
        build(output=tmp_path / "build", registry_path=tmp_path)
    dl.create(path=tmp_path / "dataset", result_renderer="disabled")
    with pytest.raises(RuntimeError, match="not a julio registry"):
        build(
            output=tmp_path / "build",
            registry_path=str(tmp_path / "dataset"),
        )
    with pytest.raises(RuntimeError, match="Failed to clone"):
        build(
            output=tmp_path / "build",
            registry_path=str(tmp_path / "missing"),
        )


def test_build_mappings(tmp_path: Path) -> None:
    """Test that the site uses the current mappings of the registry.

    Parameters
    ----------
    tmp_path : Path
        Pytest fixture that provides a temporary directory.

    """
    registry_path = _registry_with_features(tmp_path)
    output = tmp_path / "build"
    build(output=output, registry_path=registry_path)
    index = (output / "index.html").read_text()
    # Not in the mappings: the class
    assert "<td>PartlyCloudyTestingDataGrabber</td>" in index
    # In the mappings
    assert "<td>Functional Connectivity</td>" in index
    # The mappings are changed after adding the features
    config_path = registry_path / "registry-config.yaml"
    with config_path.open("r") as f:
        config = yaml.load(f)
    config["mappings"]["datagrabbers"].append(
        {
            "class": "PartlyCloudyTestingDataGrabber",
            "display_name": "Partly Cloudy",
            "tags": ["Partly Cloudy"],
            "description": "Partly Cloudy movie watching fMRI",
        }
    )
    with config_path.open("w") as f:
        yaml.dump(config, stream=f)
    build(output=output, registry_path=registry_path)
    index = (output / "index.html").read_text()
    assert "<td>Partly Cloudy</td>" in index
    assert ">Partly Cloudy</span>" in index
    md5 = next(
        iter(HDF5FeatureStorage(uri=TESTS_DIR / "store.hdf5").list_features())
    )
    page = (output / f"{md5}.html").read_text()
    assert "Partly Cloudy movie watching fMRI" in page


def test_apply_mappings() -> None:
    """Test the mappings applied to the metadata of the features."""
    mappings = {
        "datagrabbers": [
            {
                "class": "DG",
                "display_name": "Dataset",
                "description": "A dataset",
                "tags": ["A", "B"],
            }
        ],
        "markers": [
            {
                "class": "M",
                "display_name": "Marker",
                "description": "A marker",
                "tags": ["B", "C"],
            }
        ],
    }
    meta = {"md5": "x", "dataset": {"name": "DG"}, "marker": {"name": "M"}}
    out = _apply_mappings(meta, mappings)
    assert out["dataset"] == {
        "name": "DG",
        "display_name": "Dataset",
        "description": "A dataset",
    }
    assert out["marker"] == {
        "name": "M",
        "display_name": "Marker",
        "description": "A marker",
    }
    assert out["tags"] == ["A", "B", "C"]
    # The metadata is not modified
    assert meta == {
        "md5": "x",
        "dataset": {"name": "DG"},
        "marker": {"name": "M"},
    }
    # Not in the mappings, or without mappings
    for t_mappings in (mappings, {"datagrabbers": None}):
        out = _apply_mappings(
            {"dataset": {"name": "Other"}, "marker": {"name": "N"}},
            t_mappings,
        )
        assert out["dataset"] == {
            "name": "Other",
            "display_name": "Other",
            "description": "",
        }
        assert out["marker"]["display_name"] == "N"
        assert out["tags"] == []


def _read_published_feature(clone_url: str, path: Path, md5: str) -> None:
    """Clone a published registry and check the data of a feature.

    Parameters
    ----------
    clone_url : str
        URL to clone the registry from.
    path : Path
        Path to clone the registry to.
    md5 : str
        MD5 of the feature to check.

    """
    ds = dl.clone(source=clone_url, path=path, result_renderer="disabled")
    feature_path = f"features/feature-{md5}.hdf5"
    ds.get(feature_path, result_renderer="disabled")
    source = HDF5FeatureStorage(uri=TESTS_DIR / "store.hdf5")
    pd.testing.assert_frame_equal(
        HDF5FeatureStorage(uri=path / feature_path).read_df(feature_md5=md5),
        source.read_df(feature_md5=md5),
    )


def test_publish_ria(tmp_path: Path) -> None:
    """Test registry publishing to a RIA store.

    Parameters
    ----------
    tmp_path : Path
        Pytest fixture that provides a temporary directory.

    """
    registry_path = _registry_with_features(tmp_path)
    store_url = f"ria+{(tmp_path / 'store').as_uri()}"
    assert publish(registry_path, store_url) == "origin"
    md5 = next(
        iter(HDF5FeatureStorage(uri=TESTS_DIR / "store.hdf5").list_features())
    )
    ds_id = dl.Dataset(registry_path).id
    _read_published_feature(f"{store_url}#{ds_id}", tmp_path / "user", md5)
    # The site uses the published registry to clone it
    build(output=tmp_path / "build", registry_path=registry_path)
    page = (tmp_path / "build" / f"{md5}.html").read_text()
    assert str(registry_path.resolve()) not in page
    # Publishing again to the same URL uses the sibling
    assert publish(registry_path, store_url) == "origin"


def test_publish_sibling(tmp_path: Path) -> None:
    """Test registry publishing to an existing sibling.

    Parameters
    ----------
    tmp_path : Path
        Pytest fixture that provides a temporary directory.

    """
    registry_path = tmp_path / "test_registry"
    create(registry_path)
    # Allow pushing to the checked out branch of the "remote"
    subprocess.run(
        ["git", "config", "receive.denyCurrentBranch", "updateInstead"],
        cwd=registry_path,
        check=True,
    )
    local_path = tmp_path / "local"
    dl.clone(source=registry_path, path=local_path, result_renderer="disabled")
    add(
        yaml_path=TESTS_DIR / "valid.yaml",
        registry_path=local_path,
    )
    assert publish(local_path, "origin") == "origin"
    md5 = next(
        iter(HDF5FeatureStorage(uri=TESTS_DIR / "store.hdf5").list_features())
    )
    # The data is in the remote, not only the metadata
    assert (registry_path / "features" / f"feature-{md5}-meta.yaml").is_file()
    _read_published_feature(str(registry_path), tmp_path / "user", md5)


def test_publish_name(tmp_path: Path) -> None:
    """Test registry publishing to a new URL with a sibling name.

    Parameters
    ----------
    tmp_path : Path
        Pytest fixture that provides a temporary directory.

    """
    registry_path = _registry_with_features(tmp_path)
    first_url = f"ria+{(tmp_path / 'first').as_uri()}"
    second_url = f"ria+{(tmp_path / 'second').as_uri()}"
    assert publish(registry_path, first_url) == "origin"
    with pytest.raises(RuntimeError, match="already exists"):
        publish(registry_path, second_url)
    assert publish(registry_path, second_url, name="backup") == "backup"
    md5 = next(
        iter(HDF5FeatureStorage(uri=TESTS_DIR / "store.hdf5").list_features())
    )
    ds_id = dl.Dataset(registry_path).id
    _read_published_feature(f"{second_url}#{ds_id}", tmp_path / "user", md5)


def test_publish_invalid(tmp_path: Path) -> None:
    """Test registry publishing with invalid registries and remotes.

    Parameters
    ----------
    tmp_path : Path
        Pytest fixture that provides a temporary directory.

    """
    with pytest.raises(RuntimeError, match="not a julio registry"):
        publish(tmp_path, "origin")
    registry_path = tmp_path / "test_registry"
    create(registry_path)
    # Not a valid URL for a RIA store
    with pytest.raises(RuntimeError, match="Failed to add sibling"):
        publish(registry_path, "ria+invalid://store")
    # Nothing to push to
    with pytest.raises(RuntimeError, match="Failed to push"):
        publish(registry_path, str(tmp_path / "missing"))
