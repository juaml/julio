"""Test for functions."""

# Authors: Synchon Mandal <s.mandal@fz-juelich.de>
# License: AGPL

import io
import shutil
import subprocess
from pathlib import Path

import datalad.api as dl
import numpy as np
import pandas as pd
import pytest
from h5io import write_hdf5
from junifer.api import generate_yaml
from junifer.storage import HDF5FeatureStorage
from junifer.utils import yaml

from julio import add, create
from julio._utils import _data_info, _generate_meta_yaml


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
