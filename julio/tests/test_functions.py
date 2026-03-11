"""Test for functions."""

# Authors: Synchon Mandal <s.mandal@fz-juelich.de>
# License: AGPL

import shutil
from contextlib import AbstractContextManager, nullcontext
from pathlib import Path

import datalad.api as dl
import pytest

from julio import add, create


def test_create(tmp_path: Path) -> None:
    """Test registry creation.

    Parameters
    ----------
    tmp_path : Path
        Pytest fixture that provides a temporary directory.

    """
    registry_path = tmp_path / "test_registry"
    create(registry_path)
    config_path = registry_path / "registry-config.yml"
    assert config_path.is_file()
    with pytest.raises(RuntimeError):
        create(registry_path)
    shutil.rmtree(registry_path)


@pytest.mark.parametrize(
    "yml, expect",
    [
        ("valid.yml", nullcontext()),
        ("valid_extra.yml", nullcontext()),
        ("invalid_store.yml", pytest.raises(RuntimeError)),
        ("invalid_mandatory_section.yml", pytest.raises(RuntimeError)),
        ("invalid_section.yml", pytest.raises(RuntimeError)),
        ("missing_storage_uri.yml", pytest.raises(RuntimeError)),
    ],
)
def test_add(tmp_path: Path, yml: str, expect: AbstractContextManager) -> None:
    """Test feature addition.

    Parameters
    ----------
    tmp_path : Path
        Pytest fixture that provides a temporary directory.
    yml : str
        The parametrized YAML file name.
    expect : typing.ContextManager
        The parametrized ContextManager object.

    """
    registry_path = tmp_path / "test_registry"
    create(registry_path)
    with expect:
        add(
            yaml_path=Path(__file__).parent / yml,
            registry_path=registry_path,
            dataset_display_name=None,
        )
        f_dir = registry_path / "features"
        assert f_dir.is_dir()
        dl.drop(".", reckless="kill", dataset=dl.Dataset(registry_path))
        shutil.rmtree(registry_path)
    # Check for invalid dataset
    with pytest.raises(RuntimeError):
        add(
            yaml_path=Path(__file__).parent / yml,
            registry_path=tmp_path,
            dataset_display_name=None,
        )
