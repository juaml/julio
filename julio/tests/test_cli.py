"""Test for CLI."""

# Authors: Synchon Mandal <s.mandal@fz-juelich.de>
# License: AGPL

from pathlib import Path

from click.testing import CliRunner

from julio._cli import cli


TESTS_DIR = Path(__file__).parent


def test_cli(tmp_path: Path) -> None:
    """Test the commands and their exit codes.

    Parameters
    ----------
    tmp_path : Path
        Pytest fixture that provides a temporary directory.

    """
    runner = CliRunner()
    registry = str(tmp_path / "registry")
    result = runner.invoke(cli, ["create", registry])
    assert result.exit_code == 0, result.output
    # Already exists
    result = runner.invoke(cli, ["create", registry])
    assert result.exit_code == 1
    result = runner.invoke(
        cli, ["add", str(TESTS_DIR / "valid.yaml"), "-r", registry]
    )
    assert result.exit_code == 0, result.output
    result = runner.invoke(
        cli, ["add", str(TESTS_DIR / "valid.yaml"), "-r", str(tmp_path)]
    )
    assert result.exit_code == 1
    assert "not a julio registry" in result.output
    result = runner.invoke(
        cli, ["build", "-r", registry, "-o", str(tmp_path / "build")]
    )
    assert result.exit_code == 0, result.output
    assert (tmp_path / "build" / "index.html").is_file()
    result = runner.invoke(
        cli, ["build", "-r", str(tmp_path), "-o", str(tmp_path / "build")]
    )
    assert result.exit_code == 1
    result = runner.invoke(
        cli, ["publish", "-r", registry, "--remote", "origin"]
    )
    assert result.exit_code == 1
    assert "Failed to push" in result.output
