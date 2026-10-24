from typer.testing import CliRunner

from aorticturb.cli import app


def test_version_command():
    result = CliRunner().invoke(app, ["version"])
    assert result.exit_code == 0
    assert "AorticTurb 0.2.0" in result.stdout


def test_support_sensitivity_command_is_registered():
    result = CliRunner().invoke(app, ["support-sensitivity", "--help"])
    assert result.exit_code == 0
    assert "--decomposition" in result.stdout
    assert "--regime" in result.stdout
