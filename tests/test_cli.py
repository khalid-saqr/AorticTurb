from typer.testing import CliRunner

from aorticturb.cli import app


def test_version_command():
    result = CliRunner().invoke(app, ["version"])
    assert result.exit_code == 0
    assert "AorticTurb 0.2.0" in result.stdout


def test_support_sensitivity_command_is_registered():
    result = CliRunner().invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "support-sensitivity" in result.stdout


def test_backscatter_command_is_registered():
    result = CliRunner().invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "backscatter" in result.stdout
