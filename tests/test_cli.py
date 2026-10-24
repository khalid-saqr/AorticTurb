from typer.testing import CliRunner

from aorticturb.cli import app


def test_version_command():
    result = CliRunner().invoke(app, ["version"])
    assert result.exit_code == 0
    assert "AorticTurb 0.2.0" in result.stdout
