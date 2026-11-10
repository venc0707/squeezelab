"""Smoke-тест каркаса: пакет импортируется, структура на месте, CLI отвечает."""

from __future__ import annotations

import importlib
import re

import pytest

from squeezelab import __version__
from squeezelab.app.cli import main

EXPECTED_PACKAGES = [
    "squeezelab.data",
    "squeezelab.signals",
    "squeezelab.signals.squeeze",
    "squeezelab.signals.modules",
    "squeezelab.decision",
    "squeezelab.risk",
    "squeezelab.execution",
    "squeezelab.journal",
    "squeezelab.notify",
    "squeezelab.backtest",
    "squeezelab.research",
    "squeezelab.app",
]


def test_version_is_semver() -> None:
    assert re.fullmatch(r"\d+\.\d+\.\d+", __version__)


@pytest.mark.parametrize("name", EXPECTED_PACKAGES)
def test_package_importable(name: str) -> None:
    assert importlib.import_module(name) is not None


def test_cli_version(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exc:
        main(["--version"])
    assert exc.value.code == 0
    assert __version__ in capsys.readouterr().out
