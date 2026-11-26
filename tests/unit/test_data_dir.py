"""Тесты P01-T01: каталог данных (SQUEEZELAB_DATA_DIR → paths.data_dir → ./data)."""

from __future__ import annotations

from pathlib import Path

import pytest

from squeezelab.app.settings import (
    DATA_DIR_ENV,
    AppConfig,
    PathsConfig,
    load_project_config,
    resolve_data_dir,
)

REPO_CONFIG = Path(__file__).resolve().parents[2] / "config"


def _app(data_dir: str | None) -> AppConfig:
    return AppConfig(paths=PathsConfig(data_dir=data_dir, logs_dir="logs"))


def test_env_variable_has_top_priority() -> None:
    result = resolve_data_dir(_app("from-yaml"), {DATA_DIR_ENV: "/srv/data/squeezelab"})
    assert result.path == Path("/srv/data/squeezelab")
    assert result.source == "env"


def test_app_yaml_is_second() -> None:
    result = resolve_data_dir(_app("from-yaml"), {})
    assert result.path == Path("from-yaml")
    assert result.source == "app.yaml"


def test_default_is_local_data_dir() -> None:
    result = resolve_data_dir(_app(None), {})
    assert result.path == Path("data")
    assert result.source == "default"


@pytest.mark.parametrize("blank", ["", "   "])
def test_blank_env_variable_counts_as_unset(blank: str) -> None:
    assert resolve_data_dir(_app("from-yaml"), {DATA_DIR_ENV: blank}).source == "app.yaml"
    assert resolve_data_dir(_app(None), {DATA_DIR_ENV: blank}).source == "default"


def test_tilde_is_expanded(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("USERPROFILE", str(tmp_path))  # Windows
    result = resolve_data_dir(_app("~/squeeze"), {})
    assert result.path == tmp_path / "squeeze"


def test_reads_process_environment_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(DATA_DIR_ENV, "/from/process/env")
    assert resolve_data_dir(_app("from-yaml")).path == Path("/from/process/env")
    monkeypatch.delenv(DATA_DIR_ENV)
    assert resolve_data_dir(_app("from-yaml")).path == Path("from-yaml")


def test_repo_config_resolves_to_app_yaml_value(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(DATA_DIR_ENV, raising=False)
    result = resolve_data_dir(load_project_config(REPO_CONFIG).app)
    assert (result.path, result.source) == (Path("data"), "app.yaml")


def test_resolution_creates_nothing(tmp_path: Path) -> None:
    target = tmp_path / "ещё-нет"
    resolve_data_dir(_app("x"), {DATA_DIR_ENV: str(target)})
    assert not target.exists()


def test_tests_do_not_see_the_project_environment() -> None:
    """Страховка от падений только на машинах, где SQUEEZELAB_DATA_DIR уже экспортирована."""
    import os

    assert DATA_DIR_ENV not in os.environ
    assert "SQUEEZELAB_CONFIG_DIR" not in os.environ
