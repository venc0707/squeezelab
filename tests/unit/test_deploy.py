"""Тесты P01-T03 (поправка B): юниты и таймеры deploy/, установщик install.sh."""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
DEPLOY = REPO / "deploy"
UNITS = [
    "squeezelab-sync.service",
    "squeezelab-sync.timer",
    "squeezelab-backup.service",
    "squeezelab-backup.timer",
]


def directives(name: str) -> list[str]:
    """Строки юнита без комментариев и пустых."""
    lines = (DEPLOY / name).read_text(encoding="utf-8").splitlines()
    return [x.strip() for x in lines if x.strip() and not x.lstrip().startswith("#")]


def value(name: str, key: str) -> str:
    for line in directives(name):
        if line.startswith(f"{key}="):
            return line.split("=", 1)[1]
    raise AssertionError(f"{name}: нет {key}=")


# --------------------------------------------------------------------- содержимое


def test_all_units_and_the_installer_exist() -> None:
    for name in [*UNITS, "install.sh"]:
        assert (DEPLOY / name).is_file(), name


def test_units_are_templates_without_hardcoded_user_or_paths() -> None:
    for name in UNITS:
        text = "\n".join(directives(name))
        assert "User=root" not in text
        if name.endswith(".service"):
            assert "User=@USER@" in text
            assert "NoNewPrivileges=yes" in text
            assert "ProtectSystem=strict" in text
            assert "PrivateTmp=yes" in text


def test_sync_service_runs_data_sync_from_the_project_venv_with_data_dir() -> None:
    assert value("squeezelab-sync.service", "Type") == "oneshot"
    assert (
        value("squeezelab-sync.service", "ExecStart") == "@WORKDIR@/.venv/bin/squeezelab data sync"
    )
    assert value("squeezelab-sync.service", "Environment") == "SQUEEZELAB_DATA_DIR=@DATA_DIR@"
    assert value("squeezelab-sync.service", "RequiresMountsFor") == "@DATA_DIR@"
    assert value("squeezelab-sync.service", "ReadWritePaths") == "@DATA_DIR@"
    assert value("squeezelab-sync.service", "WorkingDirectory") == "@WORKDIR@"


def test_backup_is_rsync_archive_without_delete_from_data_to_backup_dir() -> None:
    exec_start = value("squeezelab-backup.service", "ExecStart")
    assert exec_start == "@RSYNC@ -a @DATA_DIR@/ @BACKUP_DIR@/"
    for line in directives("squeezelab-backup.service"):
        if line.startswith("Exec"):
            assert "--delete" not in line  # данные не удаляются и в копии
    assert value("squeezelab-backup.service", "ReadOnlyPaths") == "@DATA_DIR@"
    assert value("squeezelab-backup.service", "ReadWritePaths") == "@BACKUP_DIR@"
    assert value("squeezelab-backup.service", "RequiresMountsFor") == "@DATA_DIR@ @BACKUP_DIR@"
    assert value("squeezelab-backup.service", "ConditionPathIsDirectory") == "@DATA_DIR@"


def test_timers_are_daily_persistent_and_backup_follows_sync() -> None:
    hours = {}
    for name, service in (
        ("squeezelab-sync.timer", "squeezelab-sync.service"),
        ("squeezelab-backup.timer", "squeezelab-backup.service"),
    ):
        calendar = value(name, "OnCalendar")
        match = re.fullmatch(r"\*-\*-\* (\d\d):(\d\d):00 UTC", calendar)
        assert match, calendar
        hours[name] = int(match.group(1)) * 60 + int(match.group(2))
        assert value(name, "Persistent") == "true"
        assert value(name, "Unit") == service
        assert value(name, "WantedBy") == "timers.target"
    assert hours["squeezelab-backup.timer"] - hours["squeezelab-sync.timer"] >= 30


@pytest.mark.skipif(shutil.which("systemd-analyze") is None, reason="нет systemd-analyze")
def test_calendar_expressions_are_valid_for_systemd() -> None:
    for name in ("squeezelab-sync.timer", "squeezelab-backup.timer"):
        result = subprocess.run(
            ["systemd-analyze", "calendar", value(name, "OnCalendar")],
            capture_output=True,
            text=True,
            check=False,
        )
        assert result.returncode == 0, result.stderr


# ---------------------------------------------------------------------- установщик

pytestmark_posix = pytest.mark.skipif(
    sys.platform == "win32" or shutil.which("bash") is None, reason="нужен bash (Linux)"
)


def _run_user() -> str | None:
    if os.geteuid() != 0:  # type: ignore[attr-defined,unused-ignore]
        return subprocess.run(
            ["id", "-un"], capture_output=True, text=True, check=True
        ).stdout.strip()
    probe = subprocess.run(["id", "nobody"], capture_output=True, check=False)
    return "nobody" if probe.returncode == 0 else None


@pytestmark_posix
def test_installer_has_valid_bash_syntax() -> None:
    result = subprocess.run(
        ["bash", "-n", str(DEPLOY / "install.sh")], capture_output=True, text=True, check=False
    )
    assert result.returncode == 0, result.stderr


def run_installer(
    tmp: Path, *args: str, user: str | None = None, extra: dict[str, str] | None = None
) -> subprocess.CompletedProcess[str]:
    env = {
        **os.environ,
        "ALLOW_NON_ROOT": "1",
        "SKIP_SYSTEMCTL": "1",
        "SYSTEMD_DIR": str(tmp / "systemd"),
        "SQUEEZELAB_DATA_DIR": str(tmp / "data"),
        "SQUEEZELAB_BACKUP_DIR": str(tmp / "backup"),
        "RSYNC_BIN": "/bin/true",
        "SQUEEZELAB_USER": user or _run_user() or "",
        **(extra or {}),
    }
    return subprocess.run(
        ["bash", str(DEPLOY / "install.sh"), *args],
        capture_output=True,
        text=True,
        env=env,
        check=False,
    )


needs_venv = pytest.mark.skipif(
    not (REPO / ".venv" / "bin" / "squeezelab").exists(), reason="нет .venv/bin/squeezelab"
)


@pytestmark_posix
@needs_venv
def test_installer_renders_units_and_is_idempotent(tmp_path: Path) -> None:
    if _run_user() is None:
        pytest.skip("нет подходящего пользователя")
    first = run_installer(tmp_path)
    assert first.returncode == 0, first.stderr
    assert "Готово." in first.stdout
    systemd = tmp_path / "systemd"
    for name in UNITS:
        text = (systemd / name).read_text(encoding="utf-8")
        assert not re.search(r"@[A-Z_]+@", text), name
    sync = (systemd / "squeezelab-sync.service").read_text(encoding="utf-8")
    assert f"ExecStart={REPO}/.venv/bin/squeezelab data sync" in sync
    assert f"Environment=SQUEEZELAB_DATA_DIR={tmp_path / 'data'}" in sync
    backup = (systemd / "squeezelab-backup.service").read_text(encoding="utf-8")
    assert f"ExecStart=/bin/true -a {tmp_path / 'data'}/ {tmp_path / 'backup'}/" in backup
    assert (tmp_path / "backup").is_dir()

    stamps = {n: (systemd / n).stat().st_mtime_ns for n in UNITS}
    again = run_installer(tmp_path)
    assert again.returncode == 0, again.stderr
    assert "Всё уже установлено, изменений нет." in again.stdout
    assert all(f"без изменений: {n}" in again.stdout for n in UNITS)
    assert {n: (systemd / n).stat().st_mtime_ns for n in UNITS} == stamps

    sync_template = (DEPLOY / "squeezelab-sync.service").read_text(encoding="utf-8")
    (systemd / "squeezelab-sync.service").write_text(sync_template, encoding="utf-8")  # «испорчен»
    repaired = run_installer(tmp_path)
    assert "установлен:" in repaired.stdout
    assert "@WORKDIR@" not in (systemd / "squeezelab-sync.service").read_text(encoding="utf-8")


@pytestmark_posix
@needs_venv
def test_installer_dry_run_writes_nothing(tmp_path: Path) -> None:
    if _run_user() is None:
        pytest.skip("нет подходящего пользователя")
    result = run_installer(tmp_path, "--dry-run")
    assert result.returncode == 0, result.stderr
    assert "[dry-run] записал бы" in result.stdout
    assert not (tmp_path / "systemd").exists()
    assert not (tmp_path / "backup").exists()


@pytestmark_posix
@needs_venv
def test_installer_refuses_root_missing_user_and_unknown_arguments(tmp_path: Path) -> None:
    root = run_installer(tmp_path, user="root")
    assert root.returncode == 1
    assert "не должны работать от root" in root.stderr
    ghost = run_installer(tmp_path, user="нет-такого-пользователя-xyz")
    assert ghost.returncode == 1
    assert "не существует" in ghost.stderr
    unknown = run_installer(tmp_path, "--bad", user="root")
    assert unknown.returncode == 2
    assert not (tmp_path / "systemd").exists()


@pytestmark_posix
@needs_venv
def test_installer_requires_root_unless_overridden(tmp_path: Path) -> None:
    if os.geteuid() == 0:  # type: ignore[attr-defined,unused-ignore]
        pytest.skip("тест от root")
    result = run_installer(tmp_path, extra={"ALLOW_NON_ROOT": "0"})
    assert result.returncode == 1
    assert "sudo" in result.stderr
