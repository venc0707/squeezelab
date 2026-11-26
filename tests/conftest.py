"""Общие настройки тестов: переменные окружения проекта не должны влиять на результат.

На сервере в сессии обычно экспортирована SQUEEZELAB_DATA_DIR, и без этой фикстуры тесты, которые
ждут значение по умолчанию (например, вывод `config check`), падали бы только там.
"""

from __future__ import annotations

import pytest

PROJECT_ENV = ("SQUEEZELAB_DATA_DIR", "SQUEEZELAB_CONFIG_DIR")


@pytest.fixture(autouse=True)
def _isolate_project_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in PROJECT_ENV:
        monkeypatch.delenv(name, raising=False)
