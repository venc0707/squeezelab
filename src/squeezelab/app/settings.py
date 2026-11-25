"""Схемы конфигов (pydantic v2), загрузка YAML, секреты из окружения и хэш конфига.

Принципы:
- Любая опечатка в ключе — ошибка (extra="forbid"), модели неизменяемы.
- Открытые решения хранятся как `null` и не додумываются. Пока они не закрыты,
  режим работы может быть только `signals_only`.
- Секреты читаются только из окружения (`.env` вне git) и маскируются в repr.
- Хэш считается от проверенных моделей, а не от текста YAML: комментарии, порядок ключей и
  `6` против `6.0` на хэш не влияют. Хэш делится на две части: `strategy_hash` (presets,
  exits, risk, symbols; идёт в журнал и сигналы) и `app_hash` (app.yaml; только диагностика).
- Секреты читаются по областям: рекордеру не нужны ключи BingX, команде BingX не нужен Telegram.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import date
from enum import StrEnum
from pathlib import Path
from typing import Annotated, Any, Literal, TypeVar
from urllib.parse import urlsplit

import yaml
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    SecretStr,
    ValidationError,
    field_validator,
    model_validator,
)
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict

CONFIG_DIR_ENV = "SQUEEZELAB_CONFIG_DIR"
DEFAULT_CONFIG_DIR = "config"

# Файл конфига для каждой секции ProjectConfig.
CONFIG_FILES: dict[str, str] = {
    "app": "app.yaml",
    "presets": "presets.yaml",
    "exits": "exits.yaml",
    "risk": "risk.yaml",
    "symbols": "symbols.yaml",
}

DATA_DIR_ENV = "SQUEEZELAB_DATA_DIR"
DEFAULT_DATA_DIR = "data"

_TF_PATTERN = r"^[1-9][0-9]*[mhdw]$"
_SYMBOL_PATTERN = r"^[A-Z0-9]+-[A-Z0-9]+$"
_SUM_EPS = 1e-9

# Хэш делится на две части. Состав strategy_hash — решение мастера (Phase 01).
STRATEGY_SECTIONS = ("presets", "exits", "risk", "symbols")

# Режимы, которым нужен приоритет пресетов (D-01). Бэктест приоритет не требует: матрица
# пересечения пресетов считается в Phase 4 как раз для того, чтобы его выбрать.
_MODES_NEEDING_PRIORITY = frozenset({"paper_vst", "live"})


class ConfigError(Exception):
    """Конфиг не найден, не читается или не прошёл проверку. Текст готов к показу."""


class Mode(StrEnum):
    SIGNALS_ONLY = "signals_only"
    BACKTEST = "backtest"
    PAPER_VST = "paper_vst"
    LIVE = "live"


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    def unresolved(self) -> list[str]:
        """Имена полей верхнего уровня, которые равны null (решение ещё не принято)."""
        return [name for name, value in self if value is None]


# --------------------------------------------------------------------------- app.yaml


class PathsConfig(_Strict):
    # Строки, а не Path: хэш не должен зависеть от ОС (разделители путей).
    # data_dir: не задан = ./data. Переменная SQUEEZELAB_DATA_DIR приоритетнее (resolve_data_dir).
    data_dir: str | None = Field(default=None, min_length=1)
    logs_dir: str = Field(min_length=1)


class RecorderConfig(_Strict):
    """Настройки рекордера. Живут в app.yaml, а не в symbols.yaml: смена списка пар рекордера
    не должна менять strategy_hash."""

    symbols: list[str] = Field(default_factory=lambda: ["BNB-USDT"], min_length=1)
    # Нижняя граница 10 с — защита от случайного шквала запросов (лимит биржи 500 за 10 с).
    oi_interval_s: int = Field(default=60, ge=10, le=3600)
    funding_interval_s: int = Field(default=60, ge=10, le=3600)

    @field_validator("symbols")
    @classmethod
    def _symbols_ok(cls, value: list[str]) -> list[str]:
        pattern = re.compile(_SYMBOL_PATTERN)
        for symbol in value:
            if not pattern.match(symbol):
                raise ValueError(f"символ {symbol!r}: ожидается вид BNB-USDT")
        if len(set(value)) != len(value):
            raise ValueError("символ указан дважды")
        return value


class DownloaderConfig(_Strict):
    """Какие пары и ТФ держит в порядке `data sync` (таймер squeezelab-sync). Живёт в app.yaml, а
    не в symbols.yaml: смена списка не должна менять strategy_hash."""

    symbols: list[str] = Field(default_factory=lambda: ["BNB-USDT"], min_length=1)
    timeframes: list[str] = Field(
        default_factory=lambda: ["15m", "1m", "1h", "4h", "1d"], min_length=1
    )

    @field_validator("symbols")
    @classmethod
    def _symbols_ok(cls, value: list[str]) -> list[str]:
        pattern = re.compile(_SYMBOL_PATTERN)
        for symbol in value:
            if not pattern.match(symbol):
                raise ValueError(f"символ {symbol!r}: ожидается вид BNB-USDT")
        if len(set(value)) != len(value):
            raise ValueError("символ указан дважды")
        return value

    @field_validator("timeframes")
    @classmethod
    def _timeframes_ok(cls, value: list[str]) -> list[str]:
        pattern = re.compile(_TF_PATTERN)
        for tf in value:
            if not pattern.match(tf):
                raise ValueError(f"таймфрейм {tf!r}: ожидается вид 15m, 1h, 1d")
        if len(set(value)) != len(value):
            raise ValueError("таймфрейм указан дважды")
        return value


class AppConfig(_Strict):
    mode: Mode = Mode.SIGNALS_ONLY
    timezone: Literal["UTC"] = "UTC"
    paths: PathsConfig
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
    recorder: RecorderConfig = Field(default_factory=RecorderConfig)
    downloader: DownloaderConfig = Field(default_factory=DownloaderConfig)


# ----------------------------------------------------------------------- presets.yaml


class PresetParams(_Strict):
    """Параметры детектора поджимов (имена и границы — по Pine-скрипту)."""

    tf: str = Field(pattern=_TF_PATTERN)
    length: int = Field(ge=2)
    source: Literal["close"]
    level: float = Field(ge=0, le=50)
    min_dist: int = Field(ge=0)
    min_in: int = Field(ge=1)
    gap_tol: int = Field(ge=0)
    min_touches: int = Field(ge=3)
    min_depth: float = Field(ge=0)
    max_depth: float = Field(ge=0, le=100)
    min_grow: float = Field(ge=0)
    max_slope: float = Field(ge=0)  # пунктов на 100 баров, 0 = без ограничения
    tol: float = Field(ge=0)
    max_len: int = Field(ge=0)  # баров, 0 = без ограничения
    squeeze_upper: bool
    squeeze_lower: bool
    priority: int | None = Field(default=None, ge=1)  # открытое решение, пока null

    @model_validator(mode="after")
    def _check_ranges(self) -> PresetParams:
        if self.min_depth > self.max_depth:
            raise ValueError(f"min_depth ({self.min_depth}) больше max_depth ({self.max_depth})")
        if not (self.squeeze_upper or self.squeeze_lower):
            raise ValueError("включи хотя бы один поджим: squeeze_upper или squeeze_lower")
        return self


class PresetsConfig(_Strict):
    presets: dict[str, PresetParams] = Field(min_length=1)

    @field_validator("presets")
    @classmethod
    def _names(cls, value: dict[str, PresetParams]) -> dict[str, PresetParams]:
        for name in value:
            if not name or not all(ch.isalnum() or ch == "_" for ch in name):
                raise ValueError(f"недопустимое имя пресета {name!r}: только буквы, цифры, _")
        return value

    @model_validator(mode="after")
    def _unique_priorities(self) -> PresetsConfig:
        taken: dict[int, str] = {}
        for name, preset in self.presets.items():
            if preset.priority is None:
                continue
            if preset.priority in taken:
                raise ValueError(
                    f"priority {preset.priority} задан у двух пресетов: "
                    f"{taken[preset.priority]} и {name}"
                )
            taken[preset.priority] = name
        return self


# -------------------------------------------------------------------------- exits.yaml


class LadderConfig(_Strict):
    r_multiples: list[float] = Field(min_length=1, max_length=5)
    shares_pct: list[float] = Field(min_length=1, max_length=5)

    @model_validator(mode="after")
    def _check(self) -> LadderConfig:
        if len(self.r_multiples) != len(self.shares_pct):
            raise ValueError(
                f"r_multiples ({len(self.r_multiples)}) и shares_pct ({len(self.shares_pct)}) "
                "должны быть одной длины"
            )
        if any(r <= 0 for r in self.r_multiples):
            raise ValueError("все R должны быть больше 0")
        if any(b <= a for a, b in zip(self.r_multiples, self.r_multiples[1:], strict=False)):
            raise ValueError(f"R должны строго возрастать, получено {self.r_multiples}")
        if any(s <= 0 for s in self.shares_pct):
            raise ValueError("все доли должны быть больше 0")
        total = sum(self.shares_pct)
        if abs(total - 100.0) > _SUM_EPS:
            raise ValueError(f"сумма долей равна {total}, должна быть 100")
        return self


class ExitsConfig(_Strict):
    ladder: LadderConfig
    breakeven_after: str = Field(pattern=r"^TP[1-5]$")
    # Запас БУ в б.п. (1 б.п. = 0,01 %) от цены входа; значение — открытое решение (null).
    breakeven_buffer_bps: float | None = Field(default=None, ge=0)
    tp_order_type: Literal["limit", "take_profit_market"] | None = None

    @model_validator(mode="after")
    def _breakeven_in_ladder(self) -> ExitsConfig:
        index = int(self.breakeven_after[2:])
        if index > len(self.ladder.r_multiples):
            raise ValueError(
                f"breakeven_after={self.breakeven_after}, а в лесенке "
                f"{len(self.ladder.r_multiples)} целей"
            )
        return self


# --------------------------------------------------------------------------- risk.yaml


class RiskConfig(_Strict):
    """Все поля — открытые решения. Пока null, режимы кроме signals_only запрещены."""

    risk_per_trade_pct: float | None = Field(default=None, gt=0, le=100)
    max_leverage: int | None = Field(default=None, ge=1)
    daily_loss_limit_pct: float | None = Field(default=None, gt=0, le=100)
    weekly_loss_limit_pct: float | None = Field(default=None, gt=0, le=100)
    stop_buffer_atr: float | None = Field(default=None, ge=0)
    min_stop_atr: float | None = Field(default=None, gt=0)
    max_stop_atr: float | None = Field(default=None, gt=0)
    position_mode: Literal["hedge", "one_way"] | None = None
    margin_mode: Literal["isolated", "cross"] | None = None

    @model_validator(mode="after")
    def _stop_range(self) -> RiskConfig:
        if (
            self.min_stop_atr is not None
            and self.max_stop_atr is not None
            and self.min_stop_atr >= self.max_stop_atr
        ):
            raise ValueError(
                f"min_stop_atr ({self.min_stop_atr}) должен быть меньше "
                f"max_stop_atr ({self.max_stop_atr})"
            )
        return self


# ----------------------------------------------------------------------- symbols.yaml


class SymbolConfig(_Strict):
    symbol: str = Field(pattern=_SYMBOL_PATTERN)
    tf: str = Field(pattern=_TF_PATTERN)
    data_start: date
    warmup_bars: int = Field(ge=0)


class SymbolsConfig(_Strict):
    symbols: list[SymbolConfig] = Field(min_length=1)

    @model_validator(mode="after")
    def _unique(self) -> SymbolsConfig:
        seen: set[tuple[str, str]] = set()
        for item in self.symbols:
            key = (item.symbol, item.tf)
            if key in seen:
                raise ValueError(f"пара {item.symbol} на {item.tf} указана дважды")
            seen.add(key)
        return self


# ---------------------------------------------------------------------- весь проект


class ProjectConfig(_Strict):
    app: AppConfig
    presets: PresetsConfig
    exits: ExitsConfig
    risk: RiskConfig
    symbols: SymbolsConfig

    @model_validator(mode="after")
    def _mode_needs_decisions(self) -> ProjectConfig:
        if self.app.mode is Mode.SIGNALS_ONLY:
            return self
        missing = [f"risk.{n}" for n in self.risk.unresolved()]
        missing += [f"exits.{n}" for n in self.exits.unresolved()]
        if self.app.mode.value in _MODES_NEEDING_PRIORITY:
            missing += self._null_priorities()
        if missing:
            raise ValueError(
                f"режим {self.app.mode.value} запрещён, пока открыты решения (null): "
                + ", ".join(missing)
            )
        return self

    @model_validator(mode="after")
    def _warmup_covers_presets(self) -> ProjectConfig:
        for sym in self.symbols.symbols:
            for name, preset in self.presets.presets.items():
                if preset.tf == sym.tf and sym.warmup_bars < preset.length:
                    raise ValueError(
                        f"{sym.symbol}: warmup_bars={sym.warmup_bars} меньше length="
                        f"{preset.length} пресета {name}: %R не успеет прогреться"
                    )
        return self

    def _null_priorities(self) -> list[str]:
        return [
            f"presets.{name}.priority"
            for name, preset in self.presets.presets.items()
            if preset.priority is None
        ]

    def unresolved_decisions(self) -> list[str]:
        """Все открытые (null) решения для сводки в `config check`."""
        items = [f"risk.{n}" for n in self.risk.unresolved()]
        items += [f"exits.{n}" for n in self.exits.unresolved()]
        items += self._null_priorities()
        return items


# ------------------------------------------------------------------------- загрузка


def resolve_config_dir(config_dir: Path | str | None = None) -> Path:
    """Аргумент → переменная SQUEEZELAB_CONFIG_DIR → ./config."""
    if config_dir is not None:
        return Path(config_dir)
    return Path(os.environ.get(CONFIG_DIR_ENV) or DEFAULT_CONFIG_DIR)


def _read_yaml(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise ConfigError(f"{path}: файл не найден")
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, yaml.YAMLError) as exc:
        raise ConfigError(
            f"{path}: не удалось прочитать YAML ({type(exc).__name__}: {exc})"
        ) from exc
    if not isinstance(data, dict):
        raise ConfigError(f"{path}: на верхнем уровне должен быть словарь ключ: значение")
    # Ключи «x-...» — служебные (якоря YAML для общих блоков), в схему не попадают.
    return {k: v for k, v in data.items() if not (isinstance(k, str) and k.startswith("x-"))}


def format_validation_error(source: str, exc: ValidationError) -> str:
    lines = [f"{source}: конфиг не прошёл проверку"]
    for err in exc.errors():
        where = ".".join(str(part) for part in err["loc"]) or "(корень)"
        message = str(err["msg"]).removeprefix("Value error, ")
        lines.append(f"  - {where}: {message}")
    return "\n".join(lines)


def load_project_config(config_dir: Path | str | None = None) -> ProjectConfig:
    directory = resolve_config_dir(config_dir)
    models: dict[str, type[_Strict]] = {
        "app": AppConfig,
        "presets": PresetsConfig,
        "exits": ExitsConfig,
        "risk": RiskConfig,
        "symbols": SymbolsConfig,
    }
    sections: dict[str, Any] = {}
    for key, filename in CONFIG_FILES.items():
        path = directory / filename
        raw = _read_yaml(path)
        try:
            sections[key] = models[key].model_validate(raw)
        except ValidationError as exc:
            raise ConfigError(format_validation_error(str(path), exc)) from exc
    try:
        return ProjectConfig(**sections)
    except ValidationError as exc:
        raise ConfigError(
            format_validation_error(f"{directory} (совместная проверка)", exc)
        ) from exc


def _canonical_json(data: object) -> str:
    """Канонический JSON: ключи отсортированы, без пробелов, UTF-8."""
    return json.dumps(
        data, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    )


def canonical_strategy_json(config: ProjectConfig) -> str:
    return _canonical_json(config.model_dump(mode="json", include=set(STRATEGY_SECTIONS)))


def canonical_app_json(config: ProjectConfig) -> str:
    return _canonical_json(config.app.model_dump(mode="json"))


def strategy_hash(config: ProjectConfig) -> str:
    """sha256 от presets, exits, risk и symbols. Идёт в журнал и в каждый сигнал."""
    return hashlib.sha256(canonical_strategy_json(config).encode("utf-8")).hexdigest()


def app_hash(config: ProjectConfig) -> str:
    """sha256 от app.yaml (режим, пути, рекордер, логи). Только для диагностики."""
    return hashlib.sha256(canonical_app_json(config).encode("utf-8")).hexdigest()


# ------------------------------------------------------------------------- путь данных

DataDirSource = Literal["env", "app.yaml", "default"]


@dataclass(frozen=True, slots=True)
class DataDir:
    path: Path
    source: DataDirSource


def resolve_data_dir(app: AppConfig, environ: Mapping[str, str] | None = None) -> DataDir:
    """Каталог данных: переменная SQUEEZELAB_DATA_DIR → `paths.data_dir` в app.yaml → ./data.

    Относительный путь считается от текущего каталога процесса. Пустая переменная окружения
    считается незаданной. Каталог не создаётся и не проверяется: это дело хранилища.
    Переменная читается только из окружения процесса (не из .env), как и SQUEEZELAB_CONFIG_DIR.
    """
    env = os.environ if environ is None else environ
    raw = env.get(DATA_DIR_ENV, "").strip()
    if raw:
        return DataDir(Path(raw).expanduser(), "env")
    if app.paths.data_dir:
        return DataDir(Path(app.paths.data_dir).expanduser(), "app.yaml")
    return DataDir(Path(DEFAULT_DATA_DIR), "default")


# --------------------------------------------------------------------------- секреты


_SECRETS_CONFIG = SettingsConfigDict(
    env_file=".env",
    env_file_encoding="utf-8",
    extra="ignore",
    case_sensitive=False,
)


class TelegramSecrets(BaseSettings):
    """Секреты Telegram (все необязательны). Этого достаточно рекордеру и notify-test.

    Значения хранятся как SecretStr: в repr и str видно только «**********».
    """

    model_config = _SECRETS_CONFIG

    telegram_bot_token: SecretStr | None = None
    telegram_chat_ids: Annotated[list[int], NoDecode] = Field(default_factory=list)
    # Прокси только для Telegram (BingX всегда идёт напрямую). Адрес может содержать логин и
    # пароль, поэтому это секрет.
    telegram_proxy_url: SecretStr | None = None

    @field_validator("telegram_bot_token", "telegram_proxy_url", mode="before")
    @classmethod
    def _blank_is_none(cls, value: object) -> object:
        # Пустая строка из шаблона .env.example означает «не задано».
        if isinstance(value, str):
            value = value.strip()
            return value or None
        return value

    @field_validator("telegram_proxy_url")
    @classmethod
    def _proxy_url_shape(cls, value: SecretStr | None) -> SecretStr | None:
        if value is None:
            return None
        # Тексты ошибок не содержат значения: в адресе бывает пароль.
        try:
            parts = urlsplit(value.get_secret_value())
            _ = parts.port  # неверный порт даёт ValueError
        except ValueError:
            raise ValueError("не удалось разобрать адрес прокси") from None
        if parts.scheme not in ("http", "https") or not parts.hostname:
            raise ValueError("ожидается адрес вида http://127.0.0.1:10809 (схемы http и https)")
        return value

    @field_validator("telegram_chat_ids", mode="before")
    @classmethod
    def _parse_chat_ids(cls, value: object) -> object:
        # В .env белый список пишется через запятую: TELEGRAM_CHAT_IDS=123456,-1001234567890
        if value is None:
            return []
        if isinstance(value, str):
            parts = [p.strip() for p in value.split(",") if p.strip()]
            try:
                return [int(p) for p in parts]
            except ValueError as exc:
                raise ValueError("ожидаются целые числа через запятую") from exc
        return value

    @field_validator("telegram_chat_ids")
    @classmethod
    def _chat_ids_nonzero(cls, value: list[int]) -> list[int]:
        if any(cid == 0 for cid in value):
            raise ValueError("chat_id не может быть 0")
        return value

    @property
    def has_telegram(self) -> bool:
        return self.telegram_bot_token is not None and bool(self.telegram_chat_ids)

    def secret_values(self) -> list[str]:
        """Открытые значения секретов — только для маскирования в логах."""
        raw = [self.telegram_bot_token, self.telegram_proxy_url]
        return [s.get_secret_value() for s in raw if s is not None and s.get_secret_value()]

    def presence(self) -> dict[str, bool]:
        """Что задано (без значений) — для `config check`."""
        return {
            "TELEGRAM_BOT_TOKEN": self.telegram_bot_token is not None,
            "TELEGRAM_CHAT_IDS": bool(self.telegram_chat_ids),
            "TELEGRAM_PROXY_URL": self.telegram_proxy_url is not None,
        }


class BingXSecrets(BaseSettings):
    """Секреты BingX. Нужны только командам, которые ходят в приватный API."""

    model_config = _SECRETS_CONFIG

    bingx_api_key: SecretStr | None = None
    bingx_api_secret: SecretStr | None = None
    bingx_env: Literal["vst", "live"] = "vst"

    @field_validator("bingx_api_key", "bingx_api_secret", mode="before")
    @classmethod
    def _blank_is_none(cls, value: object) -> object:
        if isinstance(value, str):
            value = value.strip()
            return value or None
        return value

    @field_validator("bingx_env", mode="before")
    @classmethod
    def _env_lower(cls, value: object) -> object:
        return value.strip().lower() if isinstance(value, str) else value

    @property
    def has_bingx_credentials(self) -> bool:
        return self.bingx_api_key is not None and self.bingx_api_secret is not None

    def secret_values(self) -> list[str]:
        raw = [self.bingx_api_key, self.bingx_api_secret]
        return [s.get_secret_value() for s in raw if s is not None and s.get_secret_value()]

    def presence(self) -> dict[str, bool]:
        return {
            "BINGX_API_KEY": self.bingx_api_key is not None,
            "BINGX_API_SECRET": self.bingx_api_secret is not None,
        }


class Secrets(TelegramSecrets, BingXSecrets):
    """Все секреты проекта сразу. Нужен только `config check`; остальным командам — своя область."""

    def secret_values(self) -> list[str]:
        return BingXSecrets.secret_values(self) + TelegramSecrets.secret_values(self)

    def presence(self) -> dict[str, bool]:
        return {**BingXSecrets.presence(self), **TelegramSecrets.presence(self)}


_S = TypeVar("_S", bound=BaseSettings)


def _load(cls: type[_S], env_file: Path | str | None) -> _S:
    try:
        return cls(_env_file=env_file)
    except ValidationError as exc:
        # Значения секретов в сообщение не попадают: только имена полей и тексты проверок.
        lines = [f"{env_file}: секреты не прошли проверку"]
        for err in exc.errors():
            where = ".".join(str(part) for part in err["loc"]).upper()
            lines.append(f"  - {where}: {str(err['msg']).removeprefix('Value error, ')}")
        raise ConfigError("\n".join(lines)) from None


def load_secrets(env_file: Path | str | None = ".env") -> Secrets:
    """Все секреты. Отсутствующий файл .env — не ошибка (могут быть переменные окружения)."""
    return _load(Secrets, env_file)


def load_telegram_secrets(env_file: Path | str | None = ".env") -> TelegramSecrets:
    """Только Telegram: ключей BingX в такой объект не попадает даже при их наличии в .env."""
    return _load(TelegramSecrets, env_file)


def load_bingx_secrets(env_file: Path | str | None = ".env") -> BingXSecrets:
    """Только BingX."""
    return _load(BingXSecrets, env_file)
