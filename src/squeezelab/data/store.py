"""Хранилище свечей в Parquet (P01-T02).

Раскладка: `{data_dir}/candles/{symbol}/{tf}/{YYYY-MM}.parquet`, месяц по UTC. Одна схема для
всех таймфреймов: `ts_open` int64 (мс UTC, начало свечи), `o, h, l, c, v` float64.

Свойства, на которые опирается остальной код:

- запись атомарна: данные пишутся во временный файл рядом, сбрасываются на диск и только потом
  переименовываются поверх старого (`os.replace`); при сбое старый файл остаётся целым;
- слияние идемпотентно: свечи с тем же `ts_open` не дублируются; повторная запись тех же данных
  не трогает файл (не меняется даже время изменения);
- если биржа прислала для уже сохранённого `ts_open` другие значения, побеждают новые, а в
  статистике записи это видно как `updated`;
- данные автоматически не удаляются (правило 4 задания);
- рядом с файлами месяцев лежит реестр `confirmed_missing.json`: свечи, которых у биржи нет
  (P01-T03). Реестр не данные, а пометка, чтобы пропуск не запрашивали бесконечно; когда свеча
  всё же появляется в хранилище, запись о ней из реестра убирается сама.
"""

from __future__ import annotations

import json
import os
import re
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from squeezelab.data.models import Candle, tf_to_ms

SCHEMA = pa.schema(
    [
        pa.field("ts_open", pa.int64(), nullable=False),
        pa.field("o", pa.float64(), nullable=False),
        pa.field("h", pa.float64(), nullable=False),
        pa.field("l", pa.float64(), nullable=False),
        pa.field("c", pa.float64(), nullable=False),
        pa.field("v", pa.float64(), nullable=False),
    ]
)

_SYMBOL_RE = re.compile(r"^[A-Z0-9]+-[A-Z0-9]+$")
_TF_RE = re.compile(r"^[1-9][0-9]*[mhdw]$")
_MONTH_FILE_RE = re.compile(r"^(\d{4})-(\d{2})\.parquet$")

_Row = tuple[float, float, float, float, float]


class StoreError(RuntimeError):
    """Файл хранилища не соответствует ожидаемой схеме или повреждён."""


@dataclass(frozen=True, slots=True)
class WriteStats:
    added: int = 0  # новых свечей
    updated: int = 0  # ts_open уже был, значения отличались: заменены новыми
    unchanged: int = 0  # ts_open уже был, значения те же
    files_written: int = 0

    def __add__(self, other: WriteStats) -> WriteStats:
        return WriteStats(
            self.added + other.added,
            self.updated + other.updated,
            self.unchanged + other.unchanged,
            self.files_written + other.files_written,
        )


@dataclass(frozen=True, slots=True)
class RawFile:
    """Файл месяца «как лежит»: строки в порядке файла, без слияния и дедупликации."""

    path: Path
    month: tuple[int, int]  # из имени файла
    rows: list[Candle]


@dataclass(frozen=True, slots=True)
class ConfirmedMissing:
    ts_open: int
    attempts: int  # сколько попыток докачки сделано до подтверждения
    confirmed_at_ms: int


CONFIRMED_FILE = "confirmed_missing.json"


def month_of(ts_ms: int) -> tuple[int, int]:
    moment = datetime.fromtimestamp(ts_ms / 1000, tz=UTC)
    return moment.year, moment.month


def month_start_ms(year: int, month: int) -> int:
    return int(datetime(year, month, 1, tzinfo=UTC).timestamp() * 1000)


class CandleStore:
    def __init__(self, data_dir: Path | str) -> None:
        self.root = Path(data_dir)

    # ------------------------------------------------------------------ пути

    def tf_dir(self, symbol: str, tf: str) -> Path:
        if not _SYMBOL_RE.match(symbol):
            raise ValueError(f"символ {symbol!r}: ожидается вид BNB-USDT")
        if not _TF_RE.match(tf):
            raise ValueError(f"таймфрейм {tf!r}: ожидается вид 15m, 1h, 1d")
        return self.root / "candles" / symbol / tf

    def month_path(self, symbol: str, tf: str, year: int, month: int) -> Path:
        return self.tf_dir(symbol, tf) / f"{year:04d}-{month:02d}.parquet"

    def month_files(self, symbol: str, tf: str) -> list[Path]:
        """Файлы месяцев по возрастанию. Временные файлы и чужие имена игнорируются."""
        directory = self.tf_dir(symbol, tf)
        if not directory.is_dir():
            return []
        return sorted(p for p in directory.iterdir() if _MONTH_FILE_RE.match(p.name))

    def confirmed_path(self, symbol: str, tf: str) -> Path:
        return self.tf_dir(symbol, tf) / CONFIRMED_FILE

    def symbols(self) -> list[str]:
        """Пары, для которых есть каталог со свечами."""
        base = self.root / "candles"
        if not base.is_dir():
            return []
        return sorted(p.name for p in base.iterdir() if p.is_dir() and _SYMBOL_RE.match(p.name))

    def timeframes(self, symbol: str) -> list[str]:
        """Таймфреймы пары, у которых есть файлы свечей, от короткого к длинному."""
        base = self.root / "candles" / symbol
        if not _SYMBOL_RE.match(symbol) or not base.is_dir():
            return []
        found = [p.name for p in base.iterdir() if p.is_dir() and _TF_RE.match(p.name)]
        return sorted((tf for tf in found if self.month_files(symbol, tf)), key=tf_to_ms)

    # ---------------------------------------------------------------- чтение

    def read_raw(self, symbol: str, tf: str) -> list[RawFile]:
        """Файлы месяцев «как есть»: для проверки качества (дубли, порядок, чужой месяц)."""
        files: list[RawFile] = []
        for path in self.month_files(symbol, tf):
            match = _MONTH_FILE_RE.match(path.name)
            if match is None:  # month_files отдаёт только такие имена
                raise StoreError(f"{path}: имя файла не вида ГГГГ-ММ.parquet")
            table = self._read_table(path)
            ts = table.column("ts_open").to_pylist()
            o, h, lo, c, v = (table.column(n).to_pylist() for n in ("o", "h", "l", "c", "v"))
            rows = [Candle(ts[i], o[i], h[i], lo[i], c[i], v[i]) for i in range(len(ts))]
            files.append(RawFile(path, (int(match.group(1)), int(match.group(2))), rows))
        return files

    def _read_rows(self, path: Path) -> dict[int, _Row]:
        table = self._read_table(path)
        ts = table.column("ts_open").to_pylist()
        o, h, lo, c, v = (table.column(name).to_pylist() for name in ("o", "h", "l", "c", "v"))
        return {t: (o[i], h[i], lo[i], c[i], v[i]) for i, t in enumerate(ts)}

    @staticmethod
    def _read_table(path: Path, columns: list[str] | None = None) -> pa.Table:
        try:
            table = pq.read_table(path, columns=columns)
        except (OSError, pa.ArrowException) as exc:
            raise StoreError(f"{path}: не удалось прочитать файл ({exc})") from exc
        expected = SCHEMA if columns is None else pa.schema([SCHEMA.field(n) for n in columns])
        if not table.schema.equals(expected):
            raise StoreError(f"{path}: схема файла не совпадает с ожидаемой\n{table.schema}")
        return table

    def timestamps(self, symbol: str, tf: str) -> set[int]:
        """Все сохранённые `ts_open` (читается только один столбец)."""
        found: set[int] = set()
        for path in self.month_files(symbol, tf):
            found.update(self._read_table(path, ["ts_open"]).column("ts_open").to_pylist())
        return found

    def read(
        self, symbol: str, tf: str, start_ms: int | None = None, end_ms: int | None = None
    ) -> list[Candle]:
        """Свечи с `start_ms` (включительно) до `end_ms` (исключительно), по возрастанию."""
        candles: list[Candle] = []
        for path in self.month_files(symbol, tf):
            for ts, (o, h, lo, c, v) in self._read_rows(path).items():
                if (start_ms is None or ts >= start_ms) and (end_ms is None or ts < end_ms):
                    candles.append(Candle(ts, o, h, lo, c, v))
        candles.sort(key=lambda candle: candle.ts_open)
        return candles

    # ---------------------------------------------------------------- запись

    def write(self, symbol: str, tf: str, candles: list[Candle]) -> WriteStats:
        """Добавляет свечи, раскладывая их по месяцам. Возвращает статистику слияния."""
        self.tf_dir(symbol, tf)  # проверка имён до любых действий с диском
        by_month: dict[tuple[int, int], dict[int, _Row]] = defaultdict(dict)
        for candle in candles:
            by_month[month_of(candle.ts_open)][candle.ts_open] = (
                candle.o,
                candle.h,
                candle.l,
                candle.c,
                candle.v,
            )
        total = WriteStats()
        for (year, month), incoming in sorted(by_month.items()):
            total += self._merge_month(self.month_path(symbol, tf, year, month), incoming)
        if total.added or total.updated:
            self._forget_confirmed(symbol, tf, (candle.ts_open for candle in candles))
        return total

    # ------------------------------------------------- реестр «подтверждённо отсутствуют»

    def confirmed_missing(self, symbol: str, tf: str) -> dict[int, ConfirmedMissing]:
        path = self.confirmed_path(symbol, tf)
        if not path.exists():
            return {}
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            items = payload["items"]
            return {
                int(i["ts_open"]): ConfirmedMissing(
                    int(i["ts_open"]), int(i["attempts"]), int(i["confirmed_at_ms"])
                )
                for i in items
            }
        except (OSError, ValueError, KeyError, TypeError) as exc:
            raise StoreError(f"{path}: не удалось прочитать реестр пропусков ({exc})") from exc

    def add_confirmed_missing(self, symbol: str, tf: str, items: Iterable[ConfirmedMissing]) -> int:
        """Добавляет пометки (существующие обновляются). Возвращает число новых."""
        current = self.confirmed_missing(symbol, tf)
        new = 0
        for item in items:
            new += item.ts_open not in current
            current[item.ts_open] = item
        if new or current:
            self._write_confirmed(symbol, tf, current)
        return new

    def _forget_confirmed(self, symbol: str, tf: str, present: Iterable[int]) -> None:
        """Свеча появилась в хранилище: пометка «отсутствует» о ней больше не верна."""
        if not self.confirmed_path(symbol, tf).exists():
            return
        current = self.confirmed_missing(symbol, tf)
        stale = current.keys() & set(present)
        if stale:
            for ts in stale:
                del current[ts]
            self._write_confirmed(symbol, tf, current)

    def _write_confirmed(self, symbol: str, tf: str, items: dict[int, ConfirmedMissing]) -> None:
        path = self.confirmed_path(symbol, tf)
        payload = {
            "version": 1,
            "items": [
                {"ts_open": i.ts_open, "attempts": i.attempts, "confirmed_at_ms": i.confirmed_at_ms}
                for i in sorted(items.values(), key=lambda x: x.ts_open)
            ],
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(f"{path.name}.tmp-{os.getpid()}")
        try:
            tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")
            with tmp.open("rb+") as handle:
                os.fsync(handle.fileno())
            os.replace(tmp, path)
        except BaseException:
            tmp.unlink(missing_ok=True)
            raise

    def _merge_month(self, path: Path, incoming: dict[int, _Row]) -> WriteStats:
        existing = self._read_rows(path) if path.exists() else {}
        added = updated = unchanged = 0
        for ts, row in incoming.items():
            if ts not in existing:
                added += 1
            elif existing[ts] == row:
                unchanged += 1
            else:
                updated += 1
        if added == 0 and updated == 0:
            return WriteStats(unchanged=unchanged)  # файл не трогаем
        merged = {**existing, **incoming}
        self._atomic_write(path, merged)
        return WriteStats(added, updated, unchanged, files_written=1)

    @staticmethod
    def _atomic_write(path: Path, rows: dict[int, _Row]) -> None:
        ordered = sorted(rows)
        table = pa.table(
            {
                "ts_open": pa.array(ordered, type=pa.int64()),
                "o": pa.array([rows[t][0] for t in ordered], type=pa.float64()),
                "h": pa.array([rows[t][1] for t in ordered], type=pa.float64()),
                "l": pa.array([rows[t][2] for t in ordered], type=pa.float64()),
                "c": pa.array([rows[t][3] for t in ordered], type=pa.float64()),
                "v": pa.array([rows[t][4] for t in ordered], type=pa.float64()),
            },
            schema=SCHEMA,
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(f"{path.name}.tmp-{os.getpid()}")
        try:
            pq.write_table(table, tmp, compression="zstd")
            with tmp.open("rb+") as handle:
                os.fsync(handle.fileno())
            os.replace(tmp, path)
        except BaseException:
            tmp.unlink(missing_ok=True)
            raise
