"""Командная строка SqueezeLab.

Коды выхода: 0 — успех, 1 — сбой операции (сеть, API) или дефекты данных, 2 — ошибка
конфигурации или аргументов, 3 — после `data sync` остались неподтверждённые пропуски.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path

from squeezelab import __version__
from squeezelab.app.logging_setup import configure_logging
from squeezelab.app.settings import (
    DATA_DIR_ENV,
    ConfigError,
    ProjectConfig,
    app_hash,
    load_bingx_secrets,
    load_project_config,
    load_secrets,
    load_telegram_secrets,
    resolve_config_dir,
    resolve_data_dir,
    strategy_hash,
)
from squeezelab.data.bingx_rest import (
    LIVE_BASE_URL,
    MAX_KLINES_LIMIT,
    VALID_INTERVALS,
    VST_BASE_URL,
    BingXError,
    BingXPublicClient,
    BingXReadOnlyClient,
    describe_shape,
    extract_permissions,
    permission_fields,
)
from squeezelab.data.downloader import (
    DEFAULT_START,
    WINDOW_BARS,
    RateLimiter,
    RetryingClient,
    RetryTally,
    download_tf,
    render_report,
)
from squeezelab.data.models import (
    detect_order,
    format_ts,
    is_candle_closed,
    is_contiguous,
    tf_to_ms,
)
from squeezelab.data.probe import (
    DEFAULT_ANCHOR,
    DEFAULT_VOLUME_BAR,
    build_cases,
    build_paging_cases,
    coverage_samples,
    derive_coverage_findings,
    derive_depth_findings,
    derive_findings,
    derive_paging_findings,
    render_coverage,
    render_depth,
    render_results,
    run_coverage_probe,
    run_depth_probe,
    run_probe,
    to_ms,
)
from squeezelab.data.quality import (
    check_series,
    cross_check,
    cross_summary_line,
    render_markdown,
    series_summary_line,
)
from squeezelab.data.resample import (
    compare_layers,
    render_comparison,
    render_resample,
    render_resample_markdown,
    resample_range,
)
from squeezelab.data.store import RESAMPLED_LAYER, CandleStore, StoreError
from squeezelab.data.sync import render_sync, sync_gaps
from squeezelab.notify.telegram import TelegramClient, TelegramError, build_test_message

EXIT_OK = 0
EXIT_FAILURE = 1
EXIT_CONFIG = 2
EXIT_GAPS = 3


def _common_options() -> argparse.ArgumentParser:
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument(
        "--config-dir",
        type=Path,
        default=None,
        help="каталог с YAML-конфигами (по умолчанию $SQUEEZELAB_CONFIG_DIR или ./config)",
    )
    common.add_argument(
        "--env-file",
        type=Path,
        default=Path(".env"),
        help="файл с секретами (по умолчанию ./.env)",
    )
    return common


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="squeezelab",
        description="SqueezeLab: торговый бот для BingX USDT-M perpetual.",
    )
    parser.add_argument("--version", action="version", version=f"squeezelab {__version__}")
    common = _common_options()
    sub = parser.add_subparsers(dest="command", metavar="КОМАНДА")

    config = sub.add_parser("config", help="работа с конфигами")
    config_sub = config.add_subparsers(dest="config_command", metavar="ДЕЙСТВИЕ", required=True)
    config_sub.add_parser(
        "check", parents=[common], help="проверить конфиги и напечатать хэш"
    ).set_defaults(handler=cmd_config_check)

    data = sub.add_parser("data", help="данные: загрузка и проверка свечей")
    data_sub = data.add_subparsers(dest="data_command", metavar="ДЕЙСТВИЕ", required=True)
    download = data_sub.add_parser(
        "download",
        parents=[common],
        help="скачать закрытые свечи BingX в Parquet (повторный запуск докачивает)",
    )
    download.add_argument(
        "--symbol",
        action="append",
        default=None,
        help="пара, можно несколько раз; по умолчанию первая из symbols.yaml",
    )
    download.add_argument(
        "--tf",
        action="append",
        default=None,
        help="таймфрейм, можно несколько раз; по умолчанию 15m и 1m",
    )
    download.add_argument(
        "--start",
        default=DEFAULT_START.strftime("%Y-%m-%dT%H:%M"),
        help="начало периода, UTC (ГГГГ-ММ-ДД или ГГГГ-ММ-ДДTЧЧ:ММ); по умолчанию "
        + DEFAULT_START.strftime("%Y-%m-%d %H:%M"),
    )
    download.add_argument(
        "--end",
        default=None,
        help="конец периода, UTC, исключительно; по умолчанию последняя закрытая свеча",
    )
    download.add_argument("--base-url", default=LIVE_BASE_URL, help=f"по умолчанию {LIVE_BASE_URL}")
    download.add_argument(
        "--window-bars",
        type=int,
        default=WINDOW_BARS,
        help=f"баров в окне запроса (1..1440); по умолчанию {WINDOW_BARS} (предел ответа биржи)",
    )
    download.set_defaults(handler=cmd_data_download)

    check = data_sub.add_parser(
        "check",
        parents=[common],
        help="проверить качество сохранённых свечей, сверить ТФ, записать отчёт",
    )
    check.add_argument("--symbol", action="append", default=None, help="по умолчанию все пары")
    check.add_argument("--tf", action="append", default=None, help="по умолчанию все ТФ пары")
    check.add_argument("--start", default=None, help="начало проверяемого периода, UTC")
    check.add_argument("--end", default=None, help="конец периода, UTC, исключительно")
    check.add_argument(
        "--cross",
        action="append",
        default=None,
        metavar="КРУПНЫЙ:МЕЛКИЙ",
        help="кросс-проверка ТФ, например 1h:15m (можно несколько раз); по умолчанию 15m:1m",
    )
    check.add_argument("--no-cross", action="store_true", help="не делать кросс-проверку")
    check.add_argument(
        "--layer",
        choices=("native", "resampled", "all"),
        default="all",
        help="слой хранилища: нативные свечи биржи, ресемпл или оба (по умолчанию)",
    )
    check.add_argument(
        "--report-dir", default=None, help="каталог отчёта; по умолчанию {каталог данных}/reports"
    )
    check.set_defaults(handler=cmd_data_check)

    sync = data_sub.add_parser(
        "sync",
        parents=[common],
        help="докачать новые свечи и пропуски; отсутствующие у биржи пометить; код 3 при остатке",
    )
    sync.add_argument(
        "--symbol", action="append", default=None, help="по умолчанию downloader.symbols"
    )
    sync.add_argument(
        "--tf", action="append", default=None, help="по умолчанию downloader.timeframes"
    )
    sync.add_argument("--start", default=None, help="по умолчанию первая сохранённая свеча, UTC")
    sync.add_argument("--end", default=None, help="по умолчанию последняя закрытая свеча, UTC")
    sync.add_argument("--base-url", default=LIVE_BASE_URL, help=f"по умолчанию {LIVE_BASE_URL}")
    sync.set_defaults(handler=cmd_data_sync)

    resample = data_sub.add_parser(
        "resample",
        parents=[common],
        help="собрать 15m из 1m в отдельный слой и сверить с нативным 15m (вердикт)",
    )
    resample.add_argument(
        "--symbol", action="append", default=None, help="по умолчанию downloader.symbols"
    )
    resample.add_argument("--from", dest="fine_tf", default="1m", help="мелкий ТФ, по умолчанию 1m")
    resample.add_argument(
        "--to", dest="coarse_tf", default="15m", help="крупный ТФ, по умолчанию 15m"
    )
    resample.add_argument("--start", default=None, help="начало периода, UTC")
    resample.add_argument("--end", default=None, help="конец периода, UTC, исключительно")
    resample.add_argument("--no-compare", action="store_true", help="не сверять с нативным")
    resample.add_argument(
        "--report-dir", default=None, help="каталог отчёта; по умолчанию {каталог данных}/reports"
    )
    resample.set_defaults(handler=cmd_data_resample)

    notify = sub.add_parser(
        "notify-test",
        parents=[common],
        help="отправить тестовое сообщение в Telegram (только в чаты из белого списка)",
    )
    notify.add_argument(
        "--dry-run", action="store_true", help="показать сообщение и адресатов, ничего не отправляя"
    )
    notify.set_defaults(handler=cmd_notify_test)

    smoke = sub.add_parser(
        "bingx-smoke",
        parents=[common],
        help="публичный запрос свечей BingX (без ключей)",
    )
    smoke.add_argument("--symbol", default=None, help="по умолчанию первая пара из symbols.yaml")
    smoke.add_argument("--interval", default=None, help="по умолчанию её таймфрейм")
    smoke.add_argument("--limit", type=int, default=20, help="сколько свечей запросить (20)")
    smoke.add_argument("--base-url", default=LIVE_BASE_URL, help=f"по умолчанию {LIVE_BASE_URL}")
    smoke.add_argument(
        "--save-raw", type=Path, default=None, help="сохранить сырой JSON-ответ в этот файл"
    )
    smoke.set_defaults(handler=cmd_bingx_smoke)

    probe = sub.add_parser(
        "bingx-probe-klines",
        parents=[common],
        help="пробные публичные запросы свечей: границы startTime/endTime, порядок, усечение",
    )
    probe.add_argument("--symbol", default=None, help="по умолчанию первая пара из symbols.yaml")
    probe.add_argument("--base-url", default=LIVE_BASE_URL, help=f"по умолчанию {LIVE_BASE_URL}")
    probe.add_argument(
        "--anchor",
        default=DEFAULT_ANCHOR.strftime("%Y-%m-%dT%H:%M"),
        help="граница свечи в прошлом (UTC, ГГГГ-ММ-ДДTЧЧ:ММ), от которой строятся окна",
    )
    probe.add_argument(
        "--volume-bar",
        default=DEFAULT_VOLUME_BAR.strftime("%Y-%m-%dT%H:%M"),
        help="начало 15м свечи (UTC) для сверки volume с TradingView",
    )
    probe.add_argument(
        "--set",
        dest="probe_set",
        choices=("basic", "paging", "depth", "coverage"),
        default="basic",
        help="basic: границы и порядок; paging: предел ответа, endTime, 1m; depth: начало "
        "истории 15m, 1m, 1h, 4h, 1d (бисекция); coverage: карта наличия данных по месяцам",
    )
    probe.set_defaults(handler=cmd_bingx_probe_klines)

    auth = sub.add_parser(
        "bingx-auth-check",
        parents=[common],
        help="один подписанный запрос только на чтение (баланс) в окружении BINGX_ENV",
    )
    auth.add_argument(
        "--show-balance",
        action="store_true",
        help="показать суммы и для боевого окружения (для vst суммы показываются всегда)",
    )
    auth.set_defaults(handler=cmd_bingx_auth_check)

    return parser


def _load_config(config_dir: Path | None) -> ProjectConfig:
    return load_project_config(resolve_config_dir(config_dir))


def cmd_config_check(args: argparse.Namespace) -> int:
    try:
        cfg = _load_config(args.config_dir)
        secrets = load_secrets(args.env_file)
    except ConfigError as exc:
        print(f"Конфиг: ОШИБКА\n{exc}", file=sys.stderr)
        return EXIT_CONFIG

    print("Конфиг: OK")
    print(f"Каталог: {resolve_config_dir(args.config_dir)}")
    print(f"Режим: {cfg.app.mode.value}")
    print(f"Хэш стратегии (strategy_hash, sha256): {strategy_hash(cfg)}")
    print(f"Хэш приложения (app_hash, sha256, только диагностика): {app_hash(cfg)}")
    data_dir = resolve_data_dir(cfg.app)
    source_names = {
        "env": f"переменная {DATA_DIR_ENV}",
        "app.yaml": "paths.data_dir в app.yaml",
        "default": "по умолчанию ./data",
    }
    print(f"Каталог данных: {data_dir.path} (источник: {source_names[data_dir.source]})")
    print(f"Рекордер: пары {', '.join(cfg.app.recorder.symbols)}")
    print(f"Пресеты: {', '.join(cfg.presets.presets)}")
    print(
        "Пары: " + ", ".join(f"{s.symbol} {s.tf} (с {s.data_start})" for s in cfg.symbols.symbols)
    )
    open_items = cfg.unresolved_decisions()
    print(f"Открытые решения (null): {len(open_items)}")
    for item in open_items:
        print(f"  - {item}")

    env_path = Path(args.env_file)
    print(f"Файл секретов {env_path}: {'найден' if env_path.is_file() else 'не найден'}")
    print(f"BINGX_ENV: {secrets.bingx_env}")
    for name, present in secrets.presence().items():
        print(f"  {name}: {'задан' if present else 'не задан'}")
    return EXIT_OK


def cmd_notify_test(args: argparse.Namespace) -> int:
    try:
        cfg = _load_config(args.config_dir)
        secrets = load_telegram_secrets(args.env_file)
    except ConfigError as exc:
        print(f"Конфиг: ОШИБКА\n{exc}", file=sys.stderr)
        return EXIT_CONFIG
    configure_logging(cfg.app.log_level, secrets.secret_values())

    if secrets.telegram_bot_token is None or not secrets.telegram_chat_ids:
        print(
            "TELEGRAM_BOT_TOKEN и TELEGRAM_CHAT_IDS должны быть заданы в .env "
            "(инструкция: docs/runbook.md, раздел Telegram)",
            file=sys.stderr,
        )
        return EXIT_CONFIG

    text = build_test_message(cfg.app.mode.value, strategy_hash(cfg), app_hash(cfg))
    if args.dry_run:
        print("Сухой прогон, ничего не отправлено. Текст сообщения:\n")
        print(text)
        print(f"\nАдресаты (белый список): {', '.join(map(str, secrets.telegram_chat_ids))}")
        print(f"Прокси Telegram: {'задан' if secrets.telegram_proxy_url else 'не задан'}")
        return EXIT_OK

    failed = False
    with TelegramClient(
        secrets.telegram_bot_token,
        secrets.telegram_chat_ids,
        proxy=secrets.telegram_proxy_url,
    ) as client:
        for chat_id in secrets.telegram_chat_ids:
            try:
                client.send_message(chat_id, text)
            except TelegramError as exc:
                failed = True
                print(f"chat_id {chat_id}: ОШИБКА: {exc}", file=sys.stderr)
            else:
                print(f"chat_id {chat_id}: отправлено")
    return EXIT_FAILURE if failed else EXIT_OK


def cmd_bingx_smoke(args: argparse.Namespace) -> int:
    try:
        cfg = _load_config(args.config_dir)
    except ConfigError as exc:
        print(f"Конфиг: ОШИБКА\n{exc}", file=sys.stderr)
        return EXIT_CONFIG
    first = cfg.symbols.symbols[0]
    symbol = args.symbol or first.symbol
    interval = args.interval or first.tf
    try:
        tf_ms = tf_to_ms(interval)
    except ValueError as exc:
        print(exc, file=sys.stderr)
        return EXIT_CONFIG
    configure_logging(cfg.app.log_level)

    try:
        with BingXPublicClient(args.base_url) as client:
            result = client.fetch_klines(symbol, interval, args.limit)
    except ValueError as exc:
        print(exc, file=sys.stderr)
        return EXIT_CONFIG
    except BingXError as exc:
        print(f"ОШИБКА: {exc}", file=sys.stderr)
        return EXIT_FAILURE

    candles = result.candles
    now_ms = int(time.time() * 1000)
    print(f"Запрос: GET {args.base_url}/openApi/swap/v3/quote/klines")
    print(f"Символ: {symbol}, интервал: {interval}, limit: {args.limit}")
    print(f"Получено свечей: {len(candles)}")
    if not candles:
        print("Свечей нет: проверь символ и интервал.")
        return EXIT_FAILURE
    order_names = {
        "single": "одна свеча",
        "asc": "по возрастанию ts_open (старые первыми)",
        "desc": "по убыванию ts_open (новые первыми)",
        "mixed": "смешанный",
    }
    print(f"Порядок в ответе: {order_names[detect_order(candles)]}")
    print(f"Первая в ответе:  {candles[0].ts_open} ({format_ts(candles[0].ts_open)})")
    print(f"Последняя в ответе: {candles[-1].ts_open} ({format_ts(candles[-1].ts_open)})")
    newest = max(candles, key=lambda candle: candle.ts_open)
    closed = is_candle_closed(newest, tf_ms, now_ms)
    print(
        f"Самая новая свеча: {newest.ts_open} ({format_ts(newest.ts_open)}), "
        f"закрыта: {'да' if closed else 'НЕТ, ещё формируется (отбрасывать)'}"
    )
    print(f"Без пропусков и дублей: {'да' if is_contiguous(candles, tf_ms) else 'НЕТ'}")
    print(f"Локальное время сейчас: {format_ts(now_ms)} (сверка по часам этой машины)")
    if result.limit_headers:
        print("Заголовки лимитов:")
        for name, value in sorted(result.limit_headers.items()):
            print(f"  {name}: {value}")
    else:
        print("Заголовки лимитов: в ответе не обнаружено")
    if args.save_raw is not None:
        args.save_raw.parent.mkdir(parents=True, exist_ok=True)
        # Завершающий перевод строки нужен, чтобы хук end-of-file-fixer не менял файл при коммите.
        raw_text = result.raw_text if result.raw_text.endswith("\n") else result.raw_text + "\n"
        args.save_raw.write_text(raw_text, encoding="utf-8")
        print(f"Сырой ответ сохранён: {args.save_raw}")
    return EXIT_OK


def _parse_utc_minute(text: str) -> int:
    """'2026-11-01T00:00' (UTC) -> миллисекунды."""
    moment = datetime.fromisoformat(text)
    if moment.tzinfo is None:
        moment = moment.replace(tzinfo=UTC)
    return to_ms(moment.astimezone(UTC))


def cmd_bingx_probe_klines(args: argparse.Namespace) -> int:
    try:
        cfg = _load_config(args.config_dir)
        anchor_ms = _parse_utc_minute(args.anchor)
        volume_bar_ms = _parse_utc_minute(args.volume_bar)
    except (ConfigError, ValueError) as exc:
        print(f"ОШИБКА: {exc}", file=sys.stderr)
        return EXIT_CONFIG
    symbol = args.symbol or cfg.symbols.symbols[0].symbol
    configure_logging(cfg.app.log_level)
    now_ms = int(time.time() * 1000)
    if anchor_ms % 900_000 or volume_bar_ms % 900_000:
        print("ОШИБКА: --anchor и --volume-bar должны лежать на границе 15 минут", file=sys.stderr)
        return EXIT_CONFIG
    needed_bars = {"paging": 1800, "depth": 96, "coverage": 96}.get(args.probe_set, 1600)
    if anchor_ms + needed_bars * 900_000 > now_ms:
        print(
            "ОШИБКА: --anchor слишком близко к текущему времени "
            f"(нужно >= {needed_bars * 15 // 60 // 24 + 1} суток назад)",
            file=sys.stderr,
        )
        return EXIT_CONFIG

    if args.probe_set == "depth":
        return _probe_depth(args, symbol, anchor_ms, now_ms)
    if args.probe_set == "coverage":
        return _probe_coverage(args, symbol, now_ms)

    cases = (
        build_paging_cases(anchor_ms)
        if args.probe_set == "paging"
        else build_cases(anchor_ms, volume_bar_ms)
    )
    print(f"Пробные запросы klines: {args.base_url}, символ {symbol}, запросов: {len(cases)}")
    print(f"Опорная граница: {format_ts(anchor_ms)}; свеча для volume: {format_ts(volume_bar_ms)}")
    print(f"Локальное время сейчас: {format_ts(now_ms)} (по часам этой машины)\n")
    with BingXPublicClient(args.base_url) as client:
        results = run_probe(client, symbol, cases, sleep=time.sleep)
    for line in render_results(results):
        print(line)
    print("\nВыводы по этим запросам:")
    findings = (
        derive_paging_findings(results, anchor_ms, now_ms)
        if args.probe_set == "paging"
        else derive_findings(results, anchor_ms, now_ms, volume_bar_ms)
    )
    for line in findings:
        print(f"  - {line}")
    failed = [r for r in results if r.candles is None]
    return EXIT_FAILURE if failed and len(failed) == len(results) else EXIT_OK


def _stored_targets(
    store: CandleStore, symbols: list[str] | None, tfs: list[str] | None
) -> list[tuple[str, str]]:
    """Пары (символ, ТФ) для команд проверки: заданные явно или всё, что есть в хранилище."""
    targets: list[tuple[str, str]] = []
    for symbol in symbols or store.symbols():
        for tf in tfs or store.timeframes(symbol):
            targets.append((symbol, tf))
    return targets


def _parse_cross(
    items: list[str] | None, available: dict[str, list[str]]
) -> list[tuple[str, str, str]]:
    """Пары кросс-проверки `(символ, крупный, мелкий)`: заданные или 15m:1m там, где есть оба."""
    if items:
        pairs = []
        for item in items:
            coarse, _, fine = item.partition(":")
            if not coarse or not fine:
                raise ValueError(f"--cross {item!r}: ожидается вид КРУПНЫЙ:МЕЛКИЙ, например 1h:15m")
            pairs.append((coarse, fine))
        return [(symbol, c, f) for symbol in available for c, f in pairs]
    return [
        (symbol, "15m", "1m") for symbol, tfs in available.items() if "15m" in tfs and "1m" in tfs
    ]


def _write_report(report_dir: Path, name: str, text: str) -> Path:
    path = report_dir / name
    report_dir.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)
    return path


def cmd_data_check(args: argparse.Namespace) -> int:
    try:
        cfg = _load_config(args.config_dir)
        start_ms = _parse_utc_minute(args.start) if args.start else None
        end_ms = _parse_utc_minute(args.end) if args.end else None
    except (ConfigError, ValueError) as exc:
        print(f"ОШИБКА: {exc}", file=sys.stderr)
        return EXIT_CONFIG
    data_dir = resolve_data_dir(cfg.app)
    store = CandleStore(data_dir.path)
    derived = CandleStore(data_dir.path, layer=RESAMPLED_LAYER)
    try:
        native_targets = (
            [] if args.layer == "resampled" else _stored_targets(store, args.symbol, args.tf)
        )
        if args.layer == "resampled":
            derived_targets = _stored_targets(derived, args.symbol, args.tf)
        elif args.layer == "all":
            # рядом с нативными проверяем только те ряды ресемпла, которые есть
            derived_targets = [
                (symbol, tf)
                for symbol, tf in _stored_targets(derived, args.symbol, args.tf)
                if tf in derived.timeframes(symbol)
            ]
        else:
            derived_targets = []
        available: dict[str, list[str]] = {}
        for symbol, tf in native_targets:
            available.setdefault(symbol, []).append(tf)
        cross_pairs = [] if args.no_cross else _parse_cross(args.cross, available)
        for _symbol, coarse, fine in cross_pairs:
            if tf_to_ms(coarse) <= tf_to_ms(fine) or tf_to_ms(coarse) % tf_to_ms(fine):
                raise ValueError(f"--cross {coarse}:{fine}: крупный ТФ должен делиться на мелкий")
    except ValueError as exc:
        print(f"ОШИБКА: {exc}", file=sys.stderr)
        return EXIT_CONFIG
    if not native_targets and not derived_targets:
        print(
            f"ОШИБКА: в {data_dir.path} нет сохранённых свечей (сначала `data download`)",
            file=sys.stderr,
        )
        return EXIT_CONFIG
    configure_logging(cfg.app.log_level)

    now_ms = int(time.time() * 1000)
    print(f"Каталог данных: {data_dir.path} (источник: {data_dir.source})\n")
    series = []
    crosses = []
    try:
        for layer_store, targets in ((store, native_targets), (derived, derived_targets)):
            for symbol, tf in targets:
                report = check_series(
                    layer_store, symbol, tf, now_ms=now_ms, start_ms=start_ms, end_ms=end_ms
                )
                series.append(report)
                print(series_summary_line(report))
        for symbol, coarse, fine in cross_pairs:
            cross = cross_check(store, symbol, coarse, fine, start_ms=start_ms, end_ms=end_ms)
            crosses.append(cross)
            print(cross_summary_line(cross))
    except StoreError as exc:
        print(f"ОШИБКА: {exc}", file=sys.stderr)
        return EXIT_FAILURE

    report_dir = Path(args.report_dir) if args.report_dir else data_dir.path / "reports"
    stamp = datetime.fromtimestamp(now_ms / 1000, tz=UTC).strftime("%Y%m%d_%H%M%S")
    text = render_markdown(
        series, crosses, generated_ms=now_ms, data_dir=data_dir.path, version=__version__
    )
    try:
        path = _write_report(report_dir, f"quality_{stamp}.md", text)
    except OSError as exc:
        print(f"ОШИБКА: не удалось записать отчёт: {exc}", file=sys.stderr)
        return EXIT_FAILURE

    problems = sum(len(s.errors) for s in series) + sum(1 for c in crosses if c.severity == "error")
    warnings = sum(len(s.warnings) for s in series) + sum(
        1 for c in crosses if c.severity == "warning"
    )
    print(f"\nОтчёт: {path}")
    if problems:
        print(
            f"ИТОГ: найдено проблем: {problems}"
            + (f"; предупреждений: {warnings}" if warnings else "")
        )
        return EXIT_FAILURE
    print("ИТОГ: ошибок нет" + (f"; предупреждений: {warnings}" if warnings else ""))
    return EXIT_OK


def cmd_data_sync(args: argparse.Namespace) -> int:
    try:
        cfg = _load_config(args.config_dir)
        start_ms = _parse_utc_minute(args.start) if args.start else None
        end_ms = _parse_utc_minute(args.end) if args.end else None
    except (ConfigError, ValueError) as exc:
        print(f"ОШИБКА: {exc}", file=sys.stderr)
        return EXIT_CONFIG
    data_dir = resolve_data_dir(cfg.app)
    store = CandleStore(data_dir.path)
    symbols: list[str] = args.symbol or cfg.app.downloader.symbols
    tfs: list[str] = args.tf or cfg.app.downloader.timeframes
    targets = [(symbol, tf) for symbol in symbols for tf in tfs]
    try:
        for symbol, tf in targets:
            store.tf_dir(symbol, tf)
    except ValueError as exc:
        print(f"ОШИБКА: {exc}", file=sys.stderr)
        return EXIT_CONFIG
    configure_logging(cfg.app.log_level)

    now_ms = int(time.time() * 1000)
    print(f"Каталог данных: {data_dir.path} (источник: {data_dir.source})")
    print(f"Пары: {', '.join(symbols)}; таймфреймы: {', '.join(tfs)}\n")
    exit_code = EXIT_OK
    remaining = 0
    with BingXPublicClient(args.base_url) as client:
        limiter = RateLimiter()
        for symbol, tf in targets:
            tally = RetryTally()
            fetcher = RetryingClient(client, time.sleep, tally)
            try:
                stored = store.timestamps(symbol, tf)
                if not stored:
                    print(
                        f"{symbol} {tf}: ВНИМАНИЕ: нет сохранённых свечей (сначала `data download`)"
                    )
                    remaining += 1
                    continue
                first = start_ms if start_ms is not None else min(stored)
                report = sync_gaps(
                    fetcher,
                    store,
                    symbol,
                    tf,
                    first,
                    end_ms if end_ms is not None else now_ms,
                    now_ms=now_ms,
                    limiter=limiter,
                    sleep=time.sleep,
                    progress=print,
                )
            except (BingXError, StoreError) as exc:
                print(
                    f"{symbol} {tf}: ОШИБКА: {exc}\n"
                    "  накопленное сохранено; повторный запуск продолжит",
                    file=sys.stderr,
                )
                exit_code = EXIT_FAILURE
                continue
            report.retries = tally.retries
            report.last_retry_error = tally.last_retry_error
            remaining += len(report.pending)
            for line in render_sync(report):
                print(line)
    if exit_code != EXIT_OK:
        return exit_code
    if remaining:
        print(
            f"\nОСТАЛИСЬ ПРОПУСКИ: {remaining} (неподтверждённых слотов и пар без данных); "
            f"код выхода {EXIT_GAPS}"
        )
        return EXIT_GAPS
    print("\nПропусков не осталось")
    return EXIT_OK


def cmd_data_resample(args: argparse.Namespace) -> int:
    try:
        cfg = _load_config(args.config_dir)
        start_ms = _parse_utc_minute(args.start) if args.start else None
        end_ms = _parse_utc_minute(args.end) if args.end else None
        if tf_to_ms(args.coarse_tf) <= tf_to_ms(args.fine_tf) or tf_to_ms(
            args.coarse_tf
        ) % tf_to_ms(args.fine_tf):
            raise ValueError(f"{args.coarse_tf} должен делиться нацело на {args.fine_tf}")
    except (ConfigError, ValueError) as exc:
        print(f"ОШИБКА: {exc}", file=sys.stderr)
        return EXIT_CONFIG
    data_dir = resolve_data_dir(cfg.app)
    native = CandleStore(data_dir.path)
    derived = CandleStore(data_dir.path, layer=RESAMPLED_LAYER)
    symbols: list[str] = args.symbol or cfg.app.downloader.symbols
    try:
        for symbol in symbols:
            native.tf_dir(symbol, args.fine_tf)
            native.tf_dir(symbol, args.coarse_tf)
    except ValueError as exc:
        print(f"ОШИБКА: {exc}", file=sys.stderr)
        return EXIT_CONFIG
    configure_logging(cfg.app.log_level)

    now_ms = int(time.time() * 1000)
    print(f"Каталог данных: {data_dir.path} (источник: {data_dir.source})\n")
    reports = []
    comparisons = []
    exit_code = EXIT_OK
    try:
        for symbol in symbols:
            report = resample_range(
                native,
                derived,
                symbol,
                args.fine_tf,
                args.coarse_tf,
                now_ms=now_ms,
                start_ms=start_ms,
                end_ms=end_ms,
            )
            reports.append(report)
            for line in render_resample(report):
                print(line)
            if args.no_compare:
                continue
            comparison = compare_layers(
                native, derived, symbol, args.coarse_tf, start_ms=start_ms, end_ms=end_ms
            )
            comparisons.append(comparison)
            for line in render_comparison(comparison):
                print(line)
            if comparison.verdict == "not_acceptable":
                exit_code = EXIT_FAILURE
    except StoreError as exc:
        print(f"ОШИБКА: {exc}", file=sys.stderr)
        return EXIT_FAILURE

    if not any(r.buckets for r in reports):
        print(
            f"ОШИБКА: нет сохранённых {args.fine_tf} для ресемпла (сначала `data download`)",
            file=sys.stderr,
        )
        return EXIT_FAILURE
    report_dir = Path(args.report_dir) if args.report_dir else data_dir.path / "reports"
    stamp = datetime.fromtimestamp(now_ms / 1000, tz=UTC).strftime("%Y%m%d_%H%M%S")
    text = render_resample_markdown(reports, comparisons, generated_ms=now_ms, version=__version__)
    try:
        path = _write_report(report_dir, f"resample_{stamp}.md", text)
    except OSError as exc:
        print(f"ОШИБКА: не удалось записать отчёт: {exc}", file=sys.stderr)
        return EXIT_FAILURE
    print(f"\nОтчёт: {path}")
    return exit_code


def _probe_depth(args: argparse.Namespace, symbol: str, high_ms: int, now_ms: int) -> int:
    print(f"Поиск начала истории: {args.base_url}, символ {symbol}")
    print(f"Верхняя граница поиска (свечи там есть): {format_ts(high_ms)}")
    print(f"Локальное время сейчас: {format_ts(now_ms)} (по часам этой машины)\n")
    with BingXPublicClient(args.base_url) as client:
        results = run_depth_probe(client, symbol, high_ms, sleep=time.sleep)
    for line in render_depth(results):
        print(line)
    print("\nВыводы по этим запросам:")
    for line in derive_depth_findings(results, now_ms):
        print(f"  - {line}")
    failed = [r for r in results if r.history is None]
    return EXIT_FAILURE if len(failed) == len(results) else EXIT_OK


def _probe_coverage(args: argparse.Namespace, symbol: str, now_ms: int) -> int:
    total = len(coverage_samples(now_ms))
    print(f"Карта наличия данных: {args.base_url}, символ {symbol}, запросов: {total}")
    print(f"Локальное время сейчас: {format_ts(now_ms)} (по часам этой машины)")
    print("Идёт опрос (около двух минут), суточные окна по дням месяца...\n")
    with BingXPublicClient(args.base_url) as client:
        cells = run_coverage_probe(client, symbol, now_ms, sleep=time.sleep)
    for line in render_coverage(cells):
        print(line)
    print("\nВыводы по этим запросам:")
    for line in derive_coverage_findings(cells, now_ms):
        print(f"  - {line}")
    failed = [c for c in cells if c.got is None]
    return EXIT_FAILURE if cells and len(failed) == len(cells) else EXIT_OK


def cmd_data_download(args: argparse.Namespace) -> int:
    try:
        cfg = _load_config(args.config_dir)
        start_ms = _parse_utc_minute(args.start)
        end_ms = _parse_utc_minute(args.end) if args.end else None
    except (ConfigError, ValueError) as exc:
        print(f"ОШИБКА: {exc}", file=sys.stderr)
        return EXIT_CONFIG
    symbols: list[str] = args.symbol or [cfg.symbols.symbols[0].symbol]
    tfs: list[str] = args.tf or ["15m", "1m"]
    window_bars: int = args.window_bars
    data_dir = resolve_data_dir(cfg.app)
    store = CandleStore(data_dir.path)
    try:
        for tf in tfs:
            if tf not in VALID_INTERVALS:
                raise ValueError(f"интервал {tf!r} не из списка {sorted(VALID_INTERVALS)}")
            for symbol in symbols:
                store.tf_dir(symbol, tf)  # проверка вида пары и таймфрейма
        if not 1 <= window_bars <= MAX_KLINES_LIMIT:
            raise ValueError(f"--window-bars: ожидается число от 1 до {MAX_KLINES_LIMIT}")
    except ValueError as exc:
        print(f"ОШИБКА: {exc}", file=sys.stderr)
        return EXIT_CONFIG
    configure_logging(cfg.app.log_level)

    now_ms = int(time.time() * 1000)
    end_text = format_ts(end_ms) if end_ms is not None else "последняя закрытая свеча"
    print(f"Каталог данных: {data_dir.path} (источник: {data_dir.source})")
    print(f"Пары: {', '.join(symbols)}; таймфреймы: {', '.join(tfs)}")
    print(f"Период: с {format_ts(start_ms)} до {end_text}; окно {window_bars} баров\n")

    exit_code = EXIT_OK
    gaps = 0
    confirmed_total = 0
    with BingXPublicClient(args.base_url) as client:
        limiter = RateLimiter()
        for symbol in symbols:
            for tf in tfs:
                try:
                    report = download_tf(
                        client,
                        store,
                        symbol,
                        tf,
                        start_ms,
                        end_ms if end_ms is not None else now_ms,
                        now_ms=now_ms,
                        limiter=limiter,
                        progress=print,
                        window_bars=window_bars,
                        sleep=time.sleep,
                    )
                except (BingXError, StoreError) as exc:
                    print(
                        f"{symbol} {tf}: ОШИБКА: {exc}\n"
                        "  накопленное сохранено; повторный запуск продолжит с этого места",
                        file=sys.stderr,
                    )
                    exit_code = EXIT_FAILURE
                    continue
                for line in render_report(report):
                    print(line)
                gaps += len(report.missing)
                confirmed_total += len(report.confirmed_missing)
    if gaps:
        print(
            f"\nПРЕДУПРЕЖДЕНИЕ: пропусков всего {gaps}. Повторный запуск попробует их докачать; "
            "пометку «подтверждённо отсутствует» добавит проверка качества (P01-T03)."
        )
    if confirmed_total:
        print(
            f"\nВНИМАНИЕ: подтверждённо отсутствуют у биржи свечей всего: {confirmed_total} "
            "(их больше не запрашивают; полный список: `data check`)"
        )
    return exit_code


def _mask_key(key: str) -> str:
    return f"{key[:4]}…{key[-4:]}" if len(key) >= 12 else "***"


_AUTH_HINTS = (
    "Что проверить:\n"
    "  1. BINGX_API_KEY и BINGX_API_SECRET в .env без кавычек и пробелов;\n"
    "  2. BINGX_ENV совпадает с окружением, где создан ключ (vst или live);\n"
    "  3. если включена привязка к IP, текущий внешний IP этой машины есть в списке;\n"
    "  4. время на компьютере синхронизировано (окно подписи 5000 мс);\n"
    "  5. у ключа включено чтение."
)


def cmd_bingx_auth_check(args: argparse.Namespace) -> int:
    try:
        cfg = _load_config(args.config_dir)
        secrets = load_bingx_secrets(args.env_file)
    except ConfigError as exc:
        print(f"Конфиг: ОШИБКА\n{exc}", file=sys.stderr)
        return EXIT_CONFIG
    configure_logging(cfg.app.log_level, secrets.secret_values())

    if secrets.bingx_api_key is None or secrets.bingx_api_secret is None:
        print(
            "BINGX_API_KEY и BINGX_API_SECRET должны быть заданы в .env "
            "(инструкция: docs/runbook.md, раздел BingX)",
            file=sys.stderr,
        )
        return EXIT_CONFIG

    base_url = VST_BASE_URL if secrets.bingx_env == "vst" else LIVE_BASE_URL
    print(f"Окружение: {secrets.bingx_env} ({base_url})")
    print(f"Ключ: {_mask_key(secrets.bingx_api_key.get_secret_value())}")
    if secrets.bingx_env == "live":
        print("Боевой контур: выполняются только запросы на чтение.")

    show_amounts = secrets.bingx_env == "vst" or args.show_balance
    with BingXReadOnlyClient(secrets.bingx_api_key, secrets.bingx_api_secret, base_url) as client:
        try:
            balance = client.fetch_balance()
        except BingXError as exc:
            print(f"Подписанный запрос: ОШИБКА: {exc}", file=sys.stderr)
            print(_AUTH_HINTS, file=sys.stderr)
            return EXIT_FAILURE
        print("Подписанный запрос (баланс): OK, подпись принята")
        print(f"Активов в ответе: {len(balance.assets)} ({', '.join(balance.assets)})")
        if balance.fields and show_amounts:
            details = ", ".join(f"{k}={v}" for k, v in balance.fields.items())
            print(f"Актив {balance.shown_asset}: {details}")
        elif balance.fields:
            print("Суммы скрыты (добавь --show-balance, чтобы показать).")

        try:
            payload = client.fetch_api_permissions()
            permissions = extract_permissions(payload)
        except BingXError as exc:
            print(f"Права ключа: проверить не удалось ({exc}); эндпоинт не подтверждён.")
        else:
            fields = permission_fields(permissions)
            print(f"Права ключа (поля из ответа): {fields or 'полей не найдено'}")
            if "ipAddresses" in fields:
                ips = fields["ipAddresses"]
                count = len(ips) if isinstance(ips, list) else ips
                print(f"ipAddresses: в списке {count} адресов")
            print(f"Форма ответа (только имена и типы): {describe_shape(payload)}")
            print(
                "Примечание: значения кодов permissions не сверены с документацией, команда "
                "выводов о правах ключа не делает. Проверь права вручную в API Management."
            )
    return EXIT_OK


def main(argv: Sequence[str] | None = None) -> int:
    # На Windows консоль может быть не в UTF-8: сообщения на русском не должны падать.
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if callable(reconfigure):
            reconfigure(encoding="utf-8", errors="replace")
    parser = build_parser()
    args = parser.parse_args(argv)
    handler = getattr(args, "handler", None)
    if handler is None:
        parser.print_help()
        return EXIT_OK
    result: int = handler(args)
    return result


if __name__ == "__main__":
    sys.exit(main())
