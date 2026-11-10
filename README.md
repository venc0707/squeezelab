# SqueezeLab

Торговый бот для BingX USDT-M бессрочных фьючерсов. Главный сигнал — детектор поджимов на
Williams %R (период 500), таймфрейм 15 минут. Это инженерный проект, а не финансовая
рекомендация: торговля фьючерсами с плечом может привести к потере депозита.

Текущая стадия: **Phase 00** (подготовка и инфраструктура). Торговой логики в репозитории нет.

## Установка

Нужны Python 3.11+ и [uv](https://docs.astral.sh/uv/).

```bash
git clone <адрес репозитория> squeezelab
cd squeezelab
uv sync                          # создаёт .venv и ставит зависимости вместе с dev-группой
uv run pre-commit install        # хуки перед каждым коммитом
cp .env.example .env             # секреты только здесь, файл вне git
```

## Проверки

```bash
uv run pytest                    # тесты
uv run ruff check                # линтер
uv run ruff format --check       # формат
uv run mypy                      # типы
uv run pre-commit run --all-files
```

Тесты с пометкой `integration` (сеть, ключи) в обычный прогон не входят.

## Команды

```bash
uv run squeezelab --version
```

Остальные команды появляются по мере выполнения задач Phase 00 и описаны в `docs/runbook.md`.

## Структура

```
config/        конфиги (YAML)
src/squeezelab/
  data/ signals/ decision/ risk/ execution/ journal/ notify/ backtest/ research/ app/
tests/         unit/ golden/ replay/ integration/ fixtures/
scripts/ docker/ docs/
```

## Правила

- Решения принимаются только по закрытым свечам. Время везде UTC, в миллисекундах.
- Секреты только в `.env` или переменных окружения, не в git и не в логах.
- Принятые решения и открытые вопросы: `docs/decisions.md`.
