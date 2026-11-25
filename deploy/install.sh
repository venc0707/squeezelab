#!/usr/bin/env bash
# Установка таймеров squeezelab-sync и squeezelab-backup. Идемпотентна: повторный запуск ничего
# не меняет, если шаблоны и параметры те же.
#
#   sudo deploy/install.sh [--dry-run]
#
# Параметры (переменные окружения, по умолчанию как на сервере проекта):
#   SQUEEZELAB_DATA_DIR    каталог данных          /srv/data/squeezelab
#   SQUEEZELAB_BACKUP_DIR  каталог копии           /var/backups/squeezelab
#   SQUEEZELAB_USER        пользователь юнитов     $SUDO_USER (не root)
# Для тестов: ALLOW_NON_ROOT=1, SKIP_SYSTEMCTL=1, SYSTEMD_DIR=..., RSYNC_BIN=...
set -euo pipefail

DRY_RUN=0
for arg in "$@"; do
    case "$arg" in
        --dry-run) DRY_RUN=1 ;;
        -h | --help)
            sed -n '2,13p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
            exit 0
            ;;
        *)
            echo "Неизвестный аргумент: $arg" >&2
            exit 2
            ;;
    esac
done

WORKDIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_DIR="${SQUEEZELAB_DATA_DIR:-/srv/data/squeezelab}"
BACKUP_DIR="${SQUEEZELAB_BACKUP_DIR:-/var/backups/squeezelab}"
SYSTEMD_DIR="${SYSTEMD_DIR:-/etc/systemd/system}"
RUN_USER="${SQUEEZELAB_USER:-${SUDO_USER:-}}"
RSYNC="${RSYNC_BIN:-$(command -v rsync || true)}"
UNITS=(squeezelab-sync.service squeezelab-sync.timer squeezelab-backup.service squeezelab-backup.timer)

fail() {
    echo "ОШИБКА: $*" >&2
    exit 1
}

if [[ "${ALLOW_NON_ROOT:-0}" != "1" && "$(id -u)" -ne 0 ]]; then
    fail "запустите через sudo: sudo deploy/install.sh"
fi
[[ -n "$RUN_USER" ]] || fail "не определён пользователь юнитов (задайте SQUEEZELAB_USER)"
[[ "$RUN_USER" != "root" ]] || fail "юниты не должны работать от root"
id "$RUN_USER" > /dev/null 2>&1 || fail "пользователь $RUN_USER не существует"
[[ -x "$WORKDIR/.venv/bin/squeezelab" ]] || fail "нет $WORKDIR/.venv/bin/squeezelab: сначала выполните uv sync"
[[ -n "$RSYNC" && -x "$RSYNC" ]] || fail "не найден rsync (sudo apt install rsync)"

echo "Репозиторий:     $WORKDIR"
echo "Каталог данных:  $DATA_DIR"
echo "Каталог копии:   $BACKUP_DIR"
echo "Пользователь:    $RUN_USER"
echo "Каталог юнитов:  $SYSTEMD_DIR"

render() { # шаблон -> stdout
    sed -e "s|@USER@|$RUN_USER|g" \
        -e "s|@WORKDIR@|$WORKDIR|g" \
        -e "s|@DATA_DIR@|$DATA_DIR|g" \
        -e "s|@BACKUP_DIR@|$BACKUP_DIR|g" \
        -e "s|@RSYNC@|$RSYNC|g" \
        "$1"
}

changed=0
for unit in "${UNITS[@]}"; do
    template="$WORKDIR/deploy/$unit"
    target="$SYSTEMD_DIR/$unit"
    [[ -f "$template" ]] || fail "нет шаблона $template"
    rendered="$(mktemp)"
    render "$template" > "$rendered"
    if grep -q '@[A-Z_]*@' "$rendered"; then
        rm -f "$rendered"
        fail "в $unit остались неподставленные параметры"
    fi
    if [[ -f "$target" ]] && cmp -s "$rendered" "$target"; then
        echo "без изменений: $unit"
        rm -f "$rendered"
        continue
    fi
    changed=1
    if [[ "$DRY_RUN" == "1" ]]; then
        echo "[dry-run] записал бы $target"
        rm -f "$rendered"
    else
        mkdir -p "$SYSTEMD_DIR"
        install -m 0644 "$rendered" "$target"
        rm -f "$rendered"
        echo "установлен: $target"
    fi
done

if [[ "$DRY_RUN" == "1" ]]; then
    echo "[dry-run] создал бы $BACKUP_DIR и включил бы таймеры squeezelab-sync и squeezelab-backup"
    exit 0
fi

# Каталог копии: владелец пользователь юнитов (копия не от root).
if [[ "$(id -u)" -eq 0 ]]; then
    install -d -o "$RUN_USER" -g "$(id -gn "$RUN_USER")" -m 0750 "$BACKUP_DIR"
else
    mkdir -p "$BACKUP_DIR"
fi

if [[ "${SKIP_SYSTEMCTL:-0}" == "1" ]]; then
    echo "systemctl пропущен (SKIP_SYSTEMCTL=1)"
else
    systemctl daemon-reload
    systemctl enable --now squeezelab-sync.timer squeezelab-backup.timer
    systemctl list-timers 'squeezelab-*' --no-pager || true
fi
if [[ "$changed" == "0" ]]; then
    echo "Всё уже установлено, изменений нет."
else
    echo "Готово."
fi
