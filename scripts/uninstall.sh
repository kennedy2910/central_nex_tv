#!/usr/bin/env bash
set -euo pipefail
echo "=== Central-Nex v7 UNINSTALL (zera tudo) ==="

if ! command -v docker >/dev/null 2>&1; then
  echo "[ERRO] Docker não encontrado."
  exit 1
fi

if docker compose version >/dev/null 2>&1; then
  COMPOSE="docker compose"
elif command -v docker-compose >/dev/null 2>&1; then
  COMPOSE="docker-compose"
else
  echo "[ERRO] Compose não encontrado."
  exit 1
fi

if ! docker ps >/dev/null 2>&1; then
  SUDO="sudo"
else
  SUDO=""
fi

echo "[INFO] Derrubando containers..."
$SUDO $COMPOSE down -v --remove-orphans || true

echo "[INFO] Removendo data/ (DB)..."
rm -rf ./data || true

echo "[OK] Removido."
