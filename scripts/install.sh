#!/usr/bin/env bash
set -euo pipefail

echo "=== Central-Nex v7 INSTALL ==="

if ! command -v docker >/dev/null 2>&1; then
  echo "[ERRO] Docker não encontrado. Instale Docker Engine e rode novamente."
  exit 1
fi

if docker compose version >/dev/null 2>&1; then
  COMPOSE="docker compose"
elif command -v docker-compose >/dev/null 2>&1; then
  COMPOSE="docker-compose"
else
  echo "[ERRO] Compose não encontrado."
  echo "Ubuntu: sudo apt-get update && sudo apt-get install -y docker-compose"
  exit 1
fi

if ! docker ps >/dev/null 2>&1; then
  echo "[INFO] Sem permissão no Docker socket. Vou tentar com sudo..."
  SUDO="sudo"
else
  SUDO=""
fi

echo "[INFO] Build + up..."
$SUDO $COMPOSE up -d --build

echo
echo "[OK] Central-Nex rodando."
echo "Admin:        http://SEU_IP:9000/"
echo "Healthcheck:  http://SEU_IP:9000/health"
echo "Playlist:     http://SEU_IP:9000/iptv/edge.m3u?api_key=edge_xxx"
