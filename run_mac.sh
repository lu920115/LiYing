#!/bin/bash
# 停掉占用 7860 端口的旧服务，然后用中文界面启动新服务
cd "$(dirname "$0")"

if [ ! -d .venv ]; then
  python3 -m venv .venv
  .venv/bin/pip install -r requirements.txt
fi

OLD_PIDS=$(lsof -nP -tiTCP:7860 -sTCP:LISTEN)
if [ -n "$OLD_PIDS" ]; then
  echo "停止旧服务: $OLD_PIDS"
  kill $OLD_PIDS
  sleep 1
fi

export NO_PROXY=127.0.0.1,localhost
cd src/webui && exec ../../.venv/bin/python app.py --lang zh
