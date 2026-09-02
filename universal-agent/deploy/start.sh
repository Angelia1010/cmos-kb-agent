#!/bin/bash
# ──────────────────────────────────────────────────────────────────────
# kbagent 服务容器启动脚本
# 镜像内路径: /app/start.sh
#
# Dockerfile 接入方式:
#   COPY deploy/start.sh /app/start.sh
#   RUN  chmod +x /app/start.sh
#   CMD  ["/app/start.sh"]
#
# 健康检查: GET /health (与业务端口相同)
# ──────────────────────────────────────────────────────────────────────
set -euo pipefail

APP_DIR="/app"
cd "$APP_DIR"

# 1) 加载环境变量 (QWEN_API_KEY 等, 见 profile.env.sh); 文件缺失时不阻断,
#    此时 config.yaml 中 ${QWEN_API_KEY} 无法展开, 服务会回退离线 ScriptedChatModel
if [ -f "$APP_DIR/profile.env.sh" ]; then
  set -a
  # shellcheck disable=SC1091
  . "$APP_DIR/profile.env.sh"
  set +a
fi

# 2) 启动服务。
#    - server.py 自包含: 自行把 /app/src 与 /app/services 加入 sys.path,
#      并按自身位置定位同目录的 config.yaml, 无需手工设置 PYTHONPATH
#    - exec 使 python 替换当前 shell, docker stop 的 SIGTERM 可直达 uvicorn
#    - 并发模型为 asyncio 单进程(模型实例全局共享); 如需多进程可在
#      server.py 中给 uvicorn.run 追加 workers=N, 但每个 worker 会各自
#      加载一份模型实例, 内存随之翻倍
#    - 监听端口可经环境变量 PORT 覆盖(默认 8000)
#    注意: config.yaml 属机密配置未入库, 需在构建时 COPY 进镜像或运行时挂载
exec python "$APP_DIR/server.py"
