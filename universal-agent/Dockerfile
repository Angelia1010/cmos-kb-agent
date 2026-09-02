# KB-Agent 服务镜像 — 完全离线构建(不访问 PyPI)
# 注意: 所有 .whl 均为 x86_64 manylinux 轮子, 在 ARM 机器(macOS M 系列)上构建需加
#       --platform=linux/amd64
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# ── 1) 依赖安装: 离线轮子 + 锁定版本, 不连 PyPI ──────────────────────────
COPY offline-packages/ /offline-packages/
COPY requirements.txt .
RUN pip install --no-index --find-links=/offline-packages -r requirements.txt

# ── 2) 业务代码 ───────────────────────────────────────────────────────────
COPY src/ ./src/
COPY services/ ./services/
COPY skills/ ./skills/
COPY main.py .

# ── 3) 配置与环境变量文件 ─────────────────────────────────────────────────
# config: 由模板生成(离线 ScriptedChatModel)。
#   若要接真灵犀网关, 需换成流水线注入的真实 config.yaml(见下方提醒)
COPY config.example.yaml ./config.yaml

# profile.env.sh: QWEN_API_KEY 等环境变量。在 .gitignore 中,
#   需流水线在 docker build 前注入到构建上下文, 否则下一行 COPY 会失败
COPY deploy/profile.env.sh /app/profile.env.sh
RUN chmod 600 /app/profile.env.sh

ENV PYTHONPATH=/app/src:/app/services \
    UNIAGENT_CONFIG=/app/config.yaml

#USER appuser

EXPOSE 8080

# 先 source profile.env.sh 把 QWEN_API_KEY 等 export 进环境, 再启动服务。
# 必须用 sh -c 包一层: JSON exec 形式的 CMD 不经过 shell, 无法 source 文件;
# 内层 exec 让 uvicorn 替换 shell 进程, docker stop 的 SIGTERM 可直达。
CMD ["/bin/sh", "-c", ". /app/profile.env.sh && exec python -m uvicorn kbagent_service.app:app --host 0.0.0.0 --port 8080"]
