# 单容器部署：前后端同镜像，符合魔搭创空间（必须监听 0.0.0.0:7860）
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HF_ENDPOINT=https://hf-mirror.com \
    OMP_NUM_THREADS=2

WORKDIR /app

# 系统依赖：onnxruntime 需要 libgomp，Pillow 需要少量图形库
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgomp1 \
        libgl1 \
        libglib2.0-0 \
        curl \
    && rm -rf /var/lib/apt/lists/*

# 先装依赖，利用层缓存
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# 代码
COPY backend ./backend
COPY frontend ./frontend
COPY scripts ./scripts

# 数据（索引与模型预生成，图标以 zip 形式携带，启动时解包）
COPY backend/data ./backend/data

# 构建期解包图标，避免每次启动都解压
RUN python -m backend.pipeline.bundle_assets --unpack

ENV PYTHONPATH=/app
EXPOSE 7860

HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
    CMD curl -fsS http://127.0.0.1:7860/health || exit 1

# 单 worker：SQLite 写入与模型加载都是进程级的，
# 多 worker 会各自加载 23MB 模型，在 CPU 上得不偿失
CMD ["python", "-m", "backend.app.serve", "--host", "0.0.0.0", "--port", "7860"]
