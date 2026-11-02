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

# 构建期自检：魔搭等平台的 Docker 构建可能不执行 git lfs pull，
# 导致 .db/.onnx/.zip 以 LFS 指针形式进入镜像。
# 这里在构建期就把指针文件处理掉，避免运行时才重建。
RUN python - <<'PY'
import shutil, subprocess, sys
from pathlib import Path

data = Path('/app/backend/data')
lfs = b'version https://git-lfs'

def is_pointer(p: Path) -> bool:
    try:
        with p.open('rb') as f:
            return f.read(60).startswith(lfs)
    except OSError:
        return False

# 1) 模型是指针 -> 删掉，让运行时重新下载（构建期下载可能无网络）
model = data / 'models/bge-small-zh-v1.5/model_quantized.onnx'
if model.exists() and is_pointer(model):
    print('[build] 模型是 LFS 指针，删除（运行时自动下载）')
    model.unlink()

# 2) 索引是指针 -> 删除（运行时用 pages.jsonl 重建）
db = data / 'stardewpal.db'
if db.exists() and is_pointer(db):
    print('[build] 索引是 LFS 指针，删除（运行时自动重建）')
    db.unlink()

# 3) 图标 zip 是指针 -> 删除（图标缺失不影响文字问答）
zipf = data / 'icons.zip'
if zipf.exists() and is_pointer(zipf):
    print('[build] 图标包是 LFS 指针，删除')
    zipf.unlink()

# 4) 有真实 zip 就解包
if zipf.exists() and zipf.stat().st_size > 100_000:
    subprocess.run([sys.executable, '-m', 'backend.pipeline.bundle_assets', '--unpack'],
                   check=False)

# 5) 校验语料（自检重建的基础，必须存在）
corpus = data / 'raw/pages.jsonl'
if corpus.exists():
    print(f'[build] 语料就绪: {corpus.stat().st_size/1048576:.1f} MB')
else:
    print('[build] 警告：缺少 pages.jsonl，运行时无法重建索引')
PY

ENV PYTHONPATH=/app
EXPOSE 7860

HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
    CMD curl -fsS http://127.0.0.1:7860/health || exit 1

# 单 worker：SQLite 写入与模型加载都是进程级的，
# 多 worker 会各自加载 23MB 模型，在 CPU 上得不偿失
CMD ["python", "-m", "backend.app.serve", "--host", "0.0.0.0", "--port", "7860"]
