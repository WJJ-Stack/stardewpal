"""FastAPI 应用：对话、截图问答、检索调试、评测端点。"""
from __future__ import annotations

import base64
import io
import json
import time
from collections import defaultdict, deque
from typing import Annotated, Any

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from backend.app.agent import Agent
from backend.app.config import ROOT, settings
from backend.app.embedding import get_embedder
from backend.app.llm import get_llm
from backend.app.retrieval import HybridRetriever, build_context
from backend.app.store import Store
from backend.app.vision.pipeline import get_vision

app = FastAPI(title='StardewPal', version='0.1.0',
              description='星露谷物语多模态 RAG 问答（求职作品项目）')

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=False,
    allow_methods=['*'],
    allow_headers=['*'],
)

_store: Store | None = None
_rate: dict[str, deque[float]] = defaultdict(deque)


def store() -> Store:
    global _store
    if _store is None:
        _store = Store(settings.db_path)
    return _store


def _rate_limit(request: Request) -> None:
    ip = request.client.host if request.client else 'unknown'
    now = time.time()
    q = _rate[ip]
    while q and now - q[0] > 60:
        q.popleft()
    if len(q) >= settings.rate_limit_per_min:
        raise HTTPException(429, '请求过于频繁，请稍后再试')
    q.append(now)


# ---------------------------------------------------------------- 模型

class ChatTurn(BaseModel):
    """一问一答。用于多轮上下文。"""
    question: str = Field(default='', max_length=500)
    answer: str = Field(default='', max_length=4000)


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=500)
    top_k: int = Field(default=6, ge=1, le=12)
    stream: bool = False
    # 多轮上下文：前端把最近几轮问答带上。
    # 没有这个字段时，像「这题该怎么选」这种省略主语的追问必然失败。
    history: list[ChatTurn] = Field(default_factory=list, max_length=10)


class SearchRequest(BaseModel):
    query: str = Field(min_length=1, max_length=300)
    top_k: int = Field(default=8, ge=1, le=20)


# ---------------------------------------------------------------- 基础端点

@app.get('/health')
async def health() -> dict[str, Any]:
    s = store()
    st = s.stats()
    emb_ready = get_embedder().ready
    llm = get_llm()
    return {
        'status': 'ok' if st['docs'] > 0 else 'empty_index',
        'index': st,
        'embedder_ready': emb_ready,
        'llm_live': llm.live,
        'vlm_live': llm.vlm_live,
        'vlm_model': llm.vlm_model if llm.vlm_live else 'mock',
        'ocr': get_vision().ocr.available,
        'sqlite_vec': s.has_vec(),
    }


@app.get('/api/meta')
async def meta() -> dict[str, Any]:
    st = store().stats()
    raw_stats = settings.raw_dir / 'crawl_stats.json'
    crawl = json.loads(raw_stats.read_text(encoding='utf-8')) if raw_stats.exists() else {}
    llm = get_llm()
    return {
        'project': 'StardewPal',
        'corpus': {'source': 'BWiki 星露谷物语（CC BY-NC-SA）', **crawl, **st},
        'models': {
            'embedding': 'bge-small-zh-v1.5 (ONNX INT8, 512d)',
            'llm': llm.model if llm.live else 'mock',
            'llm_provider': llm.provider,
            'vlm': llm.vlm_model if llm.vlm_live else 'mock',
            'vlm_provider': llm.vlm_provider,
        },
        'notice': '本作品仅用于学习与技术展示，非商用。内容版权归 BWiki 及原作者所有。',
    }


# ---------------------------------------------------------------- 检索

@app.post('/api/search')
async def search(req: SearchRequest, request: Request) -> dict[str, Any]:
    _rate_limit(request)
    r = HybridRetriever(store(), settings.top_k_dense, settings.top_k_sparse, settings.rrf_k)
    chunks = r.retrieve(req.query, top_k=req.top_k)
    return {
        'query': req.query,
        'count': len(chunks),
        'results': [c.as_dict() for c in chunks],
        'context_preview': build_context(chunks, max_chars=1200),
    }


# ---------------------------------------------------------------- 文本问答

@app.post('/api/ask')
async def ask(req: AskRequest, request: Request) -> dict[str, Any]:
    _rate_limit(request)
    t0 = time.time()

    agent = Agent(store())
    result = await agent.run(req.question, top_k=req.top_k,
                             history=[h.model_dump() for h in req.history])
    return {
        'question': req.question,
        'answer': result.answer,
        'citations': result.citations,
        'queries': result.queries,
        'calc': result.calc,
        'trace': result.trace,
        'mocked': result.mocked,
        'elapsed_ms': int((time.time() - t0) * 1000),
    }


@app.post('/api/ask/stream')
async def ask_stream(req: AskRequest, request: Request) -> StreamingResponse:
    _rate_limit(request)

    async def gen():
        agent = Agent(store())
        t0 = time.time()
        # 先跑决策与检索，把过程以 SSE 事件推给前端
        result = await agent.run(req.question, top_k=req.top_k)
        yield _sse('trace', {'steps': result.trace, 'queries': result.queries})
        if result.calc:
            yield _sse('calc', {'results': result.calc})
        if result.citations:
            yield _sse('citations', {'items': result.citations})

        try:
            llm = get_llm()
            ctx = '\n\n'.join(
                f"[{c['index']}] 来源：《{c['title']}》{(' · ' + c['section']) if c['section'] else ''}\n{c['snippet']}"
                for c in result.citations
            )
            system = ('你是星露谷物语攻略助手。只用资料回答，数值标注来源编号，不做心算。')
            calc_block = ''
            if result.calc:
                calc_block = '【工具计算结果】\n' + '\n'.join(
                    f"- {r.get('item')}: {r.get('formula')} = {r.get('value')}"
                    for r in result.calc) + '\n\n'
            messages = [
                {'role': 'system', 'content': system},
                {'role': 'user',
                 'content': f"问题：{req.question}\n\n{calc_block}资料：\n{ctx}"},
            ]
            async for piece in llm.stream(messages):
                yield _sse('delta', {'text': piece})
        except Exception as e:  # noqa: BLE001
            # 流式失败时退回一次性答案
            for i in range(0, len(result.answer), 16):
                yield _sse('delta', {'text': result.answer[i:i + 16]})
            yield _sse('warning', {'message': str(e)[:200]})

        yield _sse('done', {'elapsed_ms': int((time.time() - t0) * 1000),
                            'mocked': result.mocked})

    return StreamingResponse(gen(), media_type='text/event-stream',
                             headers={'Cache-Control': 'no-cache',
                                      'X-Accel-Buffering': 'no'})


def _sse(event: str, data: dict) -> str:
    return f'event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n'


# ---------------------------------------------------------------- 截图问答

@app.post('/api/vision/analyze')
async def vision_analyze(
    request: Request,
    file: Annotated[UploadFile, File()],
) -> dict[str, Any]:
    _rate_limit(request)
    data = await _read_image(file)
    t0 = time.time()
    res = get_vision().analyze(data)
    return {'result': res.as_dict(), 'trace': res.trace,
            'elapsed_ms': int((time.time() - t0) * 1000)}


@app.post('/api/vision/ask')
async def vision_ask(
    request: Request,
    file: Annotated[UploadFile, File()],
    question: Annotated[str, Form()] = '',
    use_vlm: Annotated[bool, Form()] = True,
    history_json: Annotated[str, Form()] = '',
) -> dict[str, Any]:
    """截图问答：图片直接交给 VLM 回答。

    设计演进见 backend/app/vision/pipeline.py 的模块注释。
    核心结论：**少即是多**。OCR 文字、裁剪图、复杂提示词都会降低效果。
    """
    _rate_limit(request)
    data = await _read_image(file)
    t0 = time.time()

    # 多轮上下文：前端以 JSON 字符串传，避免 multipart 嵌套数组的兼容问题
    history_dicts: list[dict[str, str]] = []
    if history_json:
        try:
            raw = json.loads(history_json)
            if isinstance(raw, list):
                history_dicts = [
                    {'question': str(x.get('question', ''))[:500],
                     'answer': str(x.get('answer', ''))[:4000]}
                    for x in raw[-8:] if isinstance(x, dict)
                ]
        except (json.JSONDecodeError, TypeError):
            pass

    # 图片**直接交给多模态模型**，不做 OCR → 文字描述 的转述。
    # 每转述一道就丢一层信息：实测转述会把沙漠节认成夜市，
    # 然后检索也跟着错，整条链路全歪。
    b64 = _to_png_b64(data)

    agent = Agent(store())
    result = await agent.run(question or '这张截图里是什么？',
                             top_k=settings.top_k_final,
                             history=history_dicts,
                             image_b64=b64)

    answer = result.answer
    citations = result.citations
    calc = result.calc
    mocked = result.mocked
    trace = result.trace
    retrieved = [c['title'] for c in citations]
    vision = get_vision().analyze(data)   # 仅用于面板展示，不参与回答
    vision.trace = []
    vision.vlm_answer = answer
    return {
        'question': question,
        'vision': vision.as_dict(),
        'vision_trace': vision.trace,
        'answer': answer,
        'citations': citations,
        'calc': calc,
        'trace': trace,
        'retrieved': retrieved,
        'mocked': mocked,
        'elapsed_ms': int((time.time() - t0) * 1000),
    }


# ---------------------------------------------------------------- 截图问答辅助

def _to_png_b64(image_bytes: bytes) -> str:
    """把上传图片转成 base64 PNG，直接喂给多模态模型。

    只做尺寸控制，不做任何内容加工 —— 任何"预处理"都会丢信息。
    """
    from PIL import Image

    img = Image.open(io.BytesIO(image_bytes))
    img.load()
    if img.mode not in ('RGB', 'RGBA'):
        img = img.convert('RGB')
    if max(img.size) > 1400:
        r = 1400 / max(img.size)
        img = img.resize((int(img.width * r), int(img.height * r)), Image.LANCZOS)
    buf = io.BytesIO()
    img.convert('RGB').save(buf, format='PNG', optimize=True)
    if buf.tell() > 1400 * 1024:
        buf = io.BytesIO()
        img.convert('RGB').save(buf, format='JPEG', quality=88, optimize=True)
    return base64.b64encode(buf.getvalue()).decode()


async def _read_image(file: UploadFile) -> bytes:
    data = await file.read()
    if not data:
        raise HTTPException(400, '文件为空')
    if len(data) > settings.max_image_mb * 1024 * 1024:
        raise HTTPException(413, f'图片超过 {settings.max_image_mb}MB 限制')
    return data


# ---------------------------------------------------------------- 错误处理

@app.exception_handler(HTTPException)
async def http_err(request: Request, exc: HTTPException) -> JSONResponse:
    return JSONResponse(status_code=exc.status_code,
                        content={'error': exc.detail, 'status': exc.status_code})


# ---------------------------------------------------------------- 静态资源
# 注意：必须在所有 /api 路由之后挂载，否则会把 API 路由遮住

FRONTEND = ROOT.parent / 'frontend'
if (FRONTEND / 'static').exists():
    app.mount('/static', StaticFiles(directory=str(FRONTEND / 'static')), name='static')

ICONS_DIR = settings.icons_dir
if ICONS_DIR.exists():
    app.mount('/icons', StaticFiles(directory=str(ICONS_DIR)), name='icons')


@app.get('/', response_model=None)
async def index():
    page = FRONTEND / 'index.html'
    if page.exists():
        return FileResponse(page)
    return JSONResponse({'message': 'StardewPal API', 'docs': '/docs'})
