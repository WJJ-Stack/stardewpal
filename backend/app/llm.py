"""LLM 客户端：OpenAI 兼容协议，支持 Qwen / 任意兼容服务，无 key 时自动 mock。

设计取舍
--------
项目要在**没有 key 的情况下也能完整跑通**（本地开发、CI、面试演示降级）。
因此 mock 不是玩具，而是返回结构良好、可被评测的确定性响应。
"""
from __future__ import annotations

import json
import re
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any

import httpx

from backend.app.config import resolve_provider, settings


@dataclass
class LLMResult:
    content: str
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    usage: dict[str, int] = field(default_factory=dict)
    model: str = ''
    mocked: bool = False


class LLMError(RuntimeError):
    pass


class LLMClient:
    """OpenAI 兼容客户端，支持多 provider 切换。

    通过 LLM_PROVIDER 环境变量选择服务商：
        qwen / zhipu / modelscope / siliconflow / mock

    未配置 API Key 时自动进入 mock 模式，返回确定性响应（标记 mocked=True），
    使整条链路在无网络、无额度的情况下仍可被评测与 CI 验证。
    """

    def __init__(self) -> None:
        cfg = resolve_provider()
        self.provider = cfg['name']
        self.base_url = cfg['base_url']
        self.api_key = cfg['api_key']
        self.model = cfg['model']
        self.vlm_model = cfg['vlm_model']
        self.timeout = settings.llm_timeout

    @property
    def live(self) -> bool:
        return bool(self.api_key) and self.provider != 'mock'

    def describe(self) -> dict[str, Any]:
        return {
            'provider': self.provider,
            'model': self.model if self.live else 'mock',
            'vlm_model': self.vlm_model if self.live else 'mock',
            'live': self.live,
            'base_url': self.base_url if self.live else '',
        }

    # ---------------------------------------------------------------- chat
    async def chat(self, messages: list[dict[str, Any]], *,
                   model: str | None = None,
                   temperature: float = 0.2,
                   max_tokens: int = 1400,
                   tools: list[dict] | None = None,
                   tool_choice: str | None = None) -> LLMResult:
        if not self.live:
            return _mock_chat(messages, tools)

        payload: dict[str, Any] = {
            'model': model or self.model,
            'messages': messages,
            'temperature': temperature,
            'max_tokens': max_tokens,
        }
        if tools:
            payload['tools'] = tools
            if tool_choice:
                payload['tool_choice'] = tool_choice

        headers = {'Authorization': f'Bearer {self.api_key}',
                   'Content-Type': 'application/json'}
        last_err = ''
        for attempt in range(settings.llm_max_retries):
            try:
                async with httpx.AsyncClient(timeout=self.timeout) as client:
                    r = await client.post(f'{self.base_url}/chat/completions',
                                          json=payload, headers=headers)
                    if r.status_code >= 400:
                        last_err = f'HTTP {r.status_code}: {r.text[:300]}'
                        if r.status_code in (401, 403):
                            raise LLMError(f'鉴权失败，请检查 DASHSCOPE_API_KEY：{last_err}')
                        if r.status_code == 429 or r.status_code >= 500:
                            continue
                        raise LLMError(last_err)
                    data = r.json()
                return _parse_response(data, payload['model'])
            except LLMError:
                raise
            except Exception as e:  # noqa: BLE001
                last_err = str(e)
                if attempt == settings.llm_max_retries - 1:
                    break
        raise LLMError(f'LLM 调用失败：{last_err}')

    # ---------------------------------------------------------------- stream
    async def stream(self, messages: list[dict[str, Any]], *,
                     model: str | None = None,
                     temperature: float = 0.2,
                     max_tokens: int = 1400) -> AsyncIterator[str]:
        if not self.live:
            mock = _mock_chat(messages, None)
            for i in range(0, len(mock.content), 12):
                yield mock.content[i:i + 12]
            return

        payload = {
            'model': model or self.model,
            'messages': messages,
            'temperature': temperature,
            'max_tokens': max_tokens,
            'stream': True,
        }
        headers = {'Authorization': f'Bearer {self.api_key}',
                   'Content-Type': 'application/json'}
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            async with client.stream('POST', f'{self.base_url}/chat/completions',
                                     json=payload, headers=headers) as r:
                if r.status_code >= 400:
                    body = await r.aread()
                    raise LLMError(f'HTTP {r.status_code}: {body[:300]!r}')
                async for line in r.aiter_lines():
                    if not line or not line.startswith('data:'):
                        continue
                    chunk = line[5:].strip()
                    if chunk == '[DONE]':
                        break
                    try:
                        obj = json.loads(chunk)
                    except json.JSONDecodeError:
                        continue
                    for ch in obj.get('choices', []):
                        delta = ch.get('delta') or {}
                        text = delta.get('content')
                        if text:
                            yield text

    # ---------------------------------------------------------------- vision
    async def vision(self, prompt: str, image_b64: str, *,
                     model: str | None = None,
                     max_tokens: int = 700) -> LLMResult:
        messages = [{
            'role': 'user',
            'content': [
                {'type': 'text', 'text': prompt},
                {'type': 'image_url',
                 'image_url': {'url': f'data:image/png;base64,{image_b64}'}},
            ],
        }]
        if not self.live:
            return _mock_vision(prompt)
        return await self.chat(messages, model=model or self.vlm_model,
                               max_tokens=max_tokens, temperature=0.1)


def _parse_response(data: dict, model: str) -> LLMResult:
    choices = data.get('choices') or []
    if not choices:
        raise LLMError(f'响应无 choices: {json.dumps(data)[:300]}')
    msg = choices[0].get('message') or {}
    calls = []
    for tc in msg.get('tool_calls') or []:
        fn = tc.get('function') or {}
        try:
            args = json.loads(fn.get('arguments') or '{}')
        except json.JSONDecodeError:
            args = {}
        calls.append({'id': tc.get('id'), 'name': fn.get('name'), 'arguments': args})
    return LLMResult(
        content=msg.get('content') or '',
        tool_calls=calls,
        usage=data.get('usage') or {},
        model=model,
    )


# ---------------------------------------------------------------- mock（无 key 时）

def _mock_chat(messages: list[dict], tools: list[dict] | None) -> LLMResult:
    """确定性 mock：从上下文中抽取第一条引用并组织成可读回答。

    目的是让整条链路（检索 → 提示词 → 输出 → 评测）在无 key 时可验证，
    并明确标记 mocked=True，避免误当作真实效果。
    """
    system = next((m['content'] for m in messages if m.get('role') == 'system'), '')
    user = next((m['content'] for m in reversed(messages) if m.get('role') == 'user'), '')
    if isinstance(user, list):
        user = ' '.join(p.get('text', '') for p in user if isinstance(p, dict))

    if 'need_retrieval' in system:
        payload = {'need_retrieval': True, 'reason': 'mock: 默认检索',
                   'query': str(user)[:80], 'sufficient': False}
        return LLMResult(content=json.dumps(payload, ensure_ascii=False), mocked=True)
    if 'sufficient' in system and 'JSON' in system:
        payload = {'sufficient': True, 'reason': 'mock',
                   'missing': '', 'better_query': ''}
        return LLMResult(content=json.dumps(payload, ensure_ascii=False), mocked=True)

    u = str(user)

    # ---- 工具计算结果优先：mock 也必须把计算结果作为主答案
    calc_lines = re.findall(r'- (.+?): (.+?) = ([\d,.]+)', u)
    if calc_lines:
        parts = ['根据资料计算：']
        for item, formula, value in calc_lines:
            parts.append(f'- {item}：{formula} = **{value} 金**　[1]')
        parts.append('\n> 演示模式：计算由本地工具完成（数值可信），'
                     '自然语言润色需配置 DASHSCOPE_API_KEY。')
        return LLMResult(content='\n'.join(parts), mocked=True)

    # ---- 无资料 → 拒答（模拟真实模型的正确行为）
    has_ctx = bool(re.search(r'\[\d+\] 来源：', u))
    if not has_ctx:
        return LLMResult(
            content='资料库中没有找到与该问题相关的信息，无法回答。'
                    '如果你问的是其他游戏的内容，本知识库仅覆盖《星露谷物语》。',
            mocked=True,
        )

    # ---- 有资料：抽取首个引用作答
    ctx = ''
    m = re.search(r'\[(\d+)\][^\n]*\n(.{0,240})', u, re.S)
    if m:
        ctx = f'根据资料[{m.group(1)}]：{m.group(2).strip()[:180]}'
    content = (ctx or '（无可用资料）') + \
              '\n\n> 演示模式：当前回答由检索片段直接拼接，' \
              '配置 DASHSCOPE_API_KEY 后可获得真实生成结果。'
    return LLMResult(content=content, mocked=True)


def _mock_vision(prompt: str) -> LLMResult:
    m = re.search(r'候选[：:]\s*([^\n]+)', prompt)
    cands = []
    if m:
        cands = [c.strip() for c in re.split(r'[、,，]', m.group(1)) if c.strip()]
    pick = cands[0] if cands else '无法判断'
    return LLMResult(content=json.dumps({'choice': pick, 'confidence': 0.5,
                                         'reason': 'mock 模式'},
                                        ensure_ascii=False), mocked=True)


_client: LLMClient | None = None


def get_llm() -> LLMClient:
    global _client
    if _client is None:
        _client = LLMClient()
    return _client
