"""检索层评测：不调用大模型，零成本，数字完全可复现。

为什么单独测检索层
------------------
这个项目的核心不是「模型会不会说话」，而是「该看的 wiki 段落有没有被送进模型」。
前面几轮踩的坑（答成「秋季15日」、「服装节日后不保留」、「资料里没给答案」）
全部是检索或数据层的问题，模型只是受害者。

这类指标不依赖 LLM，因此：
  · 不花 API 额度
  · 结果确定、可复现（同一份索引 + 同一份评测集 = 同一个数）
  · 能直接做消融，说明每个组件贡献了多少

指标
----
Hit@K        正确文档是否出现在前 K 条结果里（排序质量）
MRR          正确文档排名倒数的均值（越接近 1 越好）
AnswerInCtx  最终拼给模型的上下文里，是否真的含标准答案（端到端可用性）

用法
----
    python -m backend.pipeline.eval_retrieval
    python -m backend.pipeline.eval_retrieval --limit 100 --top-k 6
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from backend.app.config import settings  # noqa: E402
from backend.app.embedding import get_embedder  # noqa: E402
from backend.app.retrieval import (  # noqa: E402
    HybridRetriever,
    Retrieved,
    _coverage_factor,
    _merge_ranked,
    _sparse_query,
    _title_affinity,
    build_context,
    expand_query,
)
from backend.app.store import Store  # noqa: E402
from backend.pipeline import eval_set  # noqa: E402

CASES_FILE = settings.raw_dir.parent / 'eval' / 'cases.jsonl'


def _kind_of(store: Store, chunk_id: int) -> str:
    try:
        row = store.conn.execute('SELECT kind FROM chunks WHERE id = ?',
                                 (chunk_id,)).fetchone()
        return row['kind'] if row else 'text'
    except Exception:
        return 'text'


def _to_hits(store: Store, rows: list[dict]) -> list[Retrieved]:
    return [
        Retrieved(chunk_id=it['chunk_id'], title=it['title'],
                  section=it.get('section'), text=it['text'],
                  score=float(it['score']), kind=_kind_of(store, it['chunk_id']))
        for it in rows
    ]


# ---------------------------------------------------------------- 四种配置

def run_bm25(store: Store, r: HybridRetriever, q: str, k: int) -> list[Retrieved]:
    return _to_hits(store, store.search_bm25(expand_query(q), k))


def run_dense(store: Store, r: HybridRetriever, q: str, k: int) -> list[Retrieved]:
    try:
        emb = get_embedder().encode_one(expand_query(q))
        return _to_hits(store, store.search_vec(emb, k))
    except Exception:
        return []


def run_hybrid(store: Store, r: HybridRetriever, q: str, k: int,
               coverage: bool = True, affinity: bool = True) -> list[Retrieved]:
    eq = expand_query(q)
    sparse_full = store.search_bm25(eq, r.top_k_sparse)
    sparse_ent = store.search_bm25(_sparse_query(q), r.top_k_sparse)
    sparse = _merge_ranked([(sparse_full, 1.0), (sparse_ent, 0.95)])

    dense: list[dict] = []
    try:
        dense = store.search_vec(get_embedder().encode_one(eq), r.top_k_dense)
    except Exception:
        pass

    fused = r._rrf(sparse, dense)
    if not fused:
        return []

    boost = {'infobox': 1.25, 'table': 1.15}
    for x in fused:
        if x.kind in boost:
            x.score *= boost[x.kind]
        if affinity:
            x.score *= _title_affinity(q, x.title)
        if coverage:
            x.score *= _coverage_factor(q, x.text, x.title)
    fused.sort(key=lambda x: -x.score)
    return fused[:k]


# ---------------------------------------------------------------- 指标

def source_hit(case: dict, hits: list[Retrieved]) -> tuple[bool, int]:
    """正确文档是否命中；返回 (是否命中, 排名从 1 开始；0 表示未命中)。"""
    src = (case.get('source') or '').split('|')
    titles = [s.strip() for s in src if s.strip()]
    if not titles:
        return False, 0
    for rank, h in enumerate(hits, start=1):
        if h.title in titles:
            return True, rank
    return False, 0


# 答案由计算工具产出、而非字面存在于 wiki 的题型
COMPUTED_KINDS = {'quality', 'artisan'}


def answer_in_context(case: dict, chunks: list[Retrieved],
                      max_chars: int = 3600) -> bool | None:
    """标准答案是否字面出现在上下文里。

    只对 fact / attr / compare 有意义；
    quality / artisan 的答案是算出来的（25×1.5=37，wiki 里没有 37），
    返回 None 表示"不适用"，由 calc_correct 单独衡量。
    """
    if case.get('kind') in COMPUTED_KINDS:
        return None
    if not chunks:
        return False
    ctx = build_context(chunks, max_chars=max_chars).replace(',', '')
    exp = str(case.get('expected') or '')
    if not exp:
        return False
    nums = re.findall(r'\d[\d,]*\.?\d*', exp)
    if nums:
        return nums[0].replace(',', '') in ctx
    return exp in ctx


def base_price_in_context(case: dict, chunks: list[Retrieved],
                          max_chars: int = 3600) -> float | None:
    """从上下文里取该文档的基础卖价（计算题的前置条件）。

    在命中块的「属性数据」里找 `卖价：N 金`。
    取不到就说明检索没给到计算所需的输入。
    """
    if not chunks:
        return None
    src = [s.strip() for s in (case.get('source') or '').split('|') if s.strip()]
    for h in chunks:
        if h.title not in src:
            continue
        for m in re.finditer(r'卖价[：:]\s*(-?\d+(?:\.\d+)?)', h.text):
            val = float(m.group(1))
            if val > 0:      # wiki 用 -1 表示无售价
                return val
    return None


def calc_correct(case: dict, chunks: list[Retrieved]) -> bool | None:
    """计算题端到端：基础价能取到 + 工具算出的值等于期望值。

    这是本项目「不让模型心算」这条设计的直接验证，
    全程不调用 LLM，因此结果确定可复现。
    """
    kind = case.get('kind')
    if kind not in COMPUTED_KINDS:
        return None
    if case.get('numeric') is None:
        return None
    base = base_price_in_context(case, chunks)
    if base is None:
        return False

    from backend.app.tools.calc import sell_price

    q = case['question']
    if kind == 'quality':
        quality = next((k for k in ('银星', '金星', '铱星') if k in q), '普通')
        got = sell_price(base, quality).value
    else:
        artisan = next((a for a in ('果酒', '果汁', '腌菜', '果酱', '蜂蜜酒')
                        if a in q), None)
        if artisan is None:
            return False
        got = sell_price(base, '普通', artisan=artisan).value

    exp = float(case['numeric'])
    return abs(got - exp) / max(abs(exp), 1e-9) < 0.01


# ---------------------------------------------------------------- 主流程

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--per-type', type=int, default=40)
    ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--top-k', type=int, default=6)
    ap.add_argument('--out', default=None)
    args = ap.parse_args()

    store = Store(settings.db_path)
    st = store.stats()
    if not st['docs']:
        raise SystemExit('索引为空，先运行 build_index')
    print(f'索引：{st["docs"]} 文档 / {st["chunks"]} 块 / {st["items"]} 物品'
          f'（向量 {st["vec"]}）')

    if not CASES_FILE.exists():
        print(f'生成评测集 -> {CASES_FILE}')
        CASES_FILE.parent.mkdir(parents=True, exist_ok=True)
        eval_set.save(eval_set.generate(store, per_type=args.per_type), CASES_FILE)
    cases = eval_set.load(CASES_FILE)
    print(f'评测集：{len(cases)} 条')

    # 只有带 source 的题能判断"检索对不对"；拒答题不参与
    cases = [c for c in cases if c.get('source') and c['kind'] != 'refusal']
    if args.limit:
        by_kind: dict[str, list[dict]] = {}
        for c in cases:
            by_kind.setdefault(c['kind'], []).append(c)
        per = max(1, args.limit // max(len(by_kind), 1))
        picked: list[dict] = []
        for k in sorted(by_kind):
            picked.extend(by_kind[k][:per])
        cases = picked
    print(f'参与检索评测：{len(cases)} 条\n')

    r = HybridRetriever(store, settings.top_k_dense, settings.top_k_sparse, settings.rrf_k)
    K = args.top_k

    variants: dict[str, object] = {
        'BM25 only': lambda c: run_bm25(store, r, c['question'], K),
        'Vector only': lambda c: run_dense(store, r, c['question'], K),
        'Hybrid (no coverage)': lambda c: run_hybrid(
            store, r, c['question'], K, coverage=False),
        'Hybrid (current)': lambda c: run_hybrid(store, r, c['question'], K),
    }

    results: dict[str, dict] = {}
    t0 = time.time()
    for name, fn in variants.items():
        hit1 = hit3 = hitk = 0
        rr_sum = 0.0
        n = 0
        misses: list[str] = []
        for c in cases:
            hits = fn(c)
            ok, rank = source_hit(c, hits)
            n += 1
            if ok:
                hitk += 1
                rr_sum += 1.0 / rank
                if rank == 1:
                    hit1 += 1
                if rank <= 3:
                    hit3 += 1
            elif len(misses) < 5:
                misses.append(c['question'])
        results[name] = {
            'n': n,
            'hit@1': hit1 / n if n else 0.0,
            'hit@3': hit3 / n if n else 0.0,
            f'hit@{K}': hitk / n if n else 0.0,
            'mrr': rr_sum / n if n else 0.0,
            'misses': misses,
        }

    # 端到端可用性：分两类题分别衡量
    #   · 字面题（fact/attr/compare）→ 答案是否进了上下文
    #   · 计算题（quality/artisan） → 基础价能取到 + 工具算对
    for label, use_expand in (('Ctx (no expand)', False), ('Ctx (+expand)', True)):
        lit_ok = lit_n = 0
        calc_ok = calc_n = 0
        for c in cases:
            raw = run_hybrid(store, r, c['question'], K)
            chunks = r.expand(raw, query=c['question']) if use_expand else raw
            lit = answer_in_context(c, chunks)
            if lit is not None:
                lit_n += 1
                lit_ok += int(lit)
            cc = calc_correct(c, chunks)
            if cc is not None:
                calc_n += 1
                calc_ok += int(cc)
        results[label] = {
            'lit_n': lit_n,
            'answer_in_ctx': (lit_ok / lit_n if lit_n else 0.0),
            'calc_n': calc_n,
            'calc_correct': (calc_ok / calc_n if calc_n else 0.0),
        }

    elapsed = time.time() - t0

    # ---- 报告
    print('=' * 80)
    print(f'{"配置":<26}{"Hit@1":>9}{"Hit@3":>9}{f"Hit@{K}":>9}{"MRR":>9}')
    print('-' * 80)
    for name in variants:
        m = results[name]
        print(f'{name:<26}{m["hit@1"]:>8.1%}{m["hit@3"]:>9.1%}'
              f'{m[f"hit@{K}"]:>9.1%}{m["mrr"]:>9.3f}')
    print('-' * 80)
    for label in ('Ctx (no expand)', 'Ctx (+expand)'):
        m = results[label]
        print(f'{label:<26} 字面题答案在上下文中 {m["answer_in_ctx"]:>7.1%}'
              f'  ({m["lit_n"]} 条)')
        print(f'{"":<26} 计算题端到端算对     {m["calc_correct"]:>7.1%}'
              f'  ({m["calc_n"]} 条)')
    print('=' * 80)

    print(f'\n按题型（Hybrid current）Hit@{K}：')
    by_kind: dict[str, list[bool]] = {}
    for c in cases:
        ok, _ = source_hit(c, variants['Hybrid (current)'](c))
        by_kind.setdefault(c['kind'], []).append(ok)
    for k in sorted(by_kind):
        v = by_kind[k]
        print(f'  {k:<10}{sum(v)/len(v):>8.1%}  ({sum(v)}/{len(v)})')

    miss = results['Hybrid (current)']['misses']
    if miss:
        print('\n未命中样例：')
        for q in miss:
            print(f'  · {q}')

    print(f'\n耗时 {elapsed:.1f}s')

    if args.out:
        Path(args.out).write_text(json.dumps(
            {'results': results, 'cases': len(cases), 'top_k': K, 'index': st},
            ensure_ascii=False, indent=1), encoding='utf-8')
        print(f'结果已保存 {args.out}')


if __name__ == '__main__':
    main()
