"""评测运行器：跑评测集，输出各项指标。

用法
----
    python -m backend.pipeline.run_eval --limit 50
    python -m backend.pipeline.run_eval --no-agent   # 消融：关闭 Agent 决策
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from backend.app.agent import Agent  # noqa: E402
from backend.app.config import settings  # noqa: E402
from backend.app.store import Store  # noqa: E402
from backend.pipeline import eval_set  # noqa: E402

# 消融开关：用于对比「有/无 Agent 决策」的效果差异
ABLATION = {'use_agent': True}


async def run_one(agent: Agent, case: dict, top_k: int) -> dict:
    t0 = time.time()
    if ABLATION['use_agent']:
        res = await agent.run(case['question'], top_k=top_k)
        answer = res.answer
        trace = res.trace
        calc = res.calc
    else:
        # 消融：固定单轮检索，不做充分性判断
        chunks = agent._search(case['question'], top_k)
        answer, _ = await agent._answer(case['question'], chunks, [], Agent.__dict__ and
                                        _Null())
        trace = [{'step': 'ablation_single_pass'}]
        calc = []

    score = eval_set.score_answer(case, answer)
    return {
        'id': case['id'],
        'kind': case['kind'],
        'question': case['question'],
        'expected': case['expected'],
        'answer': answer[:500],
        'correct': score['correct'],
        'reason': score['reason'],
        'calc_used': bool(calc),
        'steps': len(trace),
        'elapsed_ms': int((time.time() - t0) * 1000),
    }


class _Null:
    """空 trace，用于消融模式。"""

    def add(self, *a, **k):
        pass


async def main_async(args) -> None:
    store = Store(settings.db_path)
    st = store.stats()
    if st['docs'] == 0:
        raise SystemExit('索引为空，先运行 build_index')

    cases_file = settings.data_dir / 'eval' / 'cases.jsonl' if hasattr(settings, 'data_dir') \
        else settings.raw_dir.parent / 'eval' / 'cases.jsonl'
    if args.regenerate or not cases_file.exists():
        print(f'生成评测集 -> {cases_file}')
        cases = eval_set.generate(store, per_type=args.per_type)
        eval_set.save(cases, cases_file)
    cases = eval_set.load(cases_file)
    print(f'评测集 {len(cases)} 条')

    if args.limit:
        # 每个类型均匀抽样，保证 6 种题型都被覆盖
        by_kind: dict[str, list[dict]] = defaultdict(list)
        for c in cases:
            by_kind[c['kind']].append(c)
        kinds = sorted(by_kind)
        per = max(2, args.limit // max(len(kinds), 1))
        picked = []
        for k in kinds:
            picked.extend(by_kind[k][:per])
        cases = picked
        print(f'抽样 {len(cases)} 条（{len(kinds)} 类 × {per}）')

    ABLATION['use_agent'] = not args.no_agent
    agent = Agent(store)

    records = []
    for i, c in enumerate(cases, start=1):
        try:
            r = await run_one(agent, c, args.top_k)
        except Exception as e:  # noqa: BLE001
            r = {'id': c['id'], 'kind': c['kind'], 'question': c['question'],
                 'expected': c['expected'], 'answer': f'ERROR: {e}',
                 'correct': False, 'reason': str(e)[:200], 'calc_used': False,
                 'steps': 0, 'elapsed_ms': 0}
        records.append(r)
        mark = '✓' if r['correct'] else '✗'
        print(f'  [{i}/{len(cases)}] {mark} {r["kind"]:8s} {r["question"][:34]:34s} '
              f'-> {r["answer"][:40]}', flush=True)

    summary = summarize(records)
    out_dir = settings.runs_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime('%Y%m%d-%H%M%S')
    out = out_dir / f'eval-{stamp}.json'
    out.write_text(json.dumps({'summary': summary, 'records': records},
                              ensure_ascii=False, indent=1), encoding='utf-8')

    print('\n' + '=' * 62)
    print(f'整体准确率  {summary["overall"]:.1%}  ({summary["correct"]}/{summary["total"]})')
    print(f'平均耗时    {summary["avg_ms"]} ms')
    print(f'工具使用率  {summary["calc_rate"]:.1%}')
    print('-' * 62)
    print(f'{"题型":<10}{"准确率":>10}{"样本":>8}')
    for k, v in summary['by_kind'].items():
        print(f'{k:<10}{v["acc"]:>9.1%}{v["n"]:>8}')
    print('=' * 62)
    print(f'结果已保存 {out}')

    # 失败用例样例，便于分析
    fails = [r for r in records if not r['correct']][:8]
    if fails:
        print('\n失败样例:')
        for f in fails:
            print(f'  · [{f["kind"]}] {f["question"]}')
            print(f'    期望: {f["expected"][:60]}')
            print(f'    实际: {f["answer"][:90]}')


def summarize(records: list[dict]) -> dict:
    by_kind: dict[str, list[bool]] = defaultdict(list)
    for r in records:
        by_kind[r['kind']].append(r['correct'])
    total = len(records)
    correct = sum(1 for r in records if r['correct'])
    calc_used = sum(1 for r in records if r['calc_used'])
    return {
        'total': total,
        'correct': correct,
        'overall': correct / total if total else 0.0,
        'avg_ms': int(sum(r['elapsed_ms'] for r in records) / max(total, 1)),
        'calc_rate': calc_used / total if total else 0.0,
        'by_kind': {k: {'n': len(v), 'acc': sum(v) / len(v) if v else 0.0}
                    for k, v in by_kind.items()},
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--per-type', type=int, default=40)
    ap.add_argument('--top-k', type=int, default=6)
    ap.add_argument('--regenerate', action='store_true')
    ap.add_argument('--no-agent', action='store_true', help='消融：关闭 Agent 多轮决策')
    args = ap.parse_args()
    asyncio.run(main_async(args))


if __name__ == '__main__':
    main()
