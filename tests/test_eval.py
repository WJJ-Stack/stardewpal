"""评测脚本自身的正确性测试。

为什么评测代码也要测
--------------------
本项目最严重的 bug 不是系统答错，而是**评测测不出来**：
工具公式错了（腌菜漏算 +50），而评测集用同一个错误公式生成标准答案，
于是「artisan 100%」这个数字完全是假象。自洽闭环，永远通过。

那次教训的直接产物是这里的几条断言：指标本身必须能被证伪。
"""
from __future__ import annotations

import pytest

from backend.app.retrieval import Retrieved
from backend.pipeline.eval_retrieval import (
    answer_in_context,
    base_price_in_context,
    calc_correct,
    source_hit,
)


def _chunk(title: str, text: str = '', chunk_id: int = 1) -> Retrieved:
    return Retrieved(chunk_id=chunk_id, title=title, section=None,
                     text=text or title, score=0.02, kind='text')


class TestSourceHit:
    def test_rank_reported(self):
        case = {'source': '防风草'}
        hits = [_chunk('钻石'), _chunk('防风草'), _chunk('蓝莓')]
        ok, rank = source_hit(case, hits)
        assert ok and rank == 2

    def test_miss(self):
        case = {'source': '防风草'}
        ok, rank = source_hit(case, [_chunk('钻石')])
        assert not ok and rank == 0

    def test_multi_source_compare_question(self):
        """对比题的 source 是「A|B」，命中任一即算命中。"""
        case = {'source': '钻石|防风草'}
        ok, _ = source_hit(case, [_chunk('防风草')])
        assert ok

    def test_no_source_is_not_hit(self):
        assert source_hit({'source': ''}, [_chunk('任意')]) == (False, 0)


class TestAnswerInContextSkipsComputed:
    """计算题的答案是算出来的，不能用「字面在上下文」衡量。

    金星啤酒花 = 25 × 1.5 = 37，而 wiki 里只有 25，没有 37。
    早期对这类题套用「答案是否在上下文」，得到 61.7% 的假阴性
    （22/60 条全部是计算题），指标错了而不是系统错了。
    """

    def test_quality_returns_none(self):
        case = {'kind': 'quality', 'expected': '37 金'}
        assert answer_in_context(case, [_chunk('啤酒花', '卖价：25 金')]) is None

    def test_artisan_returns_none(self):
        case = {'kind': 'artisan', 'expected': '120 金'}
        assert answer_in_context(case, [_chunk('防风草', '卖价：35 金')]) is None

    def test_fact_is_measured(self):
        case = {'kind': 'fact', 'expected': '35 金'}
        assert answer_in_context(case, [_chunk('防风草', '卖价：35 金')]) is True

    def test_fact_missing(self):
        case = {'kind': 'fact', 'expected': '999 金'}
        assert answer_in_context(case, [_chunk('防风草', '卖价：35 金')]) is False


class TestCalcCorrect:
    """计算题端到端：基础价能取到 + 工具算对。

    这是「不让模型心算」这条设计的直接验证，全程不调 LLM。
    """

    def test_quality_computed(self):
        # 防风草 35 × 1.5 = 52（floor）
        case = {'kind': 'quality', 'question': '金星防风草卖多少钱？',
                'expected': '52 金', 'numeric': 52.0, 'source': '防风草'}
        chunks = [_chunk('防风草', '【防风草 · 属性数据】 卖价：35 金')]
        assert calc_correct(case, chunks) is True

    def test_pickles_flat_bonus(self):
        """腌菜 = 基础价 × 2 + 50。这条曾经漏算，是项目最严重的一次错误。"""
        case = {'kind': 'artisan', 'question': '防风草做成腌菜能卖多少钱？',
                'expected': '120 金', 'numeric': 120.0, 'source': '防风草'}
        chunks = [_chunk('防风草', '【防风草 · 属性数据】 卖价：35 金')]
        assert calc_correct(case, chunks) is True

    def test_jam_flat_bonus(self):
        """果酱 = 水果基础价 × 2 + 50。"""
        case = {'kind': 'artisan', 'question': '蓝莓做成果酱能卖多少钱？',
                'expected': '210 金', 'numeric': 210.0, 'source': '蓝莓'}
        chunks = [_chunk('蓝莓', '【蓝莓 · 属性数据】 卖价：80 金')]
        assert calc_correct(case, chunks) is True

    def test_wrong_formula_is_caught(self):
        """如果把腌菜当纯 ×2（漏掉 +50），必须判错。

        这是防止「自洽闭环」的关键断言：
        指标必须能证伪一个错误的实现。
        """
        case = {'kind': 'artisan', 'question': '防风草做成腌菜能卖多少钱？',
                'expected': '70 金',   # 错误的标准答案（纯 ×2）
                'numeric': 70.0, 'source': '防风草'}
        chunks = [_chunk('防风草', '【防风草 · 属性数据】 卖价：35 金')]
        assert calc_correct(case, chunks) is False

    def test_missing_base_price(self):
        """检索没给到基础价时，不能假装算对。"""
        case = {'kind': 'quality', 'question': '金星防风草卖多少钱？',
                'expected': '52 金', 'numeric': 52.0, 'source': '防风草'}
        assert calc_correct(case, [_chunk('钻石', '卖价：999 金')]) is False

    def test_non_computed_kind_returns_none(self):
        assert calc_correct({'kind': 'fact', 'numeric': 35.0}, []) is None


class TestBasePriceExtraction:
    def test_skips_minus_one(self):
        """wiki 用 -1 表示「无固定售价」，不能当成基础价。"""
        chunks = [_chunk('测试', '卖价：-1 卖价：350 金')]
        case = {'source': '测试'}
        assert base_price_in_context(case, chunks) == 350.0

    def test_wrong_doc_not_used(self):
        case = {'source': '防风草'}
        assert base_price_in_context(case, [_chunk('钻石', '卖价：999 金')]) is None


class TestEvalSetNotSelfAligned:
    """评测集的标准答案不能来自被测代码。

    这是原则性检查：如果 eval_set 直接 import 被测的计算函数来造答案，
    工具算错时评测永远通过。
    """

    def test_generator_uses_wiki_fields_not_hardcoded(self):
        """基础价必须取自 items 表（wiki 数据），不是写死的常量。"""
        import inspect

        from backend.pipeline import eval_set

        src = inspect.getsource(eval_set.generate)
        assert 'price_of' in src
        assert "fields.get('sellprice')" in src or "'sellprice'" in src

    def test_scorer_can_fail(self):
        """评分器必须能判错。"""
        from backend.pipeline.eval_set import score_answer

        case = {'kind': 'fact', 'expected': '35 金', 'numeric': 35.0}
        assert score_answer(case, '售价是 35 金')['correct'] is True
        assert score_answer(case, '售价是 999 金')['correct'] is False

    def test_refusal_scorer_rejects_invented_number(self):
        from backend.pipeline.eval_set import score_answer

        case = {'kind': 'refusal', 'expected': '未提及', 'numeric': None}
        assert score_answer(case, '资料中未提及该内容。')['correct'] is True
        # 拒答同时编了数字 → 判错
        assert score_answer(case, '资料中未提及，但耐久度是 1561 次。')['correct'] is False


@pytest.mark.parametrize('kind', ['fact', 'attr', 'compare', 'quality', 'artisan'])
def test_all_kinds_have_defined_measurement(kind):
    """每种题型都必须有明确的衡量方式，不能悄悄被跳过。"""
    literal = answer_in_context({'kind': kind, 'expected': '35 金'}, [])
    computed = calc_correct({'kind': kind, 'numeric': 35.0}, [])
    assert literal is not None or computed is not None
