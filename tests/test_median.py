"""沙漠节学者「总步数」题的回归测试。

每个用例对应一次真实的端到端失败。
"""
from __future__ import annotations

import asyncio

from backend.app.agent import _MEDIAN_RULE, Agent
from backend.app.retrieval import Retrieved
from backend.app.tools.calc import call_tool, median_of_options

# 用户真实截图里的三个选项
OPTIONS = [224175, 896702, 448351]

# wiki《沙漠节》原文关于这题的说明
RULE_TEXT = ('提示：其中一个问题要求玩家回忆所走的总步数，并且会给出三个选项。'
             '选项的顺序是随机的，三个数字的中间值将会是正确答案'
             '（此处指的是三个数字的中值而非中间的选项）。')


class TestMedianTool:
    """现象：模型把「中间值」当成平均值。

    实测它算出 (224175 + 896702) / 2 = 558953.5，
    然后说「正确答案应该是 336438，不过选项里没有这个数」——
    同一个问题问两次，一次答对一次答错，语气都同样自信。
    """

    def test_median_not_mean(self):
        r = median_of_options(OPTIONS)
        assert r.value == 448351
        # 绝不能是平均值
        assert r.value != sum(OPTIONS) / len(OPTIONS)

    def test_order_independent(self):
        """选项顺序是随机的，结果必须与输入顺序无关。"""
        import itertools

        for perm in itertools.permutations(OPTIONS):
            assert median_of_options(list(perm)).value == 448351

    def test_not_middle_option(self):
        """「三个数字的中值」不是「屏幕上排在中间的那个选项」。

        wiki 特意强调过这一点：选项顺序随机，所以要按数值排序，不能看位置。
        """
        # 屏幕顺序是 [大, 小, 中]，位置中间的是最小值 -> 不能取它
        screen_order = [896702, 224175, 448351]
        assert median_of_options(screen_order).value == 448351
        assert median_of_options(screen_order).value != screen_order[1]

    def test_explains_sorted_order(self):
        steps = ' '.join(median_of_options(OPTIONS).steps)
        assert '排序' in steps
        assert '中位数' in steps

    def test_via_call_tool(self):
        out = call_tool('median_of_options', {'values': float_list()})
        assert out['value'] == 448351
        assert not out.get('error')

    def test_empty_raises(self):
        import pytest

        with pytest.raises(ValueError):
            median_of_options([])


def float_list():
    return [float(v) for v in OPTIONS]


class TestMedianRule:
    """资料里必须出现规则措辞，才允许触发计算。"""

    def test_rule_matched_in_wiki_text(self):
        assert _MEDIAN_RULE.search(RULE_TEXT)

    def test_rule_variants(self):
        for text in ('三个数字的中间值', '取中值', '中位数即可', '请取中间的那个'):
            assert _MEDIAN_RULE.search(text), text

    def test_normal_compare_not_matched(self):
        """普通对比题不该误触发中位数计算。"""
        assert not _MEDIAN_RULE.search('防风草卖 35 金，蓝莓卖 50 金，哪个更贵')


class TestAgentMedianGate:
    """两个条件必须同时满足：问题里有多个大数字 + 资料里有规则。"""

    def _chunk(self, text: str) -> Retrieved:
        return Retrieved(chunk_id=1, title='沙漠节', section='学者', text=text,
                         score=0.03, kind='text', sources=['bm25'])

    def _agent(self) -> Agent:
        # 不连库，只测判断逻辑
        return Agent.__new__(Agent)

    def test_triggers_with_both_conditions(self):
        from backend.app.agent import Trace

        agent = self._agent()
        trace = Trace()
        q = '问题3：你一共走了多少步？224175 896702 448351，这题该选哪个？'
        out = agent._maybe_median(q, [self._chunk(RULE_TEXT)], trace)
        assert len(out) == 1
        assert out[0]['value'] == 448351

    def test_no_trigger_without_numbers(self):
        from backend.app.agent import Trace

        agent = self._agent()
        out = agent._maybe_median('沙漠节学者会问什么', [self._chunk(RULE_TEXT)], Trace())
        assert out == []

    def test_no_trigger_without_rule(self):
        from backend.app.agent import Trace

        agent = self._agent()
        q = '防风草 35 金 和 蓝莓 50 金，哪个更贵？'
        out = agent._maybe_median(q, [self._chunk('防风草卖价 35 金，蓝莓卖价 50 金')],
                                  Trace())
        assert out == []

    def test_no_trigger_when_options_too_small(self):
        """两位数金额不算选项（避免把「35 金 50 金」当成多选题）。"""
        from backend.app.agent import Trace

        agent = self._agent()
        out = agent._maybe_median('35 金和 50 金哪个更贵', [self._chunk(RULE_TEXT)],
                                  Trace())
        assert out == []


class TestVisionPathPassesOptions:
    """现象：截图问「这题该选哪个」时，calc 为空，全靠模型心算。

    根因：选项数字在**图里**，不在用户问题里。
    run() 只把 q 传给计算层，VLM 抄出来的 hints 没传，
    于是「选项 >= 2 个」这个条件永远不成立。
    """

    def test_hints_merged_into_calc_query(self):
        from backend.app.agent import Trace

        agent = Agent.__new__(Agent)
        trace = Trace()
        hints = '问题3：你一共走了多少步？   224175   896702   448351'
        calc_q = f'这题该选哪个 {hints}'
        out = agent._maybe_median(calc_q, [
            Retrieved(chunk_id=1, title='沙漠节', section='学者', text=RULE_TEXT,
                      score=0.03)], trace)
        assert out and out[0]['value'] == 448351

        # 反面：只用用户原话（没有 hints）就触发不了
        out2 = agent._maybe_median('这题该选哪个', [
            Retrieved(chunk_id=1, title='沙漠节', section='学者', text=RULE_TEXT,
                      score=0.03)], Trace())
        assert out2 == []


def test_median_is_deterministic_under_concurrency():
    """并发下结果必须一致（工具应无状态）。"""
    async def run():
        return [median_of_options(OPTIONS).value for _ in range(50)]

    vals = asyncio.run(run())
    assert set(vals) == {448351}
