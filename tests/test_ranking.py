"""检索层回归测试：这些逻辑决定「wiki 明明有答案却答错」。

每个用例对应一次真实的端到端失败，注释写明现象与根因。
"""
from __future__ import annotations

from backend.app.retrieval import _coverage_factor, _merge_ranked, _sparse_query, extract_entities


class TestSparseQuery:
    """现象：问「沙漠节什么时候办」，检索第 1 名是《版本历史/1.6》，
    《沙漠节》连前 10 都进不去，模型于是编造「每年秋季15日」。

    根因：BM25 用的 FTS 查询把整句切成双字前缀并用 OR 连接，
    「什么」「么时」「时候」这类虚词在语料里到处都是，
    OR 一放就主导了排序。压成实体串后立刻回到第 1。
    """

    def test_question_words_dropped(self):
        q = _sparse_query('沙漠节什么时候办')
        assert '沙漠节' in q
        assert '什么' not in q
        assert '时候' not in q

    def test_keeps_single_char_entity(self):
        """「鸡」这种单字物品名不能被当成噪声扔掉。

        早期为了「减少噪声」丢弃单字实体，
        「鸡的售价是多少」变成只查「售价多少」，
        命中《图腾雕像》《化肥》等含「售价」的页，《鸡》根本进不来。
        """
        q = _sparse_query('鸡的售价是多少')
        assert '鸡' in q

    def test_fallback_when_no_entity(self):
        assert _sparse_query('你好')  # 不能返回空串


class TestCoverageFactor:
    """现象：问「石鱼在哪钓」，前 5 名全是覆盖率 0 的页面
    （《冬季》《大树桩（地点）》《深山》），《石鱼》排第 6。

    根因：OR 连接下只要命中一个通用词就能进候选，
    短文档 + 高频词的 BM25 分数反而更高。
    """

    def test_full_coverage_no_penalty(self):
        assert _coverage_factor('防风草卖多少钱', '防风草是一种作物，卖价 35 金', '防风草') == 1.0

    def test_zero_coverage_penalised(self):
        f = _coverage_factor('石鱼在哪钓', '冬天很冷，树木凋零', '冬季')
        assert f < 0.6

    def test_partial_coverage_between(self):
        full = _coverage_factor('石鱼在哪钓', '石鱼在哪钓都可以', '石鱼')
        none = _coverage_factor('石鱼在哪钓', '冬天很冷', '冬季')
        assert none < full

    def test_question_words_not_counted(self):
        """覆盖率只统计实词。

        否则「多少」「在哪」这类词会把所有页面拉成同一档，
        覆盖率失去区分意义（实测 8 个候选全是 cov=0.14）。
        """
        # 问题词「在哪」不出现在正文里，但不应因此扣分
        f = _coverage_factor('石鱼在哪钓', '石鱼可以在矿井 20 层钓到', '石鱼')
        assert f == 1.0


class TestMergeRanked:
    """双路 BM25 融合：整句一路 + 实体一路，按 RRF 合并。"""

    def _row(self, cid: int, score: float = 1.0) -> dict:
        return {'chunk_id': cid, 'title': f't{cid}', 'section': None,
                'text': f'text{cid}', 'score': score}

    def test_dedup_keeps_single_entry(self):
        a = [self._row(1), self._row(2)]
        b = [self._row(2), self._row(3)]
        out = _merge_ranked([(a, 1.0), (b, 1.0)])
        ids = [r['chunk_id'] for r in out]
        assert ids.count(2) == 1

    def test_hit_in_both_sources_ranks_higher(self):
        """两路都命中的应该排在只命中一路的前面。"""
        a = [self._row(1), self._row(2)]
        b = [self._row(2), self._row(9)]
        out = _merge_ranked([(a, 1.0), (b, 1.0)])
        assert out[0]['chunk_id'] == 2

    def test_empty_input(self):
        assert _merge_ranked([([], 1.0)]) == []


class TestEntitySplitting:
    def test_time_question_split(self):
        """「什么时候」要能切开，否则整个短语被当成一个实体。"""
        ents = extract_entities('沙漠节什么时候办')
        assert '沙漠节' in ents

    def test_keeps_item_name(self):
        ents = extract_entities('防风草卖多少钱')
        assert '防风草' in ents
