"""检索与清洗单元测试：这些逻辑直接决定回答质量。"""
from __future__ import annotations

from backend.app.retrieval import (_core_entity, _title_affinity, classify,
                                   expand_query)
from backend.pipeline.chunker import chunk_text, numeric_facts


class TestQueryExpansion:
    def test_synonym_injection(self):
        assert '卖价' in expand_query('防风草卖多少钱')

    def test_no_change_when_no_match(self):
        q = '这个游戏怎么玩'
        assert expand_query(q) == q

    def test_classify_numeric(self):
        assert classify('防风草卖多少钱')['numeric'] is True
        assert classify('金星蓝莓值多少')['calc'] is True

    def test_classify_non_numeric(self):
        assert classify('你好')['numeric'] is False


class TestTitleAffinity:
    def test_exact_entity_boosted(self):
        # 问防风草时，《防风草》应比《防风草种子》得分高
        exact = _title_affinity('防风草卖多少钱', '防风草')
        derived = _title_affinity('防风草卖多少钱', '防风草种子')
        assert exact > 1.0
        assert derived < 1.0
        assert exact > derived

    def test_unrelated_neutral(self):
        assert _title_affinity('防风草卖多少', '钻石') == 1.0


class TestCoreEntity:
    def test_strip_question_words(self):
        assert '防风草' in _core_entity('防风草卖多少钱')
        assert '多少' not in _core_entity('防风草卖多少钱')


class TestNumericFacts:
    def test_unit_annotation(self):
        f = numeric_facts({'sellprice': '35', 'growth': '4'})
        assert '卖价：35 金' in f
        assert '生长时间：4 天' in f

    def test_skip_noise_fields(self):
        f = numeric_facts({'color': 'yellow', 'class': 'inline', 'sellprice': '35'})
        assert 'color' not in f
        assert '卖价' in f

    def test_empty(self):
        assert numeric_facts({}) == ''
        assert numeric_facts(None) == ''


class TestChunker:
    def test_infobox_chunk_created(self):
        chunks = chunk_text('防风草', '防风草是一种蔬菜。', {'sellprice': '35'})
        kinds = [c['kind'] for c in chunks]
        assert 'infobox' in kinds

    def test_numeric_injected_into_body(self):
        """数值必须进入正文块，否则语义检索命中不到。"""
        chunks = chunk_text('防风草', '防风草是一种蔬菜，可以种。' * 4, {'sellprice': '35'})
        body = [c for c in chunks if c['kind'] == 'text']
        assert body
        assert '卖价' in body[0]['text']

    def test_long_text_split(self):
        text = '这是很长的段落。' * 200
        chunks = chunk_text('测试', text)
        assert len(chunks) > 1
        for c in chunks:
            assert c['token_len'] <= 700  # 允许重叠导致略超

    def test_noise_lines_removed(self):
        text = '正常内容在这里。\n；\n。\n另一段正常内容。'
        chunks = chunk_text('测试', text)
        joined = ''.join(c['text'] for c in chunks)
        assert '；\n' not in joined
