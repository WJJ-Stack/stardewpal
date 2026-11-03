"""章节识别回归测试：这一组 bug 全部是「语料看起来正常，但结构丢了」。

结构一丢，检索就给模型半截资料，模型于是自己编——
「秋季15日」、「服装节日后不保留」、「资料里没给答案」都是这么来的。
"""
from __future__ import annotations

from backend.pipeline.crawl import (
    _flatten_table,
    _is_table_header_heading,
    wikitext_to_text,
)


class TestTableHeaderNotHeading:
    """现象：检索命中《沙漠节》的题目列表，却看不到答案规则，
    模型答「资料中没有给出正确答案」。

    根因：学者小节的表格表头是 `!colspan="3" ...`，
    剥掉 `!` 和属性后残留一个 `#`，展平后成了 `# / 问题 / 答案`。
    这行被当成 markdown 一级标题，
    chunker 于是切出一个叫「/ 问题 / 答案」的假章节，
    把「题目列表」和「答题规则」分进两个 section。
    """

    def test_colspan_header_no_hash(self):
        table = ('{|class="wikitable"\n'
                 '!colspan="3" style="text-align: left;"|学者的问题和答案\n'
                 '|-\n'
                 '|鹈鹕镇的铁匠铺是谁在经营？\n'
                 '|克林特\n'
                 '|}')
        flat = _flatten_table(table)
        assert '# /' not in flat
        assert not any(line.startswith('#') for line in flat.split('\n'))

    def test_table_at_section_start_no_fake_heading(self):
        """表格紧跟在章节标题后时，不能造出假章节。"""
        wt = ('===学者===\n'
              '学者会提出四个问题。\n\n'
              '{|class="wikitable"\n'
              '!colspan="3"|问题 / 答案\n'
              '|-\n'
              '|你一共走了多少步？\n'
              '|}\n')
        out = wikitext_to_text(wt)
        assert '## 学者' in out
        assert '/ 问题 / 答案\n#' not in out
        # 「学者」之后不该再冒出别的标题
        after = out.split('## 学者', 1)[1]
        assert '#' not in after

    def test_flatten_no_leading_newline(self):
        """展平结果不以换行开头，否则会跟标题行粘连。"""
        table = '{|\n|a\n|b\n|}'
        assert not _flatten_table(table).startswith('\n')

    def test_table_header_word_detection(self):
        assert _is_table_header_heading('/ 问题 / 答案')
        assert _is_table_header_heading('图片 / 名称 / 价格')
        assert not _is_table_header_heading('艾米丽的服装服务')
        assert not _is_table_header_heading('学者')


class TestSectionStructure:
    """结构修复的端到端保证：章节必须是独立可切分的。"""

    SAMPLE = (
        '==景点==\n'
        '===村民商铺===\n'
        '村民商铺卖这些东西。\n\n'
        '{|\n|-\n|树液 (1)\n|粘土 (20)\n|}\n\n'
        '===艾米丽的服装服务===\n'
        '每天第一次进小屋，艾米丽会挑一套新服装。\n'
    )

    def test_all_headings_present(self):
        out = wikitext_to_text(self.SAMPLE)
        assert '# 景点' in out
        assert '## 村民商铺' in out
        assert '## 艾米丽的服装服务' in out

    def test_shop_items_before_clothing_heading(self):
        """商铺商品必须出现在服装小节**之前**，不能串进去。"""
        out = wikitext_to_text(self.SAMPLE)
        shop = out.find('树液')
        cloth = out.find('## 艾米丽的服装服务')
        assert 0 < shop < cloth


class TestExpandScope:
    """上下文窗口必须限制在同一章节内。

    现象：问「艾米丽的服装服务」，模型答「可获得树液、山洞萝卜、
    手部骨骼、粘土、有缺口的土罐」——那些是**村民商铺**的商品。
    根因：切块窗口跨了章节边界，把商铺表格和服装说明拼进同一块。
    """

    def test_expand_stays_in_section(self):
        from backend.app.config import settings
        from backend.app.retrieval import HybridRetriever
        from backend.app.store import Store

        store = Store(settings.db_path)
        r = HybridRetriever(store, settings.top_k_dense, settings.top_k_sparse,
                            settings.rrf_k)
        hits = r.retrieve('艾米丽的服装服务是怎么回事', top_k=6)
        if not hits:
            import pytest
            pytest.skip('索引不可用')
        chunks = r.expand(hits, query='艾米丽的服装服务是怎么回事')
        for c in chunks:
            if c.title == '沙漠节' and c.section == '艾米丽的服装服务':
                # 服装小节里不该出现村民商铺的商品名
                assert '树液' not in c.text
                assert '有缺口的土罐' not in c.text
                # 但服装本身的关键数字要在
                assert '61' in c.text
                return
        import pytest
        pytest.skip('未命中服装小节')


class TestLeadRelevance:
    """条目首段不能无条件附带。

    现象：问「艾米丽的服装服务」，首段「卡利科三花蛋节日后消失」
    挨着服装段落一起给模型，它把这句话套到服装上。
    """

    def test_lead_skipped_when_irrelevant(self):
        from backend.app.retrieval import _lead_is_relevant

        lead = '沙漠节是每年春季 15日、16日 和 17 日在沙漠举行的节日。卡利科三花蛋节日后消失。'
        assert not _lead_is_relevant('艾米丽的服装服务', lead, '沙漠节')

    def test_lead_kept_when_relevant(self):
        from backend.app.retrieval import _lead_is_relevant

        lead = '沙漠节是每年春季 15日、16日 和 17 日在沙漠举行的节日。'
        assert _lead_is_relevant('沙漠节是什么时候', lead, '沙漠节')
