"""数据管道回归测试：每个用例对应一次真实的语料损坏。

这些 bug 的共同特点：不报错、不崩溃，只是悄悄让答案变错。
所以必须钉死。
"""
from __future__ import annotations

from backend.pipeline.crawl import parse_template_params, wikitext_to_text
from backend.pipeline.templates import template_to_text


class TestHeadingPreserved:
    """现象：问「艾米丽的服装服务」时，模型答「可换取树液、山洞萝卜、手部骨骼」。

    根因：wikitext_to_text 末尾有 `re.sub(r'^[*#:;]+\\s*', '', line)`，
    把 markdown 标题的 `## 赛跑` 也剥成了 `赛跑`，
    导致 chunker 的 `^#{1,6}\\s+` 一个都匹配不到。
    《沙漠节》整页 22 个块 section 全是「概述」，
    上一个商铺的商品表格和「艾米丽的服装服务」被硬拼进同一块，
    模型就把那些商品当成了服装服务的产出。
    """

    def test_markdown_heading_survives(self):
        md = '## 艾米丽的服装服务\n在沙漠节期间玩家可以获得新服装。'
        out = wikitext_to_text(md)
        assert '## 艾米丽的服装服务' in out

    def test_hash_levels_survive(self):
        md = '# 一级\n正文\n## 二级\n正文\n### 三级\n正文'
        out = wikitext_to_text(md)
        assert '# 一级' in out
        assert '## 二级' in out
        assert '### 三级' in out

    def test_list_markers_still_stripped(self):
        """剥列表符号的功能不能因此丢掉。"""
        md = '* 列表项\n# 编号项\n: 缩进项'
        out = wikitext_to_text(md)
        assert '* 列表项' not in out
        assert '列表项' in out

    def test_real_wiki_heading_converted(self):
        wt = '==景点==\n===赛跑===\n赛跑内容在这里，足够长的一句话。'
        out = wikitext_to_text(wt)
        assert '# 景点' in out
        assert '## 赛跑' in out


class TestSkillLevelTemplate:
    """现象：答「石鱼在 fish 级且浮漂位于 5 级水深时…」。

    根因：{{Skill level|fish|10}} 的转写只取了第一个参数，
    等级数字 10 被丢弃、技能名也没翻译，变成「fish 级」。
    全库 113 个块中招。
    """

    def test_level_number_kept(self):
        out = template_to_text('Skill level', ['fish', '10'])
        assert '10' in out
        assert 'fish' not in out.lower()

    def test_skill_name_translated(self):
        assert '钓鱼' in template_to_text('Skill level', ['fish', '10'])
        assert '战斗' in template_to_text('Skill level', ['combat', '5'])

    def test_fishing_alias(self):
        """wiki 里 fish 和 fishing 两种写法都有。"""
        assert '钓鱼' in template_to_text('Skill level', ['fishing', '10'])
        assert '钓鱼' in template_to_text('Xp', ['3', 'fish'])

    def test_no_level(self):
        assert template_to_text('Skill level', ['mining']) == '采矿'


class TestInfoboxParamLinks:
    """现象：语料里出现 `来源：[[火山地牢#火山口` 这种残缺链接（398 个块）。

    根因：parse_template_params 的值正则写成 `[^\\n|}]{0,120}`，排除了竖线。
    而链接 `[[火山地牢#火山口|火山口]]` 本身含竖线，于是在竖线处截断，
    留下半个 `[[`，后续 RE_LINK 匹配不上，清洗不掉。
    """

    def test_link_with_pipe_not_truncated(self):
        wt = '|source = [[火山地牢#火山口|火山口]]\n'
        params = parse_template_params(wt)
        assert params.get('source') == '火山口'

    def test_no_half_bracket_left(self):
        wt = '|source = [[工具#钓鱼竿|钓鱼竿]] • [[垃圾桶]]\n'
        v = parse_template_params(wt).get('source', '')
        assert '[[' not in v
        assert ']]' not in v


class TestFileLinkNesting:
    """现象：《古物》《克莱拉》等 30 篇文档残留 `]]` 碎片。

    根因：RE_FILE 用 `[^\\]]*` 匹配，遇到说明文字里的内链
    `[[File:Alex's Mum.png|thumb|克莱拉抱着[[亚历克斯]]。]]`
    会在内层 `]]` 处提前收尾，剩下 `]]。]]`。
    """

    def test_caption_with_inner_link(self):
        wt = "[[File:Alex's Mum.png|right|200px|thumb|克莱拉抱着[[亚历克斯]]。]]克莱拉是居民。"
        out = wikitext_to_text(wt)
        assert ']]' not in out
        assert '[[' not in out
        assert '克莱拉是居民' in out

    def test_plain_file_removed(self):
        wt = '[[File:CherryTree.png|thumb|一棵树]]樱桃树会结果。'
        out = wikitext_to_text(wt)
        assert 'File:' not in out
        assert '樱桃树会结果' in out


class TestStrayBrackets:
    """现象：语料残留 `被[闪电]]击中`、`[[旅行货车]处]`。

    根因：wiki 源文件本身有笔误，加上剥外链 URL 后留下半边括号。
    """

    def test_malformed_brackets_cleaned(self):
        wt = '果树被[闪电]]击中会产出煤炭，4 天后恢复。'
        out = wikitext_to_text(wt)
        assert '[' not in out
        assert ']' not in out
        assert '闪电' in out

    def test_reversed_brackets_cleaned(self):
        wt = '可以从[[旅行货车]处]以某价格购买。'
        out = wikitext_to_text(wt)
        assert '[' not in out
        assert '旅行货车' in out
