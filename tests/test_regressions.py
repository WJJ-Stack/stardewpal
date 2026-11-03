"""回归测试：真实模型评测中发现的问题，必须钉死。

每个测试对应一次真实失败，注释里写明现象与根因。
"""
from __future__ import annotations

from backend.app.agent import OTHER_DOMAIN, _mentions_other_domain
from backend.app.retrieval import extract_entities
from backend.pipeline.eval_set import _has_invented_number, score_answer


class TestSingleCharEntity:
    """现象：问「蒜要几天才能成熟」检索不到《蒜》。

    根因：单字实体（蒜）在实体提取环节被当成噪声滤掉。
    现在 extract_entities 保留单字实词（另设 SINGLE_CHAR_NOISE 过滤「成熟」这类碎片），
    BM25 查询也因此能带上「蒜」。
    """

    def test_single_char_entity_extracted(self):
        assert '蒜' in extract_entities('蒜要几天才能成熟？')

    def test_single_char_noise_filtered(self):
        """动词/副词碎片不该被当成实体。"""
        ents = extract_entities('蒜要几天才能成熟')
        assert '熟' not in ents
        assert '成' not in ents

    def test_single_char_still_not_other_domain(self):
        assert not _mentions_other_domain('蒜要几天成熟？')



class TestRefusalScoring:
    """现象：模型正确拒答「原神胡桃倍率」，被判为「编造数值」。

    根因：评分器把引用角标 [1–17] 里的数字当成编造的事实数值。
    """

    def test_citation_numbers_are_not_invented_values(self):
        answer = ('资料中未提及《原神》及胡桃的元素爆发倍率。'
                  '所有提供的资料均来自《星露谷物语》，不含《原神》数据[1–17]。')
        assert _has_invented_number(answer) is False

    def test_multiple_citation_brackets(self):
        answer = '资料中未提及该内容[1][2][3]。'
        assert _has_invented_number(answer) is False

    def test_real_invented_number_detected(self):
        # 这是真正需要抓的情况：拒答词 + 具体数值并存
        answer = '资料中未提及，但钻石镐耐久度是 1561 次。'
        assert _has_invented_number(answer) is True

    def test_percentage_not_treated_as_invented(self):
        answer = '资料中未提及具体倍率，可能约为 150%。'
        assert _has_invented_number(answer) is False

    def test_score_refusal_correct(self):
        case = {'kind': 'refusal', 'expected': '未提及', 'numeric': None}
        ans = '资料中未提及《原神》内容，无法回答[1–5]。'
        r = score_answer(case, ans)
        assert r['correct'] is True


class TestCrossDomainBlocking:
    """跨域问题应强制检索 + 门控拒答，不能凭模型记忆作答。"""

    def test_other_game_mentioned(self):
        for q in ('原神里胡桃的元素爆发倍率是多少',
                  '我的世界钻石镐耐久度是多少',
                  '明日方舟能天使的攻击力是多少'):
            assert _mentions_other_domain(q), q

    def test_stardew_question_not_blocked(self):
        for q in ('防风草卖多少钱', '杨桃要几天成熟', '蒜怎么获得'):
            assert not _mentions_other_domain(q), q

    def test_cross_domain_word_registered(self):
        """库里确实有「胡桃木茶几」，所以「胡桃」必须登记为跨域词，
        否则字面命中会被当成星露谷内容。"""
        assert '胡桃' in OTHER_DOMAIN
        assert '钻石镐' in OTHER_DOMAIN


class TestEntityExtraction:
    def test_comparison_splits_entities(self):
        ents = extract_entities('钻石和防风草哪个卖得更贵？')
        assert '钻石' in ents
        assert '防风草' in ents

    def test_single_entity(self):
        ents = extract_entities('防风草卖多少钱')
        assert ents and ents[0] == '防风草'
