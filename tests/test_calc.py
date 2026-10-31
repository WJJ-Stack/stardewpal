"""计算工具单元测试：这些是「不让模型心算」的基础，必须钉死。"""
from __future__ import annotations

import pytest

from backend.app.tools.calc import (
    ARTISAN_MULTIPLIER,
    QUALITY_MULTIPLIER,
    call_tool,
    regrow_total,
    sell_price,
)


class TestSellPrice:
    def test_base_only(self):
        r = sell_price(35)
        assert r.value == 35

    def test_quality_multiplier(self):
        # 防风草 35 金，金星 ×1.5 = 52.5 -> floor 52
        assert sell_price(35, '金星').value == 52
        assert sell_price(35, '银星').value == 43  # 43.75 floor
        assert sell_price(35, '铱星').value == 70

    def test_artisan(self):
        # 35 × 3.0 = 105
        assert sell_price(35, artisan='果酒').value == 105

    def test_artisan_exempt_from_quality(self):
        """果酒不吃品质加成——这是 wiki 明确规则，也是最容易算错的地方。"""
        base = sell_price(35, '金星', artisan='果酒')
        plain = sell_price(35, artisan='果酒')
        assert base.value == plain.value
        assert any('不受' in s for s in base.steps)

    def test_profession_bonus(self):
        # 35 × 3.0 × 1.4 = 147
        assert sell_price(35, profession='工匠', artisan='果酒').value == 147

    def test_unknown_artisan_raises(self):
        with pytest.raises(ValueError):
            sell_price(35, artisan='不存在的加工')

    def test_formula_recorded(self):
        r = sell_price(100, '金星')
        assert '1.5' in r.formula


class TestRegrow:
    def test_no_regrow(self):
        r = regrow_total(1, 35, 4, 0)
        assert r.value == 35

    def test_regrow_counts(self):
        # 蓝莓：13 天首次，4 天再生，一季 28 天
        # 收获次数 = 1 + floor((28-13)/4) = 1 + 3 = 4
        r = regrow_total(3, 50, 13, 4, 28)
        assert r.value == 4 * 3 * 50


class TestDispatcher:
    def test_default_season_days(self):
        out = call_tool('regrow_total', {'base_yield': 1, 'price': 10,
                                         'first_growth': 4, 'regrow': 0})
        assert out['value'] == 10

    def test_unknown_tool(self):
        out = call_tool('nope', {})
        assert 'error' in out

    def test_bad_args(self):
        out = call_tool('sell_price', {'wrong': 1})
        assert 'error' in out


class TestConstants:
    def test_multipliers_sane(self):
        assert QUALITY_MULTIPLIER['普通'] == 1.0
        assert QUALITY_MULTIPLIER['铱星'] == 2.0
        assert ARTISAN_MULTIPLIER['果酒'] == 3.0
