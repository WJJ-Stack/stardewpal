"""多 provider 配置解析测试。"""
from __future__ import annotations

from unittest import mock

from backend.app.config import PROVIDER_PRESETS, Settings, resolve_provider


class TestProviderPresets:
    def test_all_presets_have_required_fields(self):
        for name, p in PROVIDER_PRESETS.items():
            assert 'base_url' in p, name
            assert 'model' in p, name
            assert 'vlm_model' in p, name

    def test_urls_are_https(self):
        for name, p in PROVIDER_PRESETS.items():
            if name == 'mock':
                continue
            assert p['base_url'].startswith('https://'), name


class TestResolveProvider:
    def _resolve(self, s):
        with mock.patch('backend.app.config.settings', s):
            return resolve_provider()

    def test_zhipu_defaults(self):
        s = Settings(llm_provider='zhipu', llm_api_key='k', llm_model='', llm_base_url='')
        cfg = self._resolve(s)
        assert cfg['base_url'].startswith('https://open.bigmodel.cn')
        assert cfg['model'] == 'glm-4-flash'
        assert cfg['api_key'] == 'k'

    def test_explicit_overrides_preset(self):
        s = Settings(llm_provider='zhipu', llm_api_key='k',
                     llm_model='glm-4-plus', llm_base_url='https://custom/v1')
        cfg = self._resolve(s)
        assert cfg['model'] == 'glm-4-plus'
        assert cfg['base_url'] == 'https://custom/v1'

    def test_unknown_provider_falls_back_to_qwen(self):
        s = Settings(llm_provider='nope', llm_api_key='k', llm_model='', llm_base_url='')
        cfg = self._resolve(s)
        assert 'dashscope' in cfg['base_url']

    def test_mock_provider_no_key_needed(self):
        s = Settings(llm_provider='mock', llm_api_key='')
        cfg = self._resolve(s)
        assert cfg['name'] == 'mock'


class TestKeyResolution:
    def test_generic_key_preferred(self):
        s = Settings(llm_api_key='new-key', dashscope_api_key='old-key')
        assert s.api_key == 'new-key'

    def test_falls_back_to_dashscope_key(self):
        s = Settings(llm_api_key='', dashscope_api_key='old-key')
        assert s.api_key == 'old-key'

    def test_has_llm_false_when_mock(self):
        s = Settings(llm_provider='mock', llm_api_key='k')
        assert s.has_llm is False

    def test_has_llm_false_without_key(self):
        # 注意：Settings 会读 .env，所以两个 key 字段都要显式清空
        s = Settings(llm_provider='zhipu', llm_api_key='', dashscope_api_key='')
        assert s.has_llm is False
