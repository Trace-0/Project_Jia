#!/usr/bin/env python3
"""Project Jia 1.2.0(beta) smoke/regression tests.

This suite intentionally avoids Discord login, LLM calls, ComfyUI calls, and
network access. It covers pure functions and static integration markers for the
1.2.0(beta) feature set documented in README.md.

Run from the project root:
    python scripts/test_1_2_0_beta.py
"""

from __future__ import annotations

import contextlib
import importlib
import re
import sys
import tempfile
import types
import unittest
from pathlib import Path
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# The local venv launcher can be broken after moving machines, but pure-Python
# packages such as tomlkit can still be imported from site-packages.
VENV_SITE = ROOT / "venv" / "Lib" / "site-packages"
if VENV_SITE.exists() and str(VENV_SITE) not in sys.path:
    sys.path.append(str(VENV_SITE))


_MISSING = object()


def read_text(relative_path: str) -> str:
    return (ROOT / relative_path).read_text(encoding="utf-8")


def make_module(name: str, **attrs):
    module = types.ModuleType(name)
    for key, value in attrs.items():
        setattr(module, key, value)
    return module


@contextlib.contextmanager
def patched_modules(patches: dict[str, types.ModuleType]):
    old_values = {name: sys.modules.get(name, _MISSING) for name in patches}
    sys.modules.update(patches)
    try:
        yield
    finally:
        for name, old_value in old_values.items():
            if old_value is _MISSING:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = old_value


def import_fresh(module_name: str):
    sys.modules.pop(module_name, None)
    return importlib.import_module(module_name)


class Beta120StaticTests(unittest.TestCase):
    def test_readme_has_current_beta_items(self):
        readme = read_text("README.md")
        section = readme.split("## 1.2.0 (beta)", 1)[1].split("---", 1)[0]
        numbered_items = re.findall(r"^\d+\. ", section, flags=re.MULTILINE)
        self.assertGreaterEqual(len(numbered_items), 38)
        for marker in [
            "발화를 길드별로 모아",
            "settings.toml",
            "MCP 서버",
            "ComfyUI 이미지 생성",
            "`/jiamusic` URL 입력 제한",
            "`description`과 `use_when`",
        ]:
            self.assertIn(marker, section)

    def test_voice_batching_and_interrupt_markers_exist(self):
        pipeline = read_text("discord_interface/pipeline.py")
        discord_bot = read_text("discord_interface/discordBot.py")
        llm = read_text("LLM/langchain_llm.py")

        for marker in [
            "_pending_utterances",
            "_conversation_worker",
            "_respond_to_batch",
            "_get_voice_participants",
            "PROACTIVE_SENTINEL",
            "mark_playback_interrupted",
            "clear_pending_utterances",
            "maybe_play_auto_reaction",
        ]:
            self.assertIn(marker, pipeline)
        for marker in [
            "voice_interrupt_speech_sec",
            "_interrupt_playback",
            "pipeline.stop_foreground_audio",
            "pipeline.mark_playback_interrupted",
            "pipeline.clear_pending_utterances",
            "pipeline.start_proactive_monitor",
        ]:
            self.assertIn(marker, discord_bot)
        for marker in [
            "def _build_voice_input",
            "participants",
            "interrupted",
            "proactive",
            "다른 말은 하지 말고",
        ]:
            self.assertIn(marker, llm)

    def test_memory_schema_and_forgetting_markers_exist(self):
        rag = read_text("memory/RAG.py")
        for marker in [
            "user_profiles",
            "memory_optout",
            "last_accessed",
            "apply_forgetting",
            "rag_forget_decay_per_day",
            "rag_retrieval_boost",
            "get_profile_facts",
            "set_optout",
        ]:
            self.assertIn(marker, rag)

    def test_command_permission_and_security_markers_exist(self):
        discord_bot = read_text("discord_interface/discordBot.py")
        config = read_text("config/config.py")
        for marker in [
            "require_command_access",
            "is_command_whitelisted",
            'require_command_access(ctx, "owner", "/jiareload")',
            'require_command_access(ctx, "admin", "/jiahear")',
            'action in {"list", "search", "delete", "profile"}',
            "allow_unsafe_jiaplay",
            "music_allowed_url_hosts",
        ]:
            self.assertIn(marker, discord_bot + config)

    def test_llm_runtime_keeps_models_and_avoids_nested_event_loop(self):
        langchain_llm = read_text("LLM/langchain_llm.py")
        model_control = read_text("LLM/LLM_model_control.py")
        importance = read_text("memory/calculate_importance.py")

        self.assertIn("asyncio.run_coroutine_threadsafe", langchain_llm)
        self.assertNotIn("run_until_complete", langchain_llm)
        self.assertIn('kwargs["keep_alive"] = -1', model_control)
        self.assertIn('kwargs["num_ctx"] = config.llmNumCtx', model_control)
        self.assertIn('options={"num_ctx": config.llmNumCtx}', model_control)
        self.assertIn("get_summary_chat_model", importance)

    def test_soundboard_volume_runtime_markers_exist(self):
        soundboard = read_text("LLM/langchain_tools/soundboard.py")
        pipeline = read_text("discord_interface/pipeline.py")
        discord_bot = read_text("discord_interface/discordBot.py")

        self.assertIn("volume: float = 1.0", soundboard)
        self.assertIn("volume=_clamp_volume", soundboard)
        self.assertIn("volume=entry.volume", soundboard)
        self.assertIn("_jia_volume", pipeline)
        self.assertIn("foreground_gain", discord_bot)


class Beta120ConfigTests(unittest.TestCase):
    def test_config_defaults_include_beta_settings(self):
        config_module = import_fresh("config.config")
        cfg = config_module.Config()
        self.assertEqual(cfg.music_allowed_url_hosts, ["youtube.com", "youtu.be"])
        self.assertIn("command_whitelist_user_ids", config_module.FIELD_TO_TOML)
        self.assertIn("llm_tools", config_module.FIELD_TO_TOML)
        default_mcp = config_module.DEFAULT_MCP_SERVERS()["ddg-search"]
        self.assertIn("description", default_mcp)
        self.assertIn("use_when", default_mcp)


class Beta120McpTests(unittest.TestCase):
    def test_mcp_metadata_is_removed_from_client_spec_and_added_to_guidance(self):
        class FakeMultiServerMCPClient:
            last_servers = None

            def __init__(self, servers):
                FakeMultiServerMCPClient.last_servers = servers

        fake_config = SimpleNamespace(
            llm_tools={
                "orders": {
                    "url": "http://localhost:9000/mcp",
                    "transport": "streamable_http",
                    "description": "주문과 재고를 조회할 수 있습니다.",
                    "use_when": "주문 상태나 재고 확인을 부탁할 때 사용합니다.",
                }
            }
        )
        patches = {
            "langchain_mcp_adapters": make_module("langchain_mcp_adapters"),
            "langchain_mcp_adapters.client": make_module(
                "langchain_mcp_adapters.client",
                MultiServerMCPClient=FakeMultiServerMCPClient,
            ),
            "config.config_manager": make_module("config.config_manager", config=fake_config),
        }
        with patched_modules(patches):
            manager = import_fresh("LLM.langchain_tools.mcp_manager")
            self.assertEqual(
                FakeMultiServerMCPClient.last_servers,
                {"orders": {"url": "http://localhost:9000/mcp", "transport": "streamable_http"}},
            )
            guidance = manager.get_mcp_usage_guidance()
            self.assertIn("orders", guidance)
            self.assertIn("주문과 재고", guidance)
            self.assertNotIn("description", manager._client_spec(fake_config.llm_tools["orders"]))
            self.assertNotIn("use_when", manager._client_spec(fake_config.llm_tools["orders"]))


class Beta120MusicTests(unittest.TestCase):
    def test_jiamusic_url_guard_allows_youtube_and_blocks_other_urls(self):
        music = import_fresh("discord_interface.youtube_music")
        hosts = ["youtube.com", "youtu.be"]
        self.assertEqual(
            music.validate_music_query("www.youtube.com/watch?v=abc", hosts),
            "https://www.youtube.com/watch?v=abc",
        )
        self.assertEqual(
            music.validate_music_query("youtube.com/watch?v=abc", hosts),
            "https://youtube.com/watch?v=abc",
        )
        self.assertEqual(music.validate_music_query("lofi hip hop", hosts), "lofi hip hop")
        self.assertTrue(music.is_allowed_music_url("https://music.youtube.com/watch?v=abc", hosts))
        for blocked in [
            "https://example.com/video",
            "file:///C:/secret.mp3",
            "http://127.0.0.1:8000/audio",
            "https://youtube.com.evil.test/watch?v=abc",
        ]:
            with self.subTest(blocked=blocked):
                with self.assertRaises(ValueError):
                    music.validate_music_query(blocked, hosts)
        with self.assertRaises(RuntimeError):
            music.resolve_music_track(
                music.MusicTrack(title="bad", webpage_url="https://example.com/video", stream_url="https://stream"),
                hosts,
            )


class Beta120SoundboardTests(unittest.TestCase):
    def test_soundboard_registry_adds_new_audio_files_to_toml(self):
        class FakeTool:
            def __init__(self, *args, **kwargs):
                self.args = args
                self.kwargs = kwargs

        fake_config = SimpleNamespace(
            soundboard_auto_react=False,
            soundboard_auto_react_chance=0.35,
            soundboard_auto_react_cooldown_sec=20,
        )
        patches = {
            "langchain_core": make_module("langchain_core"),
            "langchain_core.tools": make_module("langchain_core.tools", Tool=FakeTool),
            "config.config_manager": make_module("config.config_manager", config=fake_config),
        }
        with tempfile.TemporaryDirectory() as tmpdir, patched_modules(patches):
            soundboard_dir = Path(tmpdir) / "soundboard"
            soundboard_dir.mkdir()
            (soundboard_dir / "drop in.mp3").write_bytes(b"")

            soundboard = import_fresh("LLM.langchain_tools.soundboard")
            soundboard.SOUNDBOARD_DIR = soundboard_dir
            soundboard.REGISTRY_PATH = soundboard_dir / "sounds.toml"

            sounds = soundboard.load_sound_registry()
            self.assertIn("drop in.mp3", sounds)
            registry_text = soundboard.REGISTRY_PATH.read_text(encoding="utf-8")
            self.assertIn('"drop in.mp3"', registry_text)
            self.assertIn("volume = 1.0", registry_text)

            version = soundboard.get_sound_registry_version()
            (soundboard_dir / "second.wav").write_bytes(b"")
            sounds = soundboard.load_sound_registry()
            self.assertIn("second.wav", sounds)
            self.assertGreater(soundboard.get_sound_registry_version(), version)

    def test_soundboard_registry_scoring_and_path_guard(self):
        class FakeTool:
            def __init__(self, *args, **kwargs):
                self.args = args
                self.kwargs = kwargs

        fake_config = SimpleNamespace(
            soundboard_auto_react=False,
            soundboard_auto_react_chance=0.35,
            soundboard_auto_react_cooldown_sec=20,
        )
        patches = {
            "langchain_core": make_module("langchain_core"),
            "langchain_core.tools": make_module("langchain_core.tools", Tool=FakeTool),
            "config.config_manager": make_module("config.config_manager", config=fake_config),
        }
        with patched_modules(patches):
            soundboard = import_fresh("LLM.langchain_tools.soundboard")
            entry = soundboard._entry_from_toml(
                "tada.mp3",
                {"desc": "축하 팡파레", "tags": ["celebrate"], "chance": 2, "auto": False},
            )
            self.assertEqual(entry.tags, ("celebrate",))
            self.assertEqual(entry.chance, 1.0)
            self.assertFalse(entry.auto)
            self.assertEqual(soundboard._entry_from_toml("quiet.mp3", {"volume": 0.4}).volume, 0.4)
            self.assertEqual(soundboard._entry_from_toml("loud.mp3", {"volume": 2}).volume, 1.0)
            self.assertGreater(soundboard._score_auto_reaction(entry, "생일 축하해 성공했다"), 0)
            self.assertIsNone(soundboard._resolve_sound_path("../secret.mp3"))
            self.assertIsNone(soundboard._resolve_sound_path("not-a-sound.txt"))


class Beta120ComfyUiTests(unittest.TestCase):
    def test_comfyui_model_profile_helpers(self):
        class FakeStructuredTool:
            @classmethod
            def from_function(cls, **kwargs):
                return kwargs

        class FakeBaseModel:
            pass

        def fake_field(default=None, **_kwargs):
            return default

        fake_config = SimpleNamespace(
            comfyui_checkpoint="base.safetensors",
            comfyui_steps=20,
            comfyui_cfg=7.0,
            comfyui_width=1024,
            comfyui_height=1024,
            comfyui_sampler="euler",
            comfyui_scheduler="normal",
            comfyui_negative_prompt="bad quality",
            comfyui_url="",
            comfyui_timeout_sec=120,
        )
        patches = {
            "langchain_core": make_module("langchain_core"),
            "langchain_core.tools": make_module("langchain_core.tools", StructuredTool=FakeStructuredTool),
            "pydantic": make_module("pydantic", BaseModel=FakeBaseModel, Field=fake_field),
            "requests": make_module("requests", ConnectionError=ConnectionError),
            "config.config_manager": make_module("config.config_manager", config=fake_config),
            "config.config": make_module("config.config", SETTINGS_PATH=ROOT / "__missing_settings_for_test__.toml"),
        }
        with patched_modules(patches):
            comfy = import_fresh("LLM.langchain_tools.comfyui_image")
            profile = comfy._profile_from_mapping(
                "anime",
                {
                    "checkpoint": "anime.safetensors",
                    "use_when": "애니풍 인물",
                    "tags": "anime, character",
                    "steps": "4",
                    "cfg": "1.0",
                },
            )
            self.assertIsNotNone(profile)
            self.assertEqual(profile.model_id, "anime")
            self.assertEqual(profile.tags, ("anime", "character"))
            self.assertEqual(profile.steps, 4)
            self.assertEqual(profile.cfg, 1.0)
            formatted = comfy._format_model_profiles({"anime": profile})
            self.assertIn("anime", formatted)
            self.assertIn("anime.safetensors", formatted)
            workflow = comfy._build_workflow("a cat", 123, profile)
            self.assertEqual(workflow["4"]["inputs"]["ckpt_name"], "anime.safetensors")
            self.assertEqual(workflow["3"]["inputs"]["steps"], 4)


if __name__ == "__main__":
    unittest.main(verbosity=2)
