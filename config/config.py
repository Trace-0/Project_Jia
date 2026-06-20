import json
import os
import re
import threading
import time
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Type
import logging

import tomlkit
from dotenv import dotenv_values

# 모든 설정은 프로젝트 루트의 settings.toml에 저장됩니다.
# .env는 LangSmith 추적 같은 개발용 키 전용입니다. (langchain이 환경 변수로 직접 읽으며, 일반 사용자는 필요 없음)
ROOT_DIR = Path(__file__).resolve().parents[1]
SETTINGS_PATH = ROOT_DIR / "settings.toml"
ENV_PATH = ROOT_DIR / ".env"
ENV_PREFIX = "JIA_"

def DEFAULT_MCP_SERVERS() -> dict:
    """기본으로 연결하는 MCP 서버 구성 (langchain-mcp-adapters의 MultiServerMCPClient 형식)"""
    return {
        "ddg-search": {
            "command": "uvx",
            "args": ["duckduckgo-mcp-server"],
            "transport": "stdio",
        }
    }

# Config 필드 -> settings.toml의 {섹션: {필드명: (키, 주석)}} 매핑
TOML_LAYOUT: dict[str, dict[str, tuple[str, str]]] = {
    "bot": {
        "bot_token": ("token", "디스코드 봇 토큰 (변경 시 재시작 필요)"),
        "debug_text_channel": ("debug_text_channel", "로그를 보낼 디버그 텍스트 채널 ID (0이면 사용 안 함, 변경 시 재시작 필요)"),
        "join_reply": ("join_reply", "음성 채널 접속 시 안내 메시지 전송 여부"),
        "leave_reply": ("leave_reply", "음성 채널 퇴장 시 안내 메시지 전송 여부"),
    },
    "voice": {
        "voice_timeout_sec": ("timeout_sec", "마지막 음성 패킷 이후 이 시간 동안 패킷이 없으면 발화 종료로 판단 (초)"),
        "voice_interrupt_speech_sec": ("interrupt_speech_sec", "재생 중단(barge-in) 판정에 필요한 연속 발화 시간 (초)"),
        "proactive_idle_sec": ("proactive_idle_sec", "음성 채널에서 이 시간(초) 동안 아무도 말이 없으면 지아가 먼저 말을 걸어봄 (0이면 사용 안 함)"),
    },
    "soundboard": {
        "soundboard_auto_react": ("auto_react", "대화 상황에 맞는 효과음을 자동으로 재생할지 여부"),
        "soundboard_auto_react_cooldown_sec": ("auto_react_cooldown_sec", "같은 효과음 자동 재생 사이 최소 간격 (초)"),
        "soundboard_auto_react_chance": ("auto_react_chance", "자동 반응 후보가 잡혔을 때 실제로 재생할 확률 (0.0~1.0)"),
    },
    "music": {
        "music_volume": ("volume", "음악 기본 볼륨 (0.0~1.0)"),
        "music_duck_volume": ("duck_volume", "지아가 말하거나 효과음을 재생할 때 낮출 음악 볼륨 (0.0~1.0)"),
        "music_max_playlist_items": ("max_playlist_items", "유튜브 재생목록에서 한 번에 추가할 최대 곡 수"),
    },
    "security": {
        "allow_unsafe_jiaplay": (
            "allow_unsafe_jiaplay",
            "위험 기능: /jiaplay 로컬 파일 재생을 허용합니다.\n"
            "이 값을 true로 바꾸면 Discord 명령을 입력할 수 있는 사람이 봇 PC에서 봇 프로세스 권한으로 읽을 수 있는 로컬 오디오 파일 경로를 지정할 수 있습니다.\n"
            "사적인 오디오가 음성 채널에 노출될 수 있고, 큰 파일을 메모리에 읽어 봇이 느려지거나 중단될 수 있으며, 신뢰할 수 없는 미디어를 FFmpeg가 해석하면서 디코더 취약점이나 비정상 파일 오류에 노출될 수 있습니다.\n"
            "개인 서버나 완전히 신뢰하는 사용자만 명령을 사용할 수 있는 환경에서만 true로 바꾸세요. 일반 음악 재생은 /jiamusic을 권장합니다.",
        ),
    },
    "vad": {
        "vad_threshold": ("threshold", "발화로 판정할 확률 임계값 (0.0~1.0, 낮을수록 민감)"),
        "vad_min_speech_ms": ("min_speech_ms", "이보다 짧은 발화 구간은 무시 (ms)"),
        "vad_min_silence_ms": ("min_silence_ms", "발화 구간 분리에 필요한 최소 무음 시간 (ms)"),
        "vad_max_speech_sec": ("max_speech_sec", "발화 구간 하나의 최대 길이 (초)"),
        "vad_padding_ms": ("padding_ms", "발화 구간 시작 앞에 붙이는 여유 시간 (ms)"),
    },
    "whisper": {
        "whisper_model": ("model", "STT(음성 인식) 모델 (변경 시 자동 재로딩)"),
        "whisper_device": ("device", "Whisper 실행 디바이스 (변경 시 자동 재로딩)"),
        "whisper_compute_type": ("compute_type", "Whisper 연산 정밀도 (변경 시 자동 재로딩)"),
        "whisper_beam_size": ("beam_size", "STT beam size (클수록 정확하지만 느려짐)"),
    },
    "tts": {
        "tts_model": ("model", "TTS(음성 합성) 모델 경로 (변경 시 자동 재로딩)"),
    },
    "llm": {
        "llmModel": ("model", "LLM 모델 이름. provider가 ollama면 Ollama 모델, 외부 API면 그 API의 모델 이름 (변경 시 자동 재로딩)"),
        "llm_provider": ("provider", "LLM 제공자: ollama(기본, 로컬) 또는 openai/anthropic/google_genai/groq 등 외부 API (변경 시 자동 재로딩)"),
        "llm_api_key": ("api_key", "API 키. 외부 API 또는 Ollama Cloud 사용 시 입력. provider=ollama인데 키만 넣으면 Ollama Cloud(https://ollama.com)로 연결됨 (비우면 환경 변수에서 찾음)"),
        "llm_api_base": ("api_base", "LLM 서버/API 주소 재정의 (선택). ollama면 원격 Ollama 서버 주소, 외부 API면 OpenAI 호환 서버 등. 비우면 기본 주소(로컬, 또는 키만 있으면 Ollama Cloud) 사용"),
        "llmNumCtx": ("num_ctx", "LLM 컨텍스트 윈도우 크기(토큰). 대화 기록도 이 크기에 맞춰 유지됨. provider가 ollama일 때만 적용 (변경 시 자동 재로딩)"),
        "llmSystemPrompt": ("system_prompt", "지아의 성격/말투를 정의하는 시스템 프롬프트 (변경 시 자동 재로딩)"),
        "llm_tools": ("tools", "연결할 MCP 서버 목록 (변경 시 자동 재연결). [llm.tools.서버이름] 테이블로 추가, 빈 테이블 {}이면 사용 안 함"),
        "llm_response_reserve_tokens": ("response_reserve_tokens", "컨텍스트 윈도우에서 응답 생성을 위해 남겨둘 토큰 여유분"),
    },
    "rag": {
        "embedding_model": ("embedding_model", "기억 검색용 임베딩 모델 (변경 시 /jiarestart 필요)"),
        "faiss_threshold": ("faiss_threshold", "기억 검색 결과로 인정할 최소 유사도 점수"),
        "rag_top_k": ("top_k", "기억 검색 시 가져올 최대 개수"),
        "rag_forgettable_importance": ("forgettable_importance", "이 중요도 미만은 잊어버릴 수 있는 기억으로 저장"),
        "rag_warn_importance": ("warn_importance", "이 중요도 미만의 기억은 부정확할 수 있다는 경고와 함께 사용"),
        "rag_save_importance_min": ("save_importance_min", "이 중요도 이하의 대화는 기억으로 저장하지 않음"),
        "rag_forget_decay_per_day": ("forget_decay_per_day", "잊어버릴 수 있는 기억의 하루당 중요도 감쇠량 (0이면 망각 안 함)"),
        "rag_forget_threshold": ("forget_threshold", "감쇠된 중요도가 이 값 미만이 되면 기억을 삭제"),
        "rag_retrieval_boost": ("retrieval_boost", "기억이 검색에 사용될 때마다 중요도를 이만큼 올림 (자주 쓰는 기억은 오래 유지)"),
        "rag_profile_max_facts": ("profile_max_facts", "사용자별 프로필에 보관할 최대 사실 개수 (초과 시 오래된 것부터 삭제)"),
    },
    "comfyui": {
        "comfyui_url": ("url", "ComfyUI 서버 주소 (예: http://127.0.0.1:8188). 비워두면 이미지 생성 기능을 사용하지 않음 (선택 기능)"),
        "comfyui_checkpoint": ("checkpoint", "사용할 체크포인트 파일 이름 (ComfyUI의 models/checkpoints 안 파일명)"),
        "comfyui_steps": ("steps", "이미지 생성 스텝 수 (Flux Schnell은 4, 일반 SD 모델은 20~30 권장)"),
        "comfyui_cfg": ("cfg", "CFG 스케일 (Flux Schnell은 1.0, 일반 SD 모델은 7.0 권장)"),
        "comfyui_width": ("width", "생성 이미지 가로 크기"),
        "comfyui_height": ("height", "생성 이미지 세로 크기"),
        "comfyui_sampler": ("sampler", "샘플러 이름 (예: euler)"),
        "comfyui_scheduler": ("scheduler", "스케줄러 이름 (예: normal, Flux는 simple 권장)"),
        "comfyui_negative_prompt": ("negative_prompt", "네거티브 프롬프트 (Flux 계열은 비워둠)"),
        "comfyui_timeout_sec": ("timeout_sec", "이미지 생성 대기 제한 시간 (초)"),
    },
    "settings": {
        "settings_watch_interval_sec": ("watch_interval_sec", "settings.toml 변경 감지 주기 (초, 변경 시 재시작 필요)"),
    },
}

# 필드명 -> (섹션, 키) 역방향 매핑
FIELD_TO_TOML: dict[str, tuple[str, str]] = {
    field_name: (section, key)
    for section, field_map in TOML_LAYOUT.items()
    for field_name, (key, _comment) in field_map.items()
}


def _env_key(field_name: str) -> str:
    """필드 이름을 환경 변수 키로 변환합니다. (예: llmModel -> JIA_LLM_MODEL)"""
    return ENV_PREFIX + re.sub(r"(?<!^)(?=[A-Z])", "_", field_name).upper()


def _parse_env_value(raw: str, f) -> object:
    """환경 변수(.env) 문자열 값을 필드 타입에 맞게 변환합니다. (.env 마이그레이션/환경 변수 오버라이드용)"""
    if f.type is bool:
        return raw.strip().lower() in ("1", "true", "yes", "on")
    if f.type is float:
        return float(raw)
    if f.type is int:
        return int(raw)
    if f.name == "llm_tools":
        return json.loads(raw) if raw.strip() else {}
    if f.name == "debug_text_channel":
        return int(raw) if raw.strip() else None
    return raw


def _coerce_toml_value(value, f) -> object:
    """settings.toml에서 읽은 값을 필드 타입에 맞게 변환합니다."""
    if f.name == "debug_text_channel":
        return int(value) or None  # 0은 미설정으로 취급
    if f.name == "llm_tools":
        plain = value.unwrap() if hasattr(value, "unwrap") else value
        if isinstance(plain, dict):
            return plain
        # 구버전 형식(list, 미사용이었음)은 기본 MCP 서버 구성으로 대체
        logging.info("[Config] [llm] tools가 구버전 형식(배열)이라 기본 MCP 서버 구성을 사용해요. [llm.tools.서버이름] 테이블로 바꿔 적어주세요.")
        return DEFAULT_MCP_SERVERS()
    if f.type is bool:
        return bool(value)
    if f.type is float:
        return float(value)
    if f.type is int:
        return int(value)
    return str(value)


@dataclass
class Config:
    join_reply: bool = True
    leave_reply: bool = True
    faiss_threshold: float = 0.5
    whisper_model: str = "turbo"
    llmModel: str = "gemma4:latest"
    llm_provider: str = "ollama"  # ollama(로컬) 또는 openai/anthropic/google_genai/groq 등 외부 API
    llm_api_key: str = ""  # 외부 API 사용 시 키 (ollama면 불필요)
    llm_api_base: str = ""  # LLM 서버 주소 재정의 (선택). ollama면 원격 서버 주소, 외부 API면 호환 서버 주소
    llmNumCtx: int = 16384
    tts_model: str = ""
    llmSystemPrompt: str = """너는 "지아"라는 이름의 친구야. 친구처럼 친근한 반말을 사용해.

말투 규칙:
- 이모티콘과 이모지는 절대 사용하지 마.
- "어휴" 같은 감탄사를 말 앞에 붙이지 마.
- "알겠습니다", "요청하신 정보를 제공합니다" 같은 공식적이고 딱딱한 표현은 쓰지 마.
- 기계적으로 정보를 나열하거나 감정 없이 건조하게 대답하지 마.
- 사용자에게 짜증내거나 사용자의 말을 단순히 따라하지 마.

사용자의 입력은 음성 인식을 거쳐 들어올 수 있어서 문장이 불완전할 수 있어. 어색한 부분은 문맥으로 추론해서 복원하고, 도저히 의도를 알 수 없을 때만 무슨 뜻인지 되물어봐. 그 외에는 사용자에게 굳이 질문을 던지는 응답은 피해줘."""
    llm_tools: dict = field(default_factory=DEFAULT_MCP_SERVERS)
    bot_token: str = ""
    debug_text_channel: int | None = None

    # === 음성 수신/발화 감지 (저장 즉시 반영) ===
    voice_timeout_sec: float = 0.1
    voice_interrupt_speech_sec: float = 0.5
    proactive_idle_sec: int = 0  # 0이면 먼저 말 걸기 비활성화

    # === 사운드보드 자동 반응 (저장 즉시 반영) ===
    soundboard_auto_react: bool = False
    soundboard_auto_react_cooldown_sec: int = 20
    soundboard_auto_react_chance: float = 0.35

    # === 음악 재생/덕킹 (저장 즉시 반영) ===
    music_volume: float = 0.7
    music_duck_volume: float = 0.25
    music_max_playlist_items: int = 50

    # === 위험 기능 (저장 즉시 반영) ===
    allow_unsafe_jiaplay: bool = False

    # === VAD (저장 즉시 반영) ===
    vad_threshold: float = 0.7
    vad_min_speech_ms: int = 150
    vad_min_silence_ms: int = 1000
    vad_max_speech_sec: int = 30
    vad_padding_ms: int = 200

    # === Whisper (모델/디바이스 변경 시 자동으로 다시 로드됨) ===
    whisper_device: str = "cuda"
    whisper_compute_type: str = "float16"
    whisper_beam_size: int = 5

    # === RAG/기억 ===
    embedding_model: str = "dragonkue/BGE-m3-ko"  # 변경 시 재시작(/jiarestart) 필요
    rag_top_k: int = 3
    rag_forgettable_importance: float = 0.8
    rag_warn_importance: float = 0.5
    rag_save_importance_min: float = 0.1
    rag_forget_decay_per_day: float = 0.02
    rag_forget_threshold: float = 0.15
    rag_retrieval_boost: float = 0.05
    rag_profile_max_facts: int = 20

    # === LLM 세부 (저장 즉시 반영) ===
    llm_response_reserve_tokens: int = 2048

    # === ComfyUI 이미지 생성 (선택 기능, url이 비어 있으면 비활성) ===
    comfyui_url: str = ""
    comfyui_checkpoint: str = ""
    comfyui_steps: int = 20
    comfyui_cfg: float = 7.0
    comfyui_width: int = 1024
    comfyui_height: int = 1024
    comfyui_sampler: str = "euler"
    comfyui_scheduler: str = "normal"
    comfyui_negative_prompt: str = ""
    comfyui_timeout_sec: int = 120

    # === 설정 파일 감시 ===
    settings_watch_interval_sec: float = 2.0

    def _toml_item(self, field_name: str):
        """필드 값을 (비교용 일반 값, tomlkit 아이템) 쌍으로 변환합니다."""
        value = getattr(self, field_name)
        if field_name == "debug_text_channel":
            value = int(value) if value else 0
        if isinstance(value, str) and "\n" in value:
            return value, tomlkit.string(value, multiline=True)
        return value, tomlkit.item(value)

    def _apply_toml(self, doc):
        """settings.toml 문서의 값을 Config 필드에 반영합니다. (없는 키는 환경 변수 JIA_*로 대체 가능)"""
        for f in fields(self):
            section, key = FIELD_TO_TOML[f.name]
            table = doc.get(section)
            if table is not None and key in table:
                setattr(self, f.name, _coerce_toml_value(table[key], f))
                continue
            raw = os.environ.get(_env_key(f.name))
            if raw is not None:
                setattr(self, f.name, _parse_env_value(raw, f))

    def _apply_env(self, data: dict):
        """`.env` 형식의 키-값 데이터를 Config 필드에 반영합니다. (.env -> settings.toml 마이그레이션용)"""
        for f in fields(self):
            raw = data.get(_env_key(f.name))
            if raw is not None:
                setattr(self, f.name, _parse_env_value(raw, f))

    def _migrate_from_env(self) -> bool:
        """기존 .env의 JIA_ 설정값을 가져오고, .env에는 LangSmith 등 외부 서비스 키만 남깁니다."""
        if not ENV_PATH.exists():
            return False
        data = dotenv_values(ENV_PATH)
        if not any(key.startswith(ENV_PREFIX) for key in data):
            return False
        self._apply_env(data)
        # 원본을 백업해두고 .env에서 JIA_ 키만 제거 (다른 키와 주석은 보존)
        original = ENV_PATH.read_text(encoding="utf-8")
        ENV_PATH.with_name(".env.bak").write_text(original, encoding="utf-8")
        kept = []
        for line in original.splitlines():
            m = re.match(r"\s*([A-Za-z_][A-Za-z0-9_]*)\s*=", line)
            if m and m.group(1).startswith(ENV_PREFIX):
                continue
            kept.append(line)
        ENV_PATH.write_text("\n".join(kept) + "\n", encoding="utf-8")
        logging.info("[Config:Migration] .env의 JIA_ 설정을 settings.toml로 옮기고, .env에는 외부 서비스 키만 남겼어요. (원본은 .env.bak에 백업)")
        return True

    def reload(self):
        """settings.toml에서 설정을 다시 로드하여 현재 객체를 업데이트합니다."""
        if not SETTINGS_PATH.exists():
            logging.warning("[Config:Reloader] settings.toml 파일을 찾을 수 없어 리로드에 실패했어요.")
            return
        try:
            self._apply_toml(tomlkit.parse(SETTINGS_PATH.read_text(encoding="utf-8")))
            logging.info("[Config:Reloader] 설정을 다시 불러왔어요.")
        except Exception as e:
            logging.error(f"[Config:Reloader] 설정 리로드 중 오류 발생: {e}")

    def start_auto_reload(self, on_change=None) -> threading.Thread:
        """settings.toml 파일 변경을 감시해 자동으로 reload하는 데몬 스레드를 시작합니다.

        on_change: 값이 실제로 바뀐 경우 {필드명: (이전 값, 새 값)} 딕셔너리로 호출되는 콜백.
        모델 재로딩처럼 단순 reload로 부족한 후처리를 여기에 연결할 수 있습니다.
        """
        def _watch():
            last_mtime = SETTINGS_PATH.stat().st_mtime if SETTINGS_PATH.exists() else None
            while True:
                time.sleep(self.settings_watch_interval_sec)
                try:
                    mtime = SETTINGS_PATH.stat().st_mtime if SETTINGS_PATH.exists() else None
                    if mtime is None or mtime == last_mtime:
                        continue
                    last_mtime = mtime
                    old_values = {f.name: getattr(self, f.name) for f in fields(self)}
                    self._apply_toml(tomlkit.parse(SETTINGS_PATH.read_text(encoding="utf-8")))
                    changed = {
                        name: (old, getattr(self, name))
                        for name, old in old_values.items()
                        if old != getattr(self, name)
                    }
                    if not changed:
                        continue
                    logging.info(f"[Config:Watcher] settings.toml 변경을 감지해 설정을 다시 불러왔어요. 바뀐 항목: {list(changed)}")
                    if on_change:
                        try:
                            on_change(changed)
                        except Exception as e:
                            logging.error(f"[Config:Watcher] 변경 후처리(on_change) 중 오류 발생: {e}")
                except Exception as e:
                    logging.error(f"[Config:Watcher] settings.toml 감시 중 오류 발생: {e}")

        watcher = threading.Thread(target=_watch, daemon=True, name="config-settings-watcher")
        watcher.start()
        logging.info(f"[Config:Watcher] settings.toml 자동 리로드를 시작했어요. (감지 주기: {self.settings_watch_interval_sec}초)")
        return watcher

    def save_setting(self):
        """현재 설정을 settings.toml에 저장합니다. 사용자가 적어둔 주석과 키 순서는 보존합니다."""
        if SETTINGS_PATH.exists():
            doc = tomlkit.parse(SETTINGS_PATH.read_text(encoding="utf-8"))
        else:
            doc = tomlkit.document()
            doc.add(tomlkit.comment("Project Jia 설정 파일"))
            doc.add(tomlkit.comment("값을 수정하고 저장하면 실행 중에도 자동으로 반영돼요. (모델 설정은 해당 모델만 자동 재로딩)"))
            doc.add(tomlkit.comment("'재시작 필요'로 표시된 항목은 /jiarestart 또는 프로그램 재시작 후에 적용돼요."))
            doc.add(tomlkit.nl())
        for section, field_map in TOML_LAYOUT.items():
            if section not in doc:
                doc[section] = tomlkit.table()
            table = doc[section]
            for field_name, (key, comment) in field_map.items():
                plain, item = self._toml_item(field_name)
                if key not in table:
                    if comment:
                        comment_lines = comment.splitlines()
                        if len(comment_lines) == 1:
                            item.comment(comment)
                        else:
                            for line in comment_lines:
                                table.add(tomlkit.comment(line))
                    table[key] = item
                elif table[key] != plain:
                    # 값이 바뀐 키만 갱신 (사용자가 직접 단 주석을 최대한 보존)
                    table[key] = item
        SETTINGS_PATH.write_text(tomlkit.dumps(doc), encoding="utf-8")
        logging.info("[Config:Saver] 설정을 settings.toml 파일에 저장했어요.")

    @classmethod
    def load_setting(cls: Type['Config']) -> 'Config':
        config = cls()
        if not SETTINGS_PATH.exists():
            # 기존 .env에 JIA_ 설정이 있다면 가져온 뒤 settings.toml을 생성
            migrated = config._migrate_from_env()
            config.save_setting()
            if not migrated:
                logging.info("[Config:Loader] settings.toml 파일이 없어 기본 값으로 새로 만들었어요. 파일을 삭제했거나 최초 실행이라면 전혀 문제 없는 현상이니 무시해도 괜찮아요.")
            return config
        doc = tomlkit.parse(SETTINGS_PATH.read_text(encoding="utf-8"))
        config._apply_toml(doc)
        # 업데이트로 새 설정 항목이 생겼다면 기본값을 채워 넣어 파일에서 바로 보이게 함
        missing = any(
            section not in doc or key not in doc[section]
            for section, key in FIELD_TO_TOML.values()
        )
        if missing:
            config.save_setting()
        return config
