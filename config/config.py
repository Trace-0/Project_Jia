import json
import os
import re
import threading
import time
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Type, List
import logging

import tomlkit
from dotenv import dotenv_values

# 모든 설정은 프로젝트 루트의 settings.toml에 저장됩니다.
# .env는 LangSmith 추적 같은 개발용 키 전용입니다. (langchain이 환경 변수로 직접 읽으며, 일반 사용자는 필요 없음)
ROOT_DIR = Path(__file__).resolve().parents[1]
SETTINGS_PATH = ROOT_DIR / "settings.toml"
ENV_PATH = ROOT_DIR / ".env"
ENV_PREFIX = "JIA_"

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
        "llmModel": ("model", "Ollama LLM 모델 (변경 시 자동 재로딩)"),
        "llmNumCtx": ("num_ctx", "LLM 컨텍스트 윈도우 크기(토큰). 대화 기록도 이 크기에 맞춰 유지됨 (변경 시 자동 재로딩)"),
        "llmSystemPrompt": ("system_prompt", "지아의 성격/말투를 정의하는 시스템 프롬프트 (변경 시 자동 재로딩)"),
        "llm_tools": ("tools", "추가 도구 목록 (현재 미사용)"),
        "llm_response_reserve_tokens": ("response_reserve_tokens", "컨텍스트 윈도우에서 응답 생성을 위해 남겨둘 토큰 여유분"),
    },
    "rag": {
        "embedding_model": ("embedding_model", "기억 검색용 임베딩 모델 (변경 시 /jiarestart 필요)"),
        "faiss_threshold": ("faiss_threshold", "기억 검색 결과로 인정할 최소 유사도 점수"),
        "rag_top_k": ("top_k", "기억 검색 시 가져올 최대 개수"),
        "rag_forgettable_importance": ("forgettable_importance", "이 중요도 미만은 잊어버릴 수 있는 기억으로 저장"),
        "rag_warn_importance": ("warn_importance", "이 중요도 미만의 기억은 부정확할 수 있다는 경고와 함께 사용"),
        "rag_save_importance_min": ("save_importance_min", "이 중요도 이하의 대화는 기억으로 저장하지 않음"),
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
        return json.loads(raw) if raw.strip() else []
    if f.name == "debug_text_channel":
        return int(raw) if raw.strip() else None
    return raw


def _coerce_toml_value(value, f) -> object:
    """settings.toml에서 읽은 값을 필드 타입에 맞게 변환합니다."""
    if f.name == "debug_text_channel":
        return int(value) or None  # 0은 미설정으로 취급
    if f.type is bool:
        return bool(value)
    if f.type is float:
        return float(value)
    if f.type is int:
        return int(value)
    if f.name == "llm_tools":
        return [str(item) for item in value]
    return str(value)


@dataclass
class Config:
    join_reply: bool = True
    leave_reply: bool = True
    faiss_threshold: float = 0.5
    whisper_model: str = "openai/whisper-large-v3-turbo"
    llmModel: str = "gpt-oss:20b"
    llmNumCtx: int = 16384
    tts_model: str = ""
    llmSystemPrompt: str = "너는 \"지아\"라는 이름의 친구야. 너의 말투는 친구처럼 반말을 써\n답변에 이모티콘은 절대로 사용하지마.\n답변을 생성할 때 \"야\"라는 표현은 자제하고 닉네임이나 별명으로 불러줘.\n\"어휴\" 사용하지마.\n\n금지되는 말투나 태도:\n- 너무 공식적이고 딱딱한 표현 (예: \"알겠습니다\", \"요청하신 정보를 제공합니다\")\n- 기계적으로 정보를 나열만 하는 태도\n- 감정 없이 건조하게 대답하는 것\n- 이모티콘을 사용하는 것\n- \"어휴\"를 말 앞에 붙이는 것\n- 사용자에게 짜증내는 말투\n- 사용자의 말을 단순히 따라하는 것\n\n사용자에게 무언가를 질문하는 응답은 피해야해.\n사용자의 입력은 음성 인식 프로그램을 통해 입력되고 있기 때문에 입력되는 문장이 완벽하지 않을 수 있어. 그러니 입력된 문장이 불완전, 불안정된 경우 최대한 추론하여 문장의 오류를 복원하고 그래도 문장이 불안정한 경우 사용자에게 어떤 말을 했는지 혹은 어떤 의도로 이러한 말을 했는지 질문하는 것은 허락할게."
    llm_tools: List[str] = field(default_factory=lambda: [])
    bot_token: str = ""
    debug_text_channel: int | None = None

    # === 음성 수신/발화 감지 (저장 즉시 반영) ===
    voice_timeout_sec: float = 0.1
    voice_interrupt_speech_sec: float = 0.5

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

    # === LLM 세부 (저장 즉시 반영) ===
    llm_response_reserve_tokens: int = 2048

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
                        item.comment(comment)
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
