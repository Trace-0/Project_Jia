import json
import os
import re
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Type, List
import logging

from dotenv import dotenv_values

# 설정은 프로젝트 루트의 .env 파일에 JIA_ 접두사 키로 저장됩니다.
ENV_PATH = Path(__file__).resolve().parents[1] / ".env"
ENV_PREFIX = "JIA_"


def _env_key(field_name: str) -> str:
    """필드 이름을 .env 키로 변환합니다. (예: llmModel -> JIA_LLM_MODEL)"""
    return ENV_PREFIX + re.sub(r"(?<!^)(?=[A-Z])", "_", field_name).upper()


def _parse_env_value(raw: str, f) -> object:
    if f.type is bool:
        return raw.strip().lower() in ("1", "true", "yes", "on")
    if f.type is float:
        return float(raw)
    if f.name == "llm_tools":
        return json.loads(raw) if raw.strip() else []
    if f.name == "debug_text_channel":
        return int(raw) if raw.strip() else None
    return raw


def _format_env_value(value) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if value is None:
        return ""
    if isinstance(value, list):
        return json.dumps(value, ensure_ascii=False)
    return str(value)


def _quote_env_value(value: str) -> str:
    escaped = value.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")
    return f'"{escaped}"'


@dataclass
class Config:
    join_reply: bool = True
    leave_reply: bool = True
    faiss_threshold: float = 0.5
    whisper_model: str = "openai/whisper-large-v3-turbo"
    llmModel: str = "gpt-oss:20b"
    tts_model: str = ""
    llmSystemPrompt: str = "너는 \"지아\"라는 이름의 친구야. 너의 말투는 친구처럼 반말을 써\n답변에 이모티콘은 절대로 사용하지마.\n답변을 생성할 때 \"야\"라는 표현은 자제하고 닉네임이나 별명으로 불러줘.\n\"어휴\" 사용하지마.\n\n금지되는 말투나 태도:\n- 너무 공식적이고 딱딱한 표현 (예: \"알겠습니다\", \"요청하신 정보를 제공합니다\")\n- 기계적으로 정보를 나열만 하는 태도\n- 감정 없이 건조하게 대답하는 것\n- 이모티콘을 사용하는 것\n- \"어휴\"를 말 앞에 붙이는 것\n- 사용자에게 짜증내는 말투\n- 사용자의 말을 단순히 따라하는 것\n\n사용자에게 무언가를 질문하는 응답은 피해야해.\n사용자의 입력은 음성 인식 프로그램을 통해 입력되고 있기 때문에 입력되는 문장이 완벽하지 않을 수 있어. 그러니 입력된 문장이 불완전, 불안정된 경우 최대한 추론하여 문장의 오류를 복원하고 그래도 문장이 불안정한 경우 사용자에게 어떤 말을 했는지 혹은 어떤 의도로 이러한 말을 했는지 질문하는 것은 허락할게."
    llm_tools: List[str] = field(default_factory=lambda: [])
    bot_token: str = ""
    debug_text_channel: int | None = None

    def _apply_env(self, data: dict):
        """`.env` 값(없으면 실제 환경 변수)을 Config 필드에 반영합니다."""
        for f in fields(self):
            key = _env_key(f.name)
            raw = data.get(key)
            if raw is None:
                raw = os.environ.get(key)
            if raw is not None:
                setattr(self, f.name, _parse_env_value(raw, f))

    def reload(self):
        """환경 변수(.env)에서 설정을 다시 로드하여 현재 객체를 업데이트합니다."""
        if not ENV_PATH.exists():
            logging.warning("[Config:Reloader] .env 파일을 찾을 수 없어 리로드에 실패했어요.")
            return
        try:
            self._apply_env(dotenv_values(ENV_PATH))
            logging.info("[Config:Reloader] 설정을 다시 불러왔어요.")
        except Exception as e:
            logging.error(f"[Config:Reloader] 설정 리로드 중 오류 발생: {e}")

    def save_setting(self):
        """JIA_ 접두사 키만 갱신하고 .env의 나머지 항목(API 키 등)은 보존합니다."""
        lines = []
        if ENV_PATH.exists():
            lines = ENV_PATH.read_text(encoding="utf-8").splitlines()
        new_values = {
            _env_key(f.name): _quote_env_value(_format_env_value(getattr(self, f.name)))
            for f in fields(self)
        }
        result = []
        for line in lines:
            m = re.match(r"\s*([A-Za-z_][A-Za-z0-9_]*)\s*=", line)
            if m and m.group(1) in new_values:
                key = m.group(1)
                result.append(f"{key}={new_values.pop(key)}")
            else:
                result.append(line)
        result.extend(f"{key}={value}" for key, value in new_values.items())
        ENV_PATH.write_text("\n".join(result) + "\n", encoding="utf-8")
        logging.info("[Config:Saver] 설정을 .env 파일에 저장했어요.")

    @classmethod
    def load_setting(cls: Type['Config']) -> 'Config':
        config = cls()
        if not ENV_PATH.exists():
            logging.info("[Config:Lodder] .env 파일이 없어 기본 값을 불러왔어요. 파일을 삭제했거나 최초 실행이라면 전혀 문제 없는 현상이니 무시해도 괜찮아요.")
            config.save_setting()
            return config
        config._apply_env(dotenv_values(ENV_PATH))
        return config