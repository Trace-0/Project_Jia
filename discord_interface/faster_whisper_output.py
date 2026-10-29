from faster_whisper import WhisperModel
import numpy as np
from config.config_manager import config
import logging
import re
import threading

model: WhisperModel | None = None
_loaded_signature: tuple[str, str, str] | None = None
_model_lock = threading.Lock()

# Whisper가 무음/잡음 구간에서 지어내는 것으로 잘 알려진 문장 (공백·문장부호를 뺀 소문자 기준으로 비교)
# "감사합니다", "Thank you."처럼 실제로도 자주 하는 말은 넣지 않습니다.
_HALLUCINATION_PATTERNS = [
    re.compile(p) for p in (
        r"^시청해주셔서(감사|고맙)합니다$",
        r"^구독(과|하고)?좋아요.*(부탁|눌러)",
        r"^(다음|오늘)영상에서(만나요|뵙겠습니다)$",
        r"^mbc뉴스.*입니다$",
        r"^(한글)?자막(제공|by|은)",
        r"^(thank(s|you)forwatching|pleasesubscribe|subscribetomychannel)$",
        r"^ご視聴ありがとうございました$",
        r"^チャンネル登録",
    )
]


def _current_signature() -> tuple[str, str, str]:
    return (config.whisper_model, config.whisper_device, config.whisper_compute_type)

def reload_whisper_model():
    global model, _loaded_signature
    with _model_lock:
        signature = _current_signature()
        model = WhisperModel(signature[0], device=signature[1], compute_type=signature[2])
        _loaded_signature = signature
        logging.info("[Whisper] Faster Whisper 모델이 로딩되었어요.")


def ensure_whisper_model_loaded():
    """음성 채널 접속 전에 Whisper 모델이 로드되어 있는지 보장합니다."""
    if model is not None and _loaded_signature == _current_signature():
        return
    reload_whisper_model()

def _pick_language(audio_array: np.ndarray) -> str | None:
    """설정된 후보 언어 중 가장 가능성이 높은 언어를 고릅니다. (후보가 없으면 None = 전체 자동 감지)

    짧은 발화는 엉뚱한 언어로 감지되기 쉬워서, 실제로 쓰는 언어 안에서만 고르게 합니다.
    """
    candidates = [lang.strip().lower() for lang in config.whisper_languages if lang and lang.strip()]
    if not candidates or not model.model.is_multilingual:
        return None
    if len(candidates) == 1:
        return candidates[0]
    _, _, all_probs = model.detect_language(audio_array)
    probs = dict(all_probs)
    return max(candidates, key=lambda lang: probs.get(lang, 0.0))

def _is_hallucination(text: str) -> bool:
    normalized = re.sub(r"[\s\.,!?~…·'\"。、！？]", "", text).lower()
    if not normalized:
        return True
    return any(p.search(normalized) for p in _HALLUCINATION_PATTERNS)

def transcribe_sync(audio_array: np.ndarray) -> str:
    ensure_whisper_model_loaded()

    language = _pick_language(audio_array)
    segments, info = model.transcribe(
        audio_array,
        beam_size=config.whisper_beam_size,
        language=language,
        # 앞 구간의 (잘못된) 인식 결과가 뒤 구간으로 번지지 않게 함
        condition_on_previous_text=False,
    )

    transcribed_text = ""
    for segment in segments:
        if _is_hallucination(segment.text):
            logging.info(f"[Whisper] 무음에서 지어낸 것으로 보이는 문장을 버렸어요. -> {segment.text}")
            continue
        transcribed_text += segment.text

    if transcribed_text:
        logging.info(f"[Whisper] 인식 언어: {info.language} ({info.language_probability:.2f})")
    return transcribed_text
