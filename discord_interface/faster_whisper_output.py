from faster_whisper import WhisperModel
import numpy as np
from config.config_manager import config
import logging
import threading

model: WhisperModel | None = None
_loaded_signature: tuple[str, str, str] | None = None
_model_lock = threading.Lock()


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

def transcribe_sync(audio_array: np.ndarray) -> str:
    ensure_whisper_model_loaded()

    segments, info = model.transcribe(audio_array, beam_size=config.whisper_beam_size)
    
    transcribed_text = ""
    for segment in segments:
        transcribed_text += segment.text

    return transcribed_text
