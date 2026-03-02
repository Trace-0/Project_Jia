from faster_whisper import WhisperModel
import numpy as np
from config.config_manager import config
import logging

model: WhisperModel | None = None

def reload_whisper_model():
    global model
    model = WhisperModel(config.whisper_model, device="cuda", compute_type="float16")
    logging.info("[Whisper] Faster Whisper 모델이 로딩되었어요.")

def transcribe_sync(audio_array: np.ndarray) -> str:
    if model is None:
        reload_whisper_model()
    
    segments, info = model.transcribe(audio_array, beam_size=5)
    
    transcribed_text = ""
    for segment in segments:
        transcribed_text += segment.text

    return transcribed_text