from config.config_manager import config
import torch
from espnet2.bin.tts_inference import Text2Speech
import soundfile as sf
import torchaudio
import io
import logging
import threading

text2speech: Text2Speech | None = None
_loaded_tts_model: str | None = None
_tts_lock = threading.Lock()


def _load_tts_model_locked():
    global text2speech, _loaded_tts_model
    model_path = (config.tts_model or "").strip()
    if not model_path:
        raise RuntimeError("TTS 모델 경로가 설정되어 있지 않습니다. settings.toml의 [tts] model을 확인해주세요.")
    text2speech = Text2Speech.from_pretrained(model_path)
    _loaded_tts_model = model_path
    logging.info("[TTS] ESPnet TTS 모델이 로딩되었어요.")


def reload_tts_model():
    with _tts_lock:
        _load_tts_model_locked()


def ensure_tts_model_loaded():
    """음성 채널 접속 전에 TTS 모델이 로드되어 있는지 보장합니다."""
    expected_model = (config.tts_model or "").strip()
    if text2speech is not None and _loaded_tts_model == expected_model:
        return
    with _tts_lock:
        if text2speech is None or _loaded_tts_model != expected_model:
            _load_tts_model_locked()


# 기존처럼 프로그램 시작 시 TTS 모델을 미리 로드합니다.
ensure_tts_model_loaded()


def generate_tts(text: str) -> io.BytesIO:
    ensure_tts_model_loaded()
    # 음성 생성
    with torch.no_grad():
        wav = text2speech(text)['wav']
    
    wav_48k = torchaudio.functional.resample(wav, orig_freq=24000, new_freq=48000)
    
    # 음성 저장
    buffer = io.BytesIO()
    sf.write(buffer, wav_48k.numpy(), samplerate=48000, format='WAV', subtype='PCM_16')
    buffer.seek(0)
    
    return buffer
