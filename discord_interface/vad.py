import torch
import logging
import threading
from config.config_manager import config

# Load the Silero VAD model via torch.hub (one-time global load):contentReference[oaicite:3]{index=3}
model, utils = torch.hub.load(repo_or_dir='snakers4/silero-vad', model='silero_vad')
(get_speech_timestamps, _, _, _, _) = utils
# 모델이 내부 상태를 가지므로 여러 스레드(사용자별 음성 변환, 끼어들기 확인)가 동시에 쓰지 않게 잠금
_model_lock = threading.Lock()

def pad_timestamps(timestamps, sample_rate, padding_ms):
    padding = int(sample_rate * padding_ms / 1000)
    for t in timestamps:
        t['start'] = max(0, t['start'] - padding)
    return timestamps

def get_speech_timestamps_from_array(audio_np, sampling_rate=16000):
    """
    Apply Silero VAD to a 1D NumPy array of audio (float32 or int16) at 16 kHz.
    Returns a list of {'start':sample_index, 'end':sample_index} segments.
    """
    try:
        with _model_lock:
            model.reset_states()

            # Run VAD (it returns list of speech segments in samples)
            speech_timestamps = get_speech_timestamps(audio_np, model,
                                                threshold=config.vad_threshold,
                                                min_speech_duration_ms=config.vad_min_speech_ms,
                                                min_silence_duration_ms=config.vad_min_silence_ms,
                                                max_speech_duration_s=config.vad_max_speech_sec,
                                                sampling_rate=sampling_rate)
        speech_timestamps = pad_timestamps(speech_timestamps, sampling_rate, config.vad_padding_ms)
        logging.info(f"[VAD:Get_Timestamps] 입력된 음성 패킷에서 대화로 추정되는 부분만 잘라냈어요.\n   -> {speech_timestamps}")
        return speech_timestamps
    except Exception as e:
        logging.error(f"[VAD:Get_Timestamps] VAD로 대화 인식 과정에서 오류가 발생했어요. :(\n   -> {e}")