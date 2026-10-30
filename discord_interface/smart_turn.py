"""Smart Turn v3 문장 끝(발화 종료) 판정

음성만 보고 "말이 끝났는지, 잠깐 멈춘 건지"를 확률로 알려주는 작은 모델입니다. (pipecat-ai/smart-turn-v3, BSD-2)
16kHz 모노 음성의 마지막 8초까지 보고, 한국어/영어/일본어를 포함한 23개 언어를 지원합니다.

모델을 받지 못했거나 추론 중 오류가 나면 None을 돌려주고, 호출하는 쪽은 고정 대기 시간으로 처리합니다.
"""
import logging
import threading

import numpy as np

SMART_TURN_REPO = "pipecat-ai/smart-turn-v3"
SMART_TURN_FILE = "smart-turn-v3.2-cpu.onnx"
SAMPLE_RATE = 16000
MAX_SECONDS = 8

_session = None
_feature_extractor = None
_load_failed = False
_load_lock = threading.Lock()


def ensure_smart_turn_loaded(retry: bool = False) -> bool:
    """모델을 (필요하면 내려받아) 로드합니다. 실패하면 False이고, retry=False면 다시 시도하지 않습니다."""
    global _session, _feature_extractor, _load_failed
    if _session is not None:
        return True
    with _load_lock:
        if _session is not None:
            return True
        if _load_failed and not retry:
            return False
        try:
            import onnxruntime as ort
            from huggingface_hub import hf_hub_download
            from transformers import WhisperFeatureExtractor

            model_path = hf_hub_download(SMART_TURN_REPO, SMART_TURN_FILE)
            options = ort.SessionOptions()
            options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
            options.inter_op_num_threads = 1
            options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            _feature_extractor = WhisperFeatureExtractor(chunk_length=MAX_SECONDS)
            _session = ort.InferenceSession(model_path, sess_options=options, providers=["CPUExecutionProvider"])
            _load_failed = False
            logging.info("[SmartTurn] 문장 끝 판정 모델을 로드했어요.")
            return True
        except Exception as e:
            _load_failed = True
            logging.error(f"[SmartTurn] 문장 끝 판정 모델을 로드하지 못해서 고정 대기 시간으로 처리할게요.\n   -> {e}")
            return False


def predict_turn_complete(audio_16k: np.ndarray) -> float | None:
    """말이 끝났을 확률(0.0~1.0)을 반환합니다. 판정할 수 없으면 None."""
    if audio_16k is None or len(audio_16k) == 0 or not ensure_smart_turn_loaded():
        return None
    try:
        max_samples = MAX_SECONDS * SAMPLE_RATE
        audio = np.asarray(audio_16k, dtype=np.float32)[-max_samples:]
        if len(audio) < max_samples:
            # 짧은 음성은 앞쪽을 0으로 채워 끝부분이 입력의 끝에 오도록 함 (모델 학습 방식과 동일)
            audio = np.pad(audio, (max_samples - len(audio), 0))
        inputs = _feature_extractor(
            audio,
            sampling_rate=SAMPLE_RATE,
            return_tensors="np",
            padding="max_length",
            max_length=max_samples,
            truncation=True,
            do_normalize=True,
        )
        features = inputs.input_features.squeeze(0).astype(np.float32)[np.newaxis, ...]
        outputs = _session.run(None, {"input_features": features})
        return float(outputs[0][0].item())
    except Exception as e:
        logging.error(f"[SmartTurn] 문장 끝 판정 중 오류가 발생해서 고정 대기 시간으로 처리할게요.\n   -> {e}")
        return None
