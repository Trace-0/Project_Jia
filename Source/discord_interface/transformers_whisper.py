from transformers import WhisperProcessor, WhisperForConditionalGeneration
import numpy as np
import torch
from config.config_manager import config
import logging

model = None
processor = None

def reload_whisper_model():
    global model, processor
    model = WhisperForConditionalGeneration.from_pretrained(config.whisper_model).to("cuda")
    processor = WhisperProcessor.from_pretrained(config.whisper_model)
    model.eval()
    logging.info("[Whisper] Whisper 모델이 로딩되었어요.")

def transcribe_sync(audio_array: np.ndarray) -> str:
    if model is None:
        reload_whisper_model()
    inputs = processor(audio_array, sampling_rate=16000, return_tensors="pt").to("cuda")
    with torch.no_grad():
        predicted_ids = model.generate(inputs["input_features"], pad_token_id=processor.tokenizer.pad_token_id)
    transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
    return transcription