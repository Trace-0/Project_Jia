import uuid
import time
from config.config_manager import config
import torch
from g2pk import G2p
from espnet2.bin.tts_inference import Text2Speech
import soundfile as sf
import torchaudio
import logging

text2speech = Text2Speech.from_pretrained(config.tts_model)

def reload_tts_model():
    global text2speech
    text2speech = Text2Speech.from_pretrained(config.tts_model)

def generate_tts(text: str) -> str:
    timestamp = int(time.time() * 1000)
    unique_id = uuid.uuid4().hex
    file_name = f"Source/output_temp/output_{timestamp}_{unique_id}.wav"
    
    # Convert text to phonemes
    phonemes = G2p()(text)

    logging.info(f"[TTS:g2pk] 출력할 텍스트를 발음하기 편한 형태로 변경했어요.\n -> {phonemes}")
    
    # Generate speech from phonemes
    with torch.no_grad():
        wav= text2speech(phonemes)['wav']

    wav_48k = torchaudio.functional.resample(wav, orig_freq=24000, new_freq=48000)
    
    # Save the generated audio to a file
    sf.write(file_name, wav_48k.numpy(), samplerate=48000, format='WAV', subtype='PCM_16')
    
    return file_name