import time
import uuid
from TTS.api import TTS
from g2pk import G2p
import re
import logging

device = "cuda"

tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(device)

logging.basicConfig(level=logging.INFO)

def generate_with_coqui_tts(text):
    timestamp = int(time.time() * 1000)
    unique_id = uuid.uuid4().hex
    file_name = f"output_temp/output_{timestamp}_{unique_id}.wav"

    text = re.sub(r'[^A-Za-z0-9가-힣\s]', '', text)

    phonemes = G2p()(text)

    logging.info(f"[Coqui TTS:g2pk] 출력할 텍스트를 발음하기 편한 형태로 변경했어요.\n -> {phonemes}")

    tts.tts_to_file(
        text=phonemes,
        file_path=file_name,
        speaker_wav="semple.wav",
        language="ko",
    )

generate_with_coqui_tts("안녕하세요. 지금은 'coqui TTS' 테스트 중입니다. 1 / 2 : 3 잘 들리시나요?")

# 중동쪽 사람이 어눌하게 한국어 따라하는 느낌이 남.