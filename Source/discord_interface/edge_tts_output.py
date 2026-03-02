import time
import uuid
import edge_tts
import logging

logging.basicConfig(level=logging.INFO)

def generate_with_edge_tts(text):
    timestamp = int(time.time() * 1000)
    unique_id = uuid.uuid4().hex
    voice = "ko-KR-SunHiNeural"
    file_name = f"Source/output_temp/output_{timestamp}_{unique_id}.wav"

    communicate = edge_tts.Communicate(text, voice, rate="+10%", pitch="+8Hz")
    communicate.save_sync(file_name)
    return file_name

if __name__ == "__main__":
    generate_with_edge_tts("안녕하세요. 지금은 'edge_tts' 테스트 중입니다. 1 / 2 : 3 잘 들리시나요?")