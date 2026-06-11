import asyncio
import torchaudio
import sounddevice as sd
import soundfile as sf
import os
import sys
import time

# 프로젝트 루트를 시스템 경로에 추가하여 모듈을 임포트할 수 있도록 합니다.
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from discord_interface.vad import get_speech_timestamps_from_array
from discord_interface.transformers_whisper import transcribe_sync, reload_whisper_model
from LLM.langchain_llm import astream_call_response
from discord_interface.edge_tts_output import generate_with_edge_tts
from config.config_manager import config
from LLM.LLM_model_control import load_ollama_model, unload_ollama_model

# 상수 정의
USER = "LocalUser"
GUILD_ID = 0  # 로컬 테스트를 위한 가상 길드 ID

async def play_audio_file(file_path: str):
    """생성된 WAV 파일을 재생하고 재생이 끝나면 삭제합니다."""
    try:
        data, fs = sf.read(file_path, dtype='float32')
        sd.play(data, fs)
        status = sd.wait()  # 재생이 끝날 때까지 대기
        if status:
            print(f"음성 재생 중 오류 발생: {status}")
        print(f"음성 파일 재생 완료: {file_path}")
    except Exception as e:
        print(f"음성 파일 재생/삭제 중 오류 발생: {e}")

async def process_audio_file(file_path: str):
    """오디오 파일을 처리하여 LLM 응답을 생성하고 음성으로 출력하는 메인 함수"""
    total_start_time = time.time()
    last_time = total_start_time

    if not os.path.exists(file_path):
        print(f"오디오 파일을 찾을 수 없어요: {file_path}")
        return

    # 1. 모델 로드
    print("모델을 로드하고 있어요...")
    load_ollama_model(config.llmModel)
    reload_whisper_model()
    current_time = time.time()
    print(f"모델 로드 완료. (소요 시간: {current_time - last_time:.2f}초)")
    last_time = current_time

    # 2. 오디오 파일 로드 및 리샘플링
    print(f"오디오 파일 로드 중: {file_path}")
    waveform, sample_rate = torchaudio.load(file_path)
    print(f"오디오 파일 로드 완료. (소요 시간: {current_time - last_time:.2f}초)")
    resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)
    waveform_16k = resampler(waveform).squeeze().numpy()
    current_time = time.time()
    print(f"오디오 로드 및 리샘플링 완료. (소요 시간: {current_time - last_time:.2f}초)")
    last_time = current_time

    # 3. VAD (음성 활동 감지)
    speech_timestamps = get_speech_timestamps_from_array(waveform_16k)
    current_time = time.time()
    print(f"VAD 처리 완료. (소요 시간: {current_time - last_time:.2f}초)")
    last_time = current_time
    if not speech_timestamps:
        print("오디오에서 음성을 감지할 수 없어요.")
        unload_ollama_model(config.llmModel)
        return

    # 4. Whisper (음성 -> 텍스트)
    last_segment = speech_timestamps[-1]
    end_idx = last_segment['end']
    utterance = waveform_16k[:end_idx]
    print("음성을 텍스트로 변환하고 있어요...")
    text = transcribe_sync(utterance)
    current_time = time.time()
    print(f"텍스트 변환 완료: -> {text} (소요 시간: {current_time - last_time:.2f}초)")
    last_time = current_time

    if not text:
        print("텍스트 변환 결과가 비어있어요.")
        unload_ollama_model(config.llmModel)
        return

    # 5. LLM 응답 생성 및 TTS/재생
    first_sentence_generated = False
    async for sentence in astream_call_response(USER, GUILD_ID, text):
        if sentence:
            if not first_sentence_generated:
                current_time = time.time()
                print(f"LLM 첫 문장 생성 완료. (소요 시간: {current_time - last_time:.2f}초)")
                last_time = current_time
                first_sentence_generated = True

            print(f"LLM 응답 문장: {sentence}")
            tts_file_path = generate_with_edge_tts(sentence)
            await play_audio_file(tts_file_path)

    # 6. 모델 언로드
    total_end_time = time.time()
    print(f"\n총 실행 시간: {total_end_time - total_start_time:.2f}초")
    unload_ollama_model(config.llmModel)
    print("처리가 완료되어 모델을 언로드했어요.")

if __name__ == "__main__":
    # 여기에 테스트할 오디오 파일 경로를 입력하세요.
    audio_file_to_test = "output_temp/test_input.wav"
    asyncio.run(process_audio_file(audio_file_to_test))