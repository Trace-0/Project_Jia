import asyncio
import threading
import uuid
from discord_interface.vad import get_speech_timestamps_from_array 
from discord_interface.faster_whisper_output import transcribe_sync
from discord_interface.espnet_tts_output import generate_tts
from LLM.langchain_llm import astream_call_response, generate_response
import logging
import torch
import torchaudio
import re

bot_instance = None

resampler_48_to_16 = torchaudio.transforms.Resample(orig_freq=48000, new_freq=16000)

def set_bot(bot):
    global bot_instance
    bot_instance = bot

def tts_and_play(text: str, guild: int, task_id: str):
    """TTS를 생성하고 디스코드 봇에 스트림 재생을 요청하는 함수"""
    if bot_instance and bot_instance.audio_stream_manager:
        try:
            audio_data = generate_tts(text)
            if bot_instance.loop and not bot_instance.loop.is_closed():
                try:
                    bot_instance.loop.call_soon_threadsafe(
                        bot_instance.audio_stream_manager.add_to_queue,
                        guild, task_id, audio_data
                    )
                except RuntimeError as e:
                    logging.error(f"[Pipeline] 이벤트 루프가 닫혀 큐에 추가할 수 없습니다: {e}")
            else:
                logging.error(f"[Pipeline] 봇의 이벤트 루프가 닫혔거나 사용할 수 없습니다.")
        except Exception as e:
            logging.error(f"[Pipeline] TTS 생성 및 재생 작업 큐에 추가 중 오류 발생: {e}")

def send_text_result(task_id: str, text: str, user: str):
    """디스코드 봇에 텍스트 결과 전송을 요청하는 함수"""
    if bot_instance and bot_instance.loop and not bot_instance.loop.is_closed():
        try:
            bot_instance.loop.call_soon_threadsafe(
                bot_instance.send_text_result_sync, task_id, text, user
            )
        except RuntimeError as e:
            logging.error(f"[Pipeline] 이벤트 루프가 닫혀 텍스트 결과를 보낼 수 없습니다: {e}")
    else:
        logging.error(f"[Pipeline] 봇의 이벤트 루프가 닫혔거나 사용할 수 없습니다.")

def send_text_to_channel(textchan_id: int, text: str, user: str):
    """특정 디스코드 텍스트 채널로 메시지를 전송하는 함수"""
    if not bot_instance or not bot_instance.loop or bot_instance.loop.is_closed():
        logging.error(f"[Pipeline] 봇의 이벤트 루프가 닫혔거나 사용할 수 없습니다.")
        return
    
    try:
        channel = bot_instance.get_channel(textchan_id)
        if channel and hasattr(channel, 'send'):
            bot_instance.loop.create_task(channel.send(f"[{user}]: {text}"))
        else:
            logging.error(f"[Pipeline] 채널 {textchan_id}을(를) 찾을 수 없거나 텍스트 채널이 아닙니다.")
    except Exception as e:
        logging.error(f"[Pipeline] 채널 전송 중 오류 발생: {e}")

def split_sentences(text: str) -> list:
    """문장을 , . ! ? 단위로 분리해 리스트로 반환합니다."""
    if not text:
        return []
    # 마침표나 물음표, 느낌표, 쉼표 뒤의 공백을 기준으로 문장 분할
    parts = re.split(r'(?<=[,\.!\?])\s+', text.strip())
    return [p.strip() for p in parts if p.strip()]


def generate_tts_buffers(text: str) -> list:
    """주어진 텍스트를 문장 단위로 TTS 생성"""
    buffers: list = []
    try:
        sentences = split_sentences(text)
        for s in sentences:
            try:
                buf = generate_tts(s)
                # ensure buffer is seeked to start
                try:
                    buf.seek(0)
                except Exception:
                    pass
                buffers.append(buf)
            except Exception as e:
                logging.error(f"[Pipeline] 문장별 TTS 생성 실패: {e}")
    except Exception as e:
        logging.error(f"[Pipeline] TTS 버퍼 생성 중 오류: {e}")
    return buffers


def tts_text_and_queue(text: str, guild: int, task_id: str):
    """텍스트 전체를 문장으로 분리해 TTS를 생성하고 봇 오디오 큐에 순차적으로 추가합니다."""
    if not bot_instance or not bot_instance.audio_stream_manager:
        logging.error("[Pipeline] 봇 인스턴스 또는 오디오 스트림 매니저가 없습니다.")
        return

    try:
        buffers = generate_tts_buffers(text)
        for buf in buffers:
            try:
                bot_instance.loop.call_soon_threadsafe(
                    bot_instance.audio_stream_manager.add_to_queue,
                    guild, task_id, buf
                )
            except RuntimeError as e:
                logging.error(f"[Pipeline] 이벤트 루프가 닫혀 큐에 추가할 수 없습니다: {e}")
    except Exception as e:
        logging.error(f"[Pipeline] 텍스트를 TTS로 변환하여 큐에 추가하는 중 오류 발생: {e}")

async def process_audio(task_id, user, guild, audio_data_48k, only_hear: bool = False, textchan_id: int = None):
    try:
        stream_task_id = str(uuid.uuid4())

        # 48kHz 오디오를 16kHz로 리샘플링
        try:
            audio_tensor_48k = torch.from_numpy(audio_data_48k)
            audio_tensor_16k = resampler_48_to_16(audio_tensor_48k)
            audio_data_16k = audio_tensor_16k.numpy()
        except Exception as e:
            logging.error(f"[Pipeline] 오디오 리샘플링 중 오류 발생: {e}")
            return

        speech_timestamps = get_speech_timestamps_from_array(audio_data_16k)
        if speech_timestamps:
            # 받은 음성 패킷에 대한 VAD 결과가 있다면
            last_segment = speech_timestamps[-1]
            end_idx = last_segment['end']
            logging.info(f"[{user}] 음성을 텍스트로 변환하고 있어요.")
            utterance = audio_data_16k[:end_idx]
            text = transcribe_sync(utterance)
            logging.info(f"[{user}] 텍스트 변환 결과가 나왔어요. -> {text}")
            if only_hear:
                # only_hear 모드: 특정 채널로 텍스트만 전송
                if textchan_id:
                    send_text_to_channel(textchan_id, text, user)
                else:
                    thread = threading.Thread(target=send_text_result, args=(stream_task_id, text, user))
                    thread.start()
            else:
                if text:
                    try:
                        async for sentence in astream_call_response(user, guild, text):
                            if sentence:
                                # 각 문장에 대해 TTS 생성 및 재생을 별도 스레드에서 실행
                                thread = threading.Thread(target=tts_and_play, args=(sentence, guild, stream_task_id))
                                thread.start()
                    except Exception as e:
                        logging.error(f"[Pipeline] TTS 및 재생 중 오류 발생: {e}")
    except Exception as e:
        logging.error(f"[Pipeline] 오디오 처리 중 오류 발생: {e}")

def run_audio_task(task_id, user, guild, audio_data_48k, only_hear: bool = False, textchan_id: int = None):
    if bot_instance and bot_instance.loop and not bot_instance.loop.is_closed():
        asyncio.run_coroutine_threadsafe(process_audio(task_id, user, guild, audio_data_48k, only_hear, textchan_id), bot_instance.loop)
    else:
        logging.error("[Pipeline] 봇의 이벤트 루프가 사용할 수 없습니다.")

def process_text(task_id, user, guild, text):
    result_text = generate_response(user, guild, text)
    
    if bot_instance:
        # 봇의 메인 루프에서 메시지 전송을 처리하도록 스케줄링
        bot_instance.loop.call_soon_threadsafe(
            bot_instance.send_text_result_sync, task_id, result_text
        )

def run_text_task(task_id, user, guild, text):
    threading.Thread(target=process_text, args=(task_id, user, guild, text), daemon=True).start()
