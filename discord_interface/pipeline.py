import asyncio
import threading
import queue
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

def send_text_result(task_id: str, text: str):
    """디스코드 봇에 텍스트 결과 전송을 요청하는 함수"""
    if bot_instance and bot_instance.loop and not bot_instance.loop.is_closed():
        try:
            bot_instance.loop.call_soon_threadsafe(
                bot_instance.send_text_result_sync, task_id, text
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

# === TTS 취소 신호 ===
# 길드별로 진행 중인 TTS 작업(취소 이벤트, 워커를 깨울 문장 큐)을 등록해두고,
# /jiastop 등에서 cancel_tts_tasks()로 한 번에 취소 신호를 보낼 수 있게 합니다.
_tts_cancel_lock = threading.Lock()
_tts_cancel_registry: dict[int, set] = {}

def _register_tts_cancel(guild: int, cancel_event: threading.Event, sentence_queue: queue.Queue = None):
    with _tts_cancel_lock:
        _tts_cancel_registry.setdefault(guild, set()).add((cancel_event, sentence_queue))

def _unregister_tts_cancel(guild: int, cancel_event: threading.Event, sentence_queue: queue.Queue = None):
    with _tts_cancel_lock:
        entries = _tts_cancel_registry.get(guild)
        if entries:
            entries.discard((cancel_event, sentence_queue))
            if not entries:
                del _tts_cancel_registry[guild]

def cancel_tts_tasks(guild: int) -> int:
    """해당 길드에서 진행 중인 모든 TTS 작업에 취소 신호를 보내고, 취소한 작업 수를 반환합니다."""
    with _tts_cancel_lock:
        entries = list(_tts_cancel_registry.get(guild, ()))
    for cancel_event, sentence_queue in entries:
        cancel_event.set()
        if sentence_queue is not None:
            sentence_queue.put(None)  # 큐 대기 중인 워커 깨우기
    if entries:
        logging.info(f"[Pipeline] 길드({guild})의 TTS 작업 {len(entries)}건에 취소 신호를 보냈습니다.")
    return len(entries)

def split_sentences(text: str) -> list:
    """문장을 , . ! ? 단위로 분리해 리스트로 반환합니다."""
    if not text:
        return []
    # 마침표나 물음표, 느낌표, 쉼표 뒤의 공백을 기준으로 문장 분할
    parts = re.split(r'(?<=[,\.!\?])\s+', text.strip())
    return [p.strip() for p in parts if p.strip()]


def tts_text_and_queue(text: str, guild: int, task_id: str):
    """텍스트를 문장 단위로 TTS 생성하는 즉시 봇 오디오 큐에 순차적으로 추가합니다."""
    if not bot_instance or not bot_instance.audio_stream_manager:
        logging.error("[Pipeline] 봇 인스턴스 또는 오디오 스트림 매니저가 없습니다.")
        return

    cancel_event = threading.Event()
    _register_tts_cancel(guild, cancel_event)
    try:
        for s in split_sentences(text):
            if cancel_event.is_set():  # 취소 신호: 남은 문장 생성 중단
                break
            try:
                buf = generate_tts(s)
            except Exception as e:
                logging.error(f"[Pipeline] 문장별 TTS 생성 실패: {e}")
                continue
            try:
                bot_instance.loop.call_soon_threadsafe(
                    bot_instance.audio_stream_manager.add_to_queue,
                    guild, task_id, buf
                )
            except RuntimeError as e:
                logging.error(f"[Pipeline] 이벤트 루프가 닫혀 큐에 추가할 수 없습니다: {e}")
    except Exception as e:
        logging.error(f"[Pipeline] 텍스트를 TTS로 변환하여 큐에 추가하는 중 오류 발생: {e}")
    finally:
        _unregister_tts_cancel(guild, cancel_event)

def _transcribe_48k(user: str, audio_data_48k) -> str:
    """48kHz 오디오를 리샘플링하고 VAD로 발화 구간을 찾아 텍스트로 변환합니다. (블로킹 작업)"""
    audio_tensor_48k = torch.from_numpy(audio_data_48k)
    audio_data_16k = resampler_48_to_16(audio_tensor_48k).numpy()

    speech_timestamps = get_speech_timestamps_from_array(audio_data_16k)
    if not speech_timestamps:
        return ""

    # 받은 음성 패킷에 대한 VAD 결과가 있다면 마지막 발화 구간 끝까지 잘라서 변환
    end_idx = speech_timestamps[-1]['end']
    logging.info(f"[{user}] 음성을 텍스트로 변환하고 있어요.")
    text = transcribe_sync(audio_data_16k[:end_idx])
    logging.info(f"[{user}] 텍스트 변환 결과가 나왔어요. -> {text}")
    return text

def _tts_worker(sentence_queue: queue.Queue, guild: int, stream_task_id: str, cancel_event: threading.Event):
    """문장을 받은 순서대로 TTS 생성 후 재생 큐에 추가하는 워커 (응답 1건당 1개)"""
    while not cancel_event.is_set():
        sentence = sentence_queue.get()
        if sentence is None or cancel_event.is_set():  # 종료/취소 신호
            break
        tts_and_play(sentence, guild, stream_task_id)

async def process_audio(user, guild, audio_data_48k, only_hear: bool = False, textchan_id: int = None):
    try:
        # 블로킹 작업(리샘플링/VAD/Whisper)이 봇 이벤트 루프를 멈추지 않도록 스레드에서 실행
        text = await asyncio.to_thread(_transcribe_48k, user, audio_data_48k)
        if not text:
            return

        if only_hear:
            # only_hear 모드: 특정 채널로 텍스트만 전송
            if textchan_id:
                send_text_to_channel(textchan_id, text, user)
            else:
                logging.warning(f"[Pipeline] 텍스트 채널이 지정되지 않아 변환 결과를 전달할 곳이 없습니다.")
            return

        # 같은 응답의 문장들이 순서대로 재생되도록 단일 워커가 순서대로 TTS 생성
        stream_task_id = str(uuid.uuid4())
        sentence_queue: queue.Queue = queue.Queue()
        cancel_event = threading.Event()
        _register_tts_cancel(guild, cancel_event, sentence_queue)
        worker = threading.Thread(target=_tts_worker, args=(sentence_queue, guild, stream_task_id, cancel_event), daemon=True)
        worker.start()
        try:
            async for sentence in astream_call_response(user, guild, text):
                # 취소돼도 응답 생성은 끝까지 진행해 대화 기록은 보존하고, 재생만 건너뜀
                if sentence and not cancel_event.is_set():
                    sentence_queue.put(sentence)
        except Exception as e:
            logging.error(f"[Pipeline] TTS 및 재생 중 오류 발생: {e}")
        finally:
            sentence_queue.put(None)  # 워커 종료 신호
            _unregister_tts_cancel(guild, cancel_event, sentence_queue)
    except Exception as e:
        logging.error(f"[Pipeline] 오디오 처리 중 오류 발생: {e}")

def run_audio_task(user, guild, audio_data_48k, only_hear: bool = False, textchan_id: int = None):
    if bot_instance and bot_instance.loop and not bot_instance.loop.is_closed():
        asyncio.run_coroutine_threadsafe(process_audio(user, guild, audio_data_48k, only_hear, textchan_id), bot_instance.loop)
    else:
        logging.error("[Pipeline] 봇의 이벤트 루프가 사용할 수 없습니다.")

def process_text(task_id, user, guild, text):
    result_text = generate_response(user, guild, text)
    send_text_result(task_id, result_text)

def run_text_task(task_id, user, guild, text):
    threading.Thread(target=process_text, args=(task_id, user, guild, text), daemon=True).start()
