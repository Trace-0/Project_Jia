import asyncio
import io
import threading
import queue
import time
import uuid
import discord
from discord_interface.vad import get_speech_timestamps_from_array
from discord_interface.faster_whisper_output import transcribe_sync
from discord_interface.espnet_tts_output import generate_tts
from LLM.langchain_llm import astream_call_response, generate_response
from config.config_manager import config
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

def play_sound_file(guild: int, file_path: str) -> tuple[bool, str]:
    """사운드보드 효과음 파일을 봇 오디오 재생 큐에 추가합니다. (성공 여부, 메시지)를 반환합니다.

    파일 경로 검증은 호출하는 쪽(LLM/langchain_tools/soundboard.py)에서 수행합니다.
    경로(str) 대신 파일 내용을 BytesIO로 읽어 큐에 넣는 이유: AudioStreamManager.after_play가
    문자열 경로로 받은 오디오를 임시 TTS 파일로 간주해 재생 후 삭제하기 때문입니다.
    """
    if not bot_instance or not bot_instance.audio_stream_manager:
        return False, "봇 인스턴스 또는 오디오 스트림 매니저가 없습니다."
    guild_obj = bot_instance.get_guild(guild)
    voice_client = guild_obj.voice_client if guild_obj else None
    if not voice_client or not voice_client.is_connected():
        return False, "지아가 음성 채널에 접속해 있지 않습니다."
    try:
        with open(file_path, "rb") as f:
            audio_buffer = io.BytesIO(f.read())
    except OSError as e:
        logging.error(f"[Pipeline] 효과음 파일을 읽지 못했어요: {e}")
        return False, f"효과음 파일을 읽을 수 없습니다: {e}"
    if not bot_instance.loop or bot_instance.loop.is_closed():
        return False, "봇의 이벤트 루프가 닫혔거나 사용할 수 없습니다."
    try:
        bot_instance.loop.call_soon_threadsafe(
            bot_instance.audio_stream_manager.add_to_queue,
            guild, str(uuid.uuid4()), audio_buffer
        )
    except RuntimeError as e:
        logging.error(f"[Pipeline] 이벤트 루프가 닫혀 효과음을 큐에 추가할 수 없습니다: {e}")
        return False, "봇의 이벤트 루프가 닫혀 재생할 수 없습니다."
    return True, "효과음을 재생 큐에 추가했습니다."

# 길드별로 마지막 텍스트 대화(/jia, autotalk)가 오간 채널. 생성된 이미지를 보낼 곳을 정할 때 사용합니다.
_active_text_channels: dict[int, int] = {}

def set_active_text_channel(guild: int, channel_id: int):
    """텍스트 대화가 시작된 채널을 기록합니다. (discordBot의 TextGen에서 호출)"""
    _active_text_channels[guild] = channel_id

def send_image_to_guild(guild: int, image_bytes: bytes, filename: str, caption: str = "", prefer_voice_channel: bool = False) -> tuple[bool, str]:
    """생성된 이미지를 길드의 적절한 채널에 첨부 파일로 전송합니다. (성공 여부, 메시지)를 반환합니다.

    prefer_voice_channel이 True면 접속 중인 음성 채널의 채팅에 우선 전송하고,
    아니면 마지막 텍스트 대화 채널 -> 음성 채널 채팅 순서로 시도합니다.
    """
    if not bot_instance or not bot_instance.loop or bot_instance.loop.is_closed():
        return False, "봇의 이벤트 루프가 닫혔거나 사용할 수 없습니다."

    guild_obj = bot_instance.get_guild(guild)
    voice_client = guild_obj.voice_client if guild_obj else None
    voice_channel = getattr(voice_client, "channel", None) if (voice_client and voice_client.is_connected()) else None
    text_channel = bot_instance.get_channel(_active_text_channels.get(guild, 0))

    # 음성 대화에서 요청됐으면 음성 채널 채팅 우선, 텍스트 대화에서 요청됐으면 그 채널 우선
    candidates = [voice_channel, text_channel] if prefer_voice_channel else [text_channel, voice_channel]
    channel = next((c for c in candidates if c is not None and hasattr(c, "send")), None)
    if channel is None:
        return False, "이미지를 보낼 채널을 찾지 못했습니다."

    try:
        file = discord.File(io.BytesIO(image_bytes), filename=filename)
        asyncio.run_coroutine_threadsafe(
            channel.send(content=caption or None, file=file), bot_instance.loop
        ).result(timeout=30)
    except Exception as e:
        logging.error(f"[Pipeline] 이미지 전송 중 오류 발생: {e}")
        return False, f"이미지 전송 중 오류가 발생했습니다: {e}"
    logging.info(f"[Pipeline] 생성된 이미지를 채널({channel})에 올렸어요. (guild={guild})")
    return True, "이미지를 채널에 올렸습니다."

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

# === 다인 대화 배칭 ===
# 발화를 길드별로 모아두고, 길드당 하나의 대화 워커가 배치 단위로 LLM 응답을 생성합니다.
# 지아가 응답을 생성/재생하는 동안 들어온 발화는 다음 배치로 묶여 한 번의 호출로 처리됩니다.
# (아래 자료구조는 모두 봇 이벤트 루프에서만 접근하므로 별도 잠금이 필요 없습니다)
_pending_utterances: dict[int, list[tuple[str, str]]] = {}
_convo_tasks: dict[int, asyncio.Task] = {}

# 지아의 음성 재생이 사용자 발화로 중단(barge-in)된 길드 표시.
# 다음 LLM 호출에 "직전 응답이 끊겼다"는 안내를 함께 전달한 뒤 해제됩니다.
_interrupted_guilds: set[int] = set()

# === 먼저 말 걸기 (proactive) ===
# 음성 채널이 proactive_idle_sec 동안 조용하면 지아가 먼저 말을 걸어볼지 LLM에 물어봅니다.
# 발화 큐에 센티널을 넣어 기존 대화 워커로 직렬 처리되므로 실제 발화와 충돌하지 않습니다.
PROACTIVE_SENTINEL = "__JIA_PROACTIVE__"
_proactive_tasks: dict[int, asyncio.Task] = {}
_last_activity: dict[int, float] = {}
# 지아가 먼저 말을 건 뒤 아직 아무도 대답하지 않은 길드. (사용자 발화가 올 때까지 다시 말 걸지 않음)
_proactive_waiting: set[int] = set()

def start_proactive_monitor(guild: int):
    """길드의 음성 채널 유휴 감시를 시작합니다. (봇 이벤트 루프에서 호출)"""
    _last_activity[guild] = time.monotonic()
    _proactive_waiting.discard(guild)
    task = _proactive_tasks.get(guild)
    if task is None or task.done():
        _proactive_tasks[guild] = asyncio.create_task(_proactive_monitor(guild))

def stop_proactive_monitor(guild: int):
    """길드의 음성 채널 유휴 감시를 중단합니다."""
    task = _proactive_tasks.pop(guild, None)
    if task and not task.done():
        task.cancel()
    _last_activity.pop(guild, None)
    _proactive_waiting.discard(guild)

async def _proactive_monitor(guild: int):
    """주기적으로 유휴 시간을 확인해, 충분히 조용하면 먼저 말 걸기 시도를 발화 큐에 넣는 워커"""
    try:
        while True:
            await asyncio.sleep(5)
            g = bot_instance.get_guild(guild) if bot_instance else None
            voice_client = g.voice_client if g else None
            if not voice_client or not voice_client.is_connected():
                break  # 음성 연결이 끊기면 감시 종료
            idle_sec = config.proactive_idle_sec
            if idle_sec <= 0 or guild in _proactive_waiting:
                continue
            # 채널에 사람이 없으면 타이머만 초기화 (들어오자마자 말 걸지 않도록)
            if not _get_voice_participants(guild):
                _last_activity[guild] = time.monotonic()
                continue
            # 응답 생성/재생 중이거나 처리 대기 발화가 있으면 유휴 상태가 아님
            manager = bot_instance.audio_stream_manager if bot_instance else None
            busy = bool(_pending_utterances.get(guild)) or (
                manager and (manager.playing.get(guild) or any(manager.streams.get(guild, {}).values()))
            )
            if busy:
                continue
            if time.monotonic() - _last_activity.get(guild, time.monotonic()) < idle_sec:
                continue
            # 먼저 말 걸기 시도: 센티널 발화를 큐에 넣어 기존 워커로 직렬 처리
            logging.info(f"[Pipeline:Proactive] 길드({guild})가 {idle_sec}초 동안 조용해서 먼저 말을 걸어볼게요.")
            _proactive_waiting.add(guild)
            _last_activity[guild] = time.monotonic()
            _pending_utterances.setdefault(guild, []).append((PROACTIVE_SENTINEL, ""))
            task = _convo_tasks.get(guild)
            if task is None or task.done():
                _convo_tasks[guild] = asyncio.create_task(_conversation_worker(guild))
    except asyncio.CancelledError:
        pass
    except Exception as e:
        logging.error(f"[Pipeline:Proactive] 유휴 감시 중 오류 발생: {e}")
    finally:
        _proactive_tasks.pop(guild, None)

def mark_playback_interrupted(guild: int):
    """재생이 사용자 발화로 중단되었음을 기록합니다."""
    _interrupted_guilds.add(guild)

def _pop_playback_interrupted(guild: int) -> bool:
    if guild in _interrupted_guilds:
        _interrupted_guilds.discard(guild)
        return True
    return False

def clear_pending_utterances(guild: int) -> int:
    """아직 처리되지 않은 대기 발화와 인터럽트 표시를 비우고, 비운 발화 수를 반환합니다. (/jiastop, /jialeave용)"""
    _interrupted_guilds.discard(guild)
    pending = _pending_utterances.get(guild)
    count = len(pending) if pending else 0
    if pending:
        pending.clear()
    return count

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

        # 사용자 발화가 들어왔으니 유휴 타이머를 초기화하고, 먼저 말 걸기 대기 상태를 해제
        _last_activity[guild] = time.monotonic()
        _proactive_waiting.discard(guild)

        # 발화를 길드별 대기열에 쌓고, 대화 워커가 없으면 새로 시작
        _pending_utterances.setdefault(guild, []).append((user, text))
        task = _convo_tasks.get(guild)
        if task is None or task.done():
            _convo_tasks[guild] = asyncio.create_task(_conversation_worker(guild))
    except Exception as e:
        logging.error(f"[Pipeline] 오디오 처리 중 오류 발생: {e}")

async def _conversation_worker(guild: int):
    """모인 발화를 배치 단위로 처리하는 길드별 단일 워커 (LLM 호출/재생 직렬화)"""
    try:
        while True:
            utterances = _pending_utterances.get(guild)
            if not utterances:
                break
            _pending_utterances[guild] = []
            await _respond_to_batch(guild, utterances)
            # 재생이 끝날 때까지 기다리는 동안 들어온 발화는 다음 배치로 묶임
            await _wait_for_playback(guild)
            # 지아의 발화가 끝난 시점부터 유휴 시간을 다시 계산
            _last_activity[guild] = time.monotonic()
    except Exception as e:
        logging.error(f"[Pipeline] 대화 워커 처리 중 오류 발생: {e}")

def _get_voice_participants(guild: int) -> list[str]:
    """봇이 접속한 음성 채널에 함께 있는 사용자(봇 제외) 이름 목록을 반환합니다."""
    try:
        g = bot_instance.get_guild(guild) if bot_instance else None
        voice_client = g.voice_client if g else None
        channel = getattr(voice_client, "channel", None)
        if not channel:
            return []
        return [m.name for m in channel.members if not m.bot]
    except Exception as e:
        logging.warning(f"[Pipeline] 음성 채널 참가자 목록 조회 실패: {e}")
        return []

async def _respond_to_batch(guild: int, utterances: list[tuple[str, str]]):
    """발화 배치 하나에 대해 LLM 응답을 스트리밍하며 TTS 재생 큐에 추가합니다."""
    # 먼저 말 걸기 센티널 분리: 실제 발화가 함께 들어왔다면 일반 응답을 우선함
    real_utterances = [u for u in utterances if u[0] != PROACTIVE_SENTINEL]
    proactive = (len(real_utterances) == 0) and (len(utterances) > 0) and utterances[0][0] == PROACTIVE_SENTINEL
    if not real_utterances and not proactive:
        return
    # 직전 응답 재생이 사용자 발화로 중단됐다면 이번 호출에서 LLM에 알림
    interrupted = _pop_playback_interrupted(guild)
    # 현재 음성 채널 참가자 목록 (LLM이 누구에게 하는 말인지 판단할 참고 자료)
    participants = _get_voice_participants(guild)
    # 같은 응답의 문장들이 순서대로 재생되도록 단일 워커가 순서대로 TTS 생성
    stream_task_id = str(uuid.uuid4())
    sentence_queue: queue.Queue = queue.Queue()
    cancel_event = threading.Event()
    _register_tts_cancel(guild, cancel_event, sentence_queue)
    worker = threading.Thread(target=_tts_worker, args=(sentence_queue, guild, stream_task_id, cancel_event), daemon=True)
    worker.start()
    try:
        async for sentence in astream_call_response(guild, real_utterances, interrupted=interrupted, participants=participants, proactive=proactive):
            # 취소돼도 응답 생성은 끝까지 진행해 대화 기록은 보존하고, 재생만 건너뜀
            if sentence and not cancel_event.is_set():
                sentence_queue.put(sentence)
    except Exception as e:
        logging.error(f"[Pipeline] TTS 및 재생 중 오류 발생: {e}")
    finally:
        sentence_queue.put(None)  # 워커 종료 신호
        # 이번 응답의 TTS 생성이 모두 끝날 때까지 대기 (다음 배치 응답과 재생 순서가 섞이지 않게)
        # 생성이 끝나기 전까지는 /jiastop 취소 신호를 받을 수 있도록 등록을 유지
        await asyncio.to_thread(worker.join)
        _unregister_tts_cancel(guild, cancel_event, sentence_queue)

async def _wait_for_playback(guild: int):
    """이 길드의 오디오 재생이 모두 끝날 때까지 대기합니다."""
    manager = bot_instance.audio_stream_manager if bot_instance else None
    if not manager:
        return
    while manager.playing.get(guild) or any(manager.streams.get(guild, {}).values()):
        await asyncio.sleep(0.2)

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
