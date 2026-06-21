import discord
from discord.ext import commands
from discord_interface import voice_receive
import logging
import time
import numpy as np
import audioop
from discord_interface.faster_whisper_output import reload_whisper_model
import asyncio
import threading
from config.config_manager import config
import os
import sys
from memory.RAG import get_rag_instance
from LLM.LLM_model_control import unload_ollama_model, load_ollama_model
from collections import deque
import uuid
from discord_interface import pipeline
from discord_interface.youtube_music import MusicTrack, build_music_queue, resolve_music_track

for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger(__name__)

logger.setLevel(logging.INFO)

bot_client = None

PCM_FRAME_BYTES = 3840  # 48kHz, 16-bit, stereo, 20ms
PCM_SILENCE = b"\x00" * PCM_FRAME_BYTES
MUSIC_FFMPEG_BEFORE_OPTIONS = "-reconnect 1 -reconnect_streamed 1 -reconnect_delay_max 5"
MUSIC_FFMPEG_OPTIONS = "-vn"


class MixedAudioSource(discord.AudioSource):
    """음악과 TTS/효과음을 하나의 PCM 스트림으로 섞어 보내는 AudioSource"""

    def __init__(self, manager: "AudioStreamManager", guild_id: int):
        self.manager = manager
        self.guild_id = guild_id

    def read(self) -> bytes:
        return self.manager.read_mixed_frame(self.guild_id)

    def is_opus(self) -> bool:
        return False


class AudioStreamManager:
    """길드별 오디오 스트림을 관리하는 클래스

    하나의 Discord VoiceClient는 한 번에 하나의 AudioSource만 재생할 수 있으므로,
    이 매니저는 길드별 믹서 AudioSource 하나를 재생하고 그 내부에서 음악과 TTS/효과음을 섞습니다.
    """

    def __init__(self, bot):
        self.bot = bot
        self.streams: dict[int, dict[str, deque]] = {}
        self.playing: dict[int, bool] = {}  # foreground(TTS/효과음) 재생 여부
        self._current_foreground: dict[int, discord.AudioSource | None] = {}
        self._music_sources: dict[int, discord.AudioSource] = {}
        self._music_titles: dict[int, str] = {}
        self._music_queues: dict[int, deque[MusicTrack]] = {}
        self._music_volumes: dict[int, float] = {}
        self._music_paused: set[int] = set()
        self._music_loading: set[int] = set()
        self._music_generation: dict[int, int] = {}
        self._mixer_active: dict[int, bool] = {}
        self._lock = threading.RLock()

    def add_to_queue(self, guild_id: int, task_id: str, audio_source):
        try:
            source = self._make_source(audio_source)
        except Exception as e:
            logger.error(f"오디오 소스 생성 실패: {e}")
            return

        with self._lock:
            if guild_id not in self.streams:
                self.streams[guild_id] = {}
                self.playing[guild_id] = False
            if task_id not in self.streams[guild_id]:
                self.streams[guild_id][task_id] = deque()
            self.streams[guild_id][task_id].append(source)
        logger.info(f"오디오 스트림 큐에 추가: 서버({guild_id})")

        self._schedule_mixer(guild_id)

    def _make_source(self, audio_source) -> discord.AudioSource:
        if isinstance(audio_source, discord.AudioSource):
            return audio_source
        if isinstance(audio_source, str):
            return discord.FFmpegPCMAudio(audio_source)
        return discord.FFmpegPCMAudio(audio_source, pipe=True)

    def _schedule_mixer(self, guild_id: int):
        if not self.bot.loop or self.bot.loop.is_closed():
            return
        self.bot.loop.create_task(self.ensure_mixer(guild_id))

    async def ensure_mixer(self, guild_id: int) -> tuple[bool, str]:
        with self._lock:
            if self._mixer_active.get(guild_id):
                return True, "믹서가 이미 실행 중입니다."
        guild = self.bot.get_guild(guild_id)
        voice_client = guild.voice_client if guild else None
        if not isinstance(voice_client, discord.VoiceClient) or not voice_client.is_connected():
            return False, "지아가 음성 채널에 접속해 있지 않습니다."
        if voice_client.is_playing():
            return False, "다른 오디오 소스가 이미 재생 중입니다."

        source = MixedAudioSource(self, guild_id)
        with self._lock:
            self._mixer_active[guild_id] = True
        voice_client.play(
            source,
            after=lambda e: self.bot.loop.call_soon_threadsafe(self._after_mixer, guild_id, e),
        )
        return True, "오디오 믹서를 시작했습니다."

    def _after_mixer(self, guild_id: int, error):
        if error:
            logger.error(f"오디오 믹서 재생 후 오류: {error}")
        with self._lock:
            self._mixer_active[guild_id] = False
            self.playing[guild_id] = False
            has_audio = self._has_foreground_locked(guild_id) or self._has_music_locked(guild_id)
        if has_audio and self.bot.loop and not self.bot.loop.is_closed():
            self.bot.loop.create_task(self.ensure_mixer(guild_id))

    def _has_foreground_locked(self, guild_id: int) -> bool:
        return bool(self._current_foreground.get(guild_id)) or any(self.streams.get(guild_id, {}).values())

    def _pop_next_foreground_locked(self, guild_id: int) -> discord.AudioSource | None:
        guild_streams = self.streams.get(guild_id, {})
        for task_id in list(guild_streams):
            queue = guild_streams[task_id]
            if queue:
                return queue.popleft()
            del guild_streams[task_id]
        return None

    def _cleanup_source(self, source):
        try:
            source.cleanup()
        except Exception:
            pass

    def _pad_frame(self, frame: bytes) -> bytes:
        if len(frame) >= PCM_FRAME_BYTES:
            return frame[:PCM_FRAME_BYTES]
        return frame + b"\x00" * (PCM_FRAME_BYTES - len(frame))

    def _scale_frame(self, frame: bytes, volume: float) -> bytes:
        volume = max(0.0, min(1.0, volume))
        if volume == 1.0:
            return frame
        return audioop.mul(frame, 2, volume)

    def _mix_frames(self, music_frame: bytes, foreground_frame: bytes, music_gain: float) -> bytes:
        mixed = PCM_SILENCE
        if music_frame:
            mixed = self._scale_frame(self._pad_frame(music_frame), music_gain)
        if foreground_frame:
            foreground = self._scale_frame(self._pad_frame(foreground_frame), 1.0)
            mixed = audioop.add(mixed, foreground, 2)
        return mixed

    def read_mixed_frame(self, guild_id: int) -> bytes:
        with self._lock:
            foreground_frame = b""
            while True:
                foreground = self._current_foreground.get(guild_id)
                if foreground is None:
                    foreground = self._pop_next_foreground_locked(guild_id)
                    self._current_foreground[guild_id] = foreground
                if foreground is None:
                    self.playing[guild_id] = False
                    break
                foreground_frame = foreground.read()
                if foreground_frame:
                    self.playing[guild_id] = True
                    break
                self._cleanup_source(foreground)
                self._current_foreground[guild_id] = None
                self.playing[guild_id] = False

            music_frame = b""
            music = self._music_sources.get(guild_id)
            if music and guild_id not in self._music_paused:
                music_frame = music.read()
                if not music_frame:
                    self._cleanup_source(music)
                    self._clear_current_music_locked(guild_id)
                    self._schedule_next_music(guild_id)

            music_active = self._has_music_locked(guild_id)
            foreground_active = bool(foreground_frame) or self._has_foreground_locked(guild_id)
            if not foreground_frame and not music_frame and not music_active:
                return b""

            normal_volume = self._music_volumes.get(guild_id, config.music_volume)
            music_gain = config.music_duck_volume if foreground_active else normal_volume
            return self._mix_frames(music_frame, foreground_frame, music_gain)

    def stop_foreground(self, guild_id: int) -> int:
        with self._lock:
            count = sum(len(q) for q in self.streams.get(guild_id, {}).values())
            if self._current_foreground.get(guild_id) is not None:
                count += 1
                self._cleanup_source(self._current_foreground[guild_id])
            self.streams.get(guild_id, {}).clear()
            self._current_foreground[guild_id] = None
            self.playing[guild_id] = False
            return count

    def _has_music_locked(self, guild_id: int) -> bool:
        return (
            self._music_sources.get(guild_id) is not None
            or bool(self._music_queues.get(guild_id))
            or guild_id in self._music_loading
        )

    def _clear_current_music_locked(self, guild_id: int):
        self._music_sources.pop(guild_id, None)
        self._music_titles.pop(guild_id, None)

    def _next_music_generation_locked(self, guild_id: int) -> int:
        generation = self._music_generation.get(guild_id, 0) + 1
        self._music_generation[guild_id] = generation
        return generation

    def _schedule_next_music(self, guild_id: int):
        if not self.bot.loop or self.bot.loop.is_closed():
            return
        with self._lock:
            if self._music_sources.get(guild_id) is not None:
                return
            if guild_id in self._music_loading or guild_id in self._music_paused:
                return
            if not self._music_queues.get(guild_id):
                return
            self._music_loading.add(guild_id)
            generation = self._music_generation.get(guild_id, 0)
        self.bot.loop.create_task(self._start_next_music_track(guild_id, generation))

    async def _start_next_music_track(self, guild_id: int, generation: int):
        with self._lock:
            queue = self._music_queues.get(guild_id)
            if (
                self._music_generation.get(guild_id, 0) != generation
                or self._music_sources.get(guild_id) is not None
                or guild_id in self._music_paused
                or not queue
            ):
                self._music_loading.discard(guild_id)
                return
            track = queue.popleft()

        source = None
        resolved_track = track
        try:
            loop = asyncio.get_running_loop()
            resolved_track = await loop.run_in_executor(None, resolve_music_track, track)
            source = discord.FFmpegPCMAudio(
                resolved_track.stream_url,
                before_options=MUSIC_FFMPEG_BEFORE_OPTIONS,
                options=MUSIC_FFMPEG_OPTIONS,
            )
        except Exception as e:
            logger.error(f"[Discord:Music] 다음 곡을 준비하지 못했어요: {track.title} / {e}")

        should_schedule_next = False
        with self._lock:
            self._music_loading.discard(guild_id)
            if self._music_generation.get(guild_id, 0) != generation:
                if source:
                    self._cleanup_source(source)
                return
            if source is None:
                should_schedule_next = bool(self._music_queues.get(guild_id))
            else:
                self._music_sources[guild_id] = source
                self._music_titles[guild_id] = resolved_track.title
                self._music_volumes.setdefault(guild_id, max(0.0, min(1.0, config.music_volume)))

        if source is None and should_schedule_next:
            self._schedule_next_music(guild_id)
        self._schedule_mixer(guild_id)

    def start_music_tracks(self, guild_id: int, tracks: list[MusicTrack]) -> tuple[bool, str]:
        if not tracks:
            return False, "재생할 유튜브 음악을 찾지 못했습니다."

        with self._lock:
            self._next_music_generation_locked(guild_id)
            old_source = self._music_sources.pop(guild_id, None)
            if old_source:
                self._cleanup_source(old_source)
            self._music_titles.pop(guild_id, None)
            self._music_queues[guild_id] = deque(tracks)
            self._music_volumes[guild_id] = max(0.0, min(1.0, config.music_volume))
            self._music_paused.discard(guild_id)
            self._music_loading.discard(guild_id)

        self._schedule_next_music(guild_id)
        first = tracks[0].title
        if len(tracks) == 1:
            return True, f"유튜브 음악 재생을 준비할게요: {first}"
        return True, f"유튜브 재생목록 {len(tracks)}곡을 재생 큐에 넣었어요. 첫 곡: {first}"

    def queue_music_tracks(self, guild_id: int, tracks: list[MusicTrack]) -> tuple[bool, str]:
        if not tracks:
            return False, "추가할 유튜브 음악을 찾지 못했습니다."

        with self._lock:
            if guild_id not in self._music_queues:
                self._music_queues[guild_id] = deque()
            self._music_queues[guild_id].extend(tracks)
            self._music_volumes.setdefault(guild_id, max(0.0, min(1.0, config.music_volume)))
            should_start = self._music_sources.get(guild_id) is None and guild_id not in self._music_loading

        if should_start:
            self._schedule_next_music(guild_id)
        first = tracks[0].title
        if len(tracks) == 1:
            return True, f"대기열에 추가했어요: {first}"
        return True, f"대기열에 {len(tracks)}곡을 추가했어요. 첫 곡: {first}"

    def stop_music(self, guild_id: int) -> tuple[bool, str]:
        with self._lock:
            self._next_music_generation_locked(guild_id)
            source = self._music_sources.pop(guild_id, None)
            title = self._music_titles.pop(guild_id, None)
            queued = len(self._music_queues.get(guild_id, ()))
            loading = guild_id in self._music_loading
            self._music_queues.pop(guild_id, None)
            self._music_volumes.pop(guild_id, None)
            self._music_paused.discard(guild_id)
            self._music_loading.discard(guild_id)
        if source is None and not queued and not loading:
            return False, "재생 중인 음악이 없습니다."
        if source:
            self._cleanup_source(source)
        if queued or loading:
            return True, f"음악을 멈췄어요. 대기열 {queued}곡도 비웠습니다."
        return True, f"음악을 멈췄어요: {title}"

    def pause_music(self, guild_id: int) -> tuple[bool, str]:
        with self._lock:
            if not self._has_music_locked(guild_id):
                return False, "재생 중인 음악이 없습니다."
            self._music_paused.add(guild_id)
        return True, "음악을 일시정지했어요."

    def resume_music(self, guild_id: int) -> tuple[bool, str]:
        with self._lock:
            if not self._has_music_locked(guild_id):
                return False, "재생 중인 음악이 없습니다."
            self._music_paused.discard(guild_id)
            should_start = self._music_sources.get(guild_id) is None
        if should_start:
            self._schedule_next_music(guild_id)
        self._schedule_mixer(guild_id)
        return True, "음악을 다시 재생할게요."

    def set_music_volume(self, guild_id: int, volume: float) -> tuple[bool, str]:
        volume = max(0.0, min(1.0, volume))
        with self._lock:
            if not self._has_music_locked(guild_id):
                return False, "재생 중인 음악이 없습니다."
            self._music_volumes[guild_id] = volume
        return True, f"음악 볼륨을 {volume:.2f}로 설정했어요."

    def skip_music(self, guild_id: int) -> tuple[bool, str]:
        with self._lock:
            if not self._has_music_locked(guild_id):
                return False, "재생 중인 음악이 없습니다."
            self._next_music_generation_locked(guild_id)
            source = self._music_sources.pop(guild_id, None)
            title = self._music_titles.pop(guild_id, None)
            self._music_loading.discard(guild_id)
            has_next = bool(self._music_queues.get(guild_id))
        if source:
            self._cleanup_source(source)
        if has_next:
            self._schedule_next_music(guild_id)
            return True, f"현재 곡을 건너뛰었어요: {title or '준비 중인 곡'}"
        return True, f"현재 곡을 건너뛰었어요. 남은 대기열은 없습니다: {title or '준비 중인 곡'}"

    def music_status(self, guild_id: int) -> str:
        with self._lock:
            title = self._music_titles.get(guild_id)
            queued = len(self._music_queues.get(guild_id, ()))
            loading = guild_id in self._music_loading
            if not title and loading:
                return f"다음 유튜브 음악을 준비 중입니다. (대기열 {queued}곡)"
            if not title and queued:
                state = "일시정지" if guild_id in self._music_paused else "대기 중"
                return f"유튜브 음악 {state}입니다. (대기열 {queued}곡)"
            if not title:
                return "재생 중인 음악이 없습니다."
            state = "일시정지" if guild_id in self._music_paused else "재생 중"
            volume = self._music_volumes.get(guild_id, config.music_volume)
        return f"음악 {state}: {title} (대기열 {queued}곡, 볼륨 {volume:.2f}, 말할 때 {config.music_duck_volume:.2f})"

    def has_music(self, guild_id: int) -> bool:
        with self._lock:
            return self._has_music_locked(guild_id)

    def stop_all(self, guild_id: int):
        self.stop_foreground(guild_id)
        self.stop_music(guild_id)

class JiaBot(commands.Bot):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.textdict: dict = {} # 텍스트 작업의 task_id와 context를 매핑하는 딕셔너리
        self.autotalk_channels: dict[int, set[int]] = {} # 길드 ID와 자동 대화 채널 ID 집합을 매핑하는 딕셔너리
        self.def_channel: discord.TextChannel | None = None # 기본 로그 채널
        self.audio_stream_manager: AudioStreamManager | None = None

    async def setup_hook(self):
        self.audio_stream_manager = AudioStreamManager(self)

    def send_text_result_sync(self, task_id: str, result: str):
        """스레드에서 호출되어 봇의 루프에 메시지 전송 태스크를 생성하는 함수"""
        # 작업당 한 번만 전송하므로 매핑을 꺼내면서 제거 (누적 방지)
        ctx = self.textdict.pop(task_id, None)
        if ctx and result:
            # 비동기 함수인 ctx.send를 메인 루프에서 실행되도록 예약
            self.loop.create_task(ctx.send(result))

def bot():
    try:
        intents = discord.Intents.default()
        intents.message_content = True
        intents.voice_states = True
        intents.guilds = True
        intents.guild_messages = True
        intents.messages = True


        bot = JiaBot(command_prefix="/", intents=intents)
        global bot_client
        bot_client = bot

        def llm_still_in_use(exclude_guild_id: int | None = None) -> bool:
            """자동 대화 채널이나 음성 연결이 남아 있어 LLM이 아직 필요한지 확인합니다.

            exclude_guild_id: 지금 연결을 해제 중인 길드는 사용 중으로 치지 않습니다.
            """
            if any(bot.autotalk_channels.values()):
                return True
            for vc in bot.voice_clients:
                if exclude_guild_id is not None and vc.guild.id == exclude_guild_id:
                    continue
                return True
            return False

        async def is_command_whitelisted(ctx) -> bool:
            try:
                whitelist = {int(user_id) for user_id in config.command_whitelist_user_ids}
            except (TypeError, ValueError):
                whitelist = set()
            if getattr(ctx.author, "id", None) in whitelist:
                return True
            try:
                return await bot.is_owner(ctx.author)
            except Exception:
                return False

        def has_server_admin_permission(ctx) -> bool:
            permissions = getattr(ctx.author, "guild_permissions", None)
            return bool(
                getattr(permissions, "administrator", False)
                or getattr(permissions, "manage_guild", False)
            )

        async def require_command_access(ctx, level: str, command_name: str) -> bool:
            if await is_command_whitelisted(ctx):
                return True
            if level == "admin" and has_server_admin_permission(ctx):
                return True

            if level == "owner":
                message = f"`{command_name}`은 봇 소유자 또는 명령어 화이트리스트 유저만 사용할 수 있어요."
            else:
                message = f"`{command_name}`은 서버 관리자 또는 명령어 화이트리스트 유저만 사용할 수 있어요."
            await ctx.send(message)
            if bot.def_channel:
                guild_name = getattr(getattr(ctx, "guild", None), "name", "DM")
                channel_name = getattr(getattr(ctx, "channel", None), "name", "unknown")
                await bot.def_channel.send(f"{guild_name}/{channel_name} : 권한 부족 ({command_name}, {ctx.author})")
            return False

        def on_config_changed(changed: dict):
            """settings.toml 자동 리로드 후 호출되는 후처리 콜백. (config 워처 스레드에서 실행)

            단순 값 변경은 reload만으로 즉시 반영되지만, 모델 관련 설정은
            바뀐 항목에 해당하는 모델만 골라서 다시 로드합니다.
            """
            from LLM.langchain_llm import reload_llm
            from discord_interface.espnet_tts_output import reload_tts_model
            if {"whisper_model", "whisper_device", "whisper_compute_type"} & changed.keys():
                reload_whisper_model()
            if "tts_model" in changed:
                reload_tts_model()
            if "llm_tools" in changed.keys():
                # MCP 서버 구성이 바뀌면 클라이언트를 다시 만듦 (에이전트 캐시는 아래 reload_llm이 비움)
                from LLM.langchain_tools.mcp_manager import rebuild_client
                rebuild_client()
            if {"llmModel", "llmNumCtx", "llmSystemPrompt", "llm_tools",
                "llm_provider", "llm_api_key", "llm_api_base"} & changed.keys():
                reload_llm()
                if "llmModel" in changed:
                    old_model, _ = changed["llmModel"]
                    unload_ollama_model(old_model)  # 이전 모델을 Ollama 메모리에서 내림 (외부 API면 무시됨)
            elif {"comfyui_url", "comfyui_checkpoint"} & changed.keys():
                # 이미지 생성 도구는 에이전트 생성 시점에 등록되므로 캐시를 비워 도구 목록을 다시 구성
                reload_llm()

        class TextGen():
            def __init__(self, ctx):
                self.ctx = ctx

            def text_gen_requestor(self, prompt):
                task_id = str(uuid.uuid4())
                # 작업 시작 전에 응답을 보낼 채널(ctx)을 먼저 등록해 응답 유실을 방지
                bot.textdict[task_id] = self.ctx
                # 생성된 이미지 등을 보낼 수 있도록 대화 중인 채널을 기록
                pipeline.set_active_text_channel(self.ctx.guild.id, self.ctx.channel.id)
                # 파이프라인 작업 실행
                pipeline.run_text_task(task_id, self.ctx.author.name, self.ctx.channel.guild.id, prompt)
                return task_id

        # 음성 데이터를 수신하고 처리하는 클래스
        class TranscriptionSink(voice_receive.AudioSink):
            def __init__(self, guild: discord.Guild, voice_channel: discord.VoiceChannel, only_hear: bool = False, textchan_id: int = None):
                super().__init__()
                self.buffers: dict[int, list[np.ndarray]] = {} # 사용자별 오디오 청크를 저장하는 버퍼
                self._current_task: dict[int, asyncio.Task] = {} # 사용자별 발화 종료 감시 태스크
                self.last_packet_time: dict[int, float] = {} # 사용자별 마지막 패킷 수신 시각 (monotonic)
                # 발화 종료 대기 시간과 barge-in 판정 시간은 config(voice_timeout_sec, voice_interrupt_speech_sec)에서
                # 사용 시점에 읽으므로 settings.toml 수정만으로 즉시 반영됩니다.
                self._interrupt_fired: set[int] = set() # 이번 발화에서 이미 인터럽트를 보낸 사용자
                self.guild = guild
                self.loop = bot.loop
                self.voice_channel = voice_channel
                self.only_hear = only_hear
                self.textchan_id = textchan_id

            def wants_opus(self) -> bool:
                # OPUS 패킷이 아닌 PCM 오디오(float)를 받아야 함
                return False

            def write(self, user: discord.User, data: voice_receive.VoiceData):
                if not user or user.bot:
                    return

                # 마지막 패킷 시각만 기록하고, 감시 태스크가 없을 때만 새로 만듦
                # (무음 시에는 패킷 자체가 오지 않으므로 시각 경과만으로 발화 종료를 감지 가능)
                self.last_packet_time[user.id] = time.monotonic()
                task = self._current_task.get(user.id)
                if task is None or task.done():
                    def runner():
                        t = self._current_task.get(user.id)
                        if t is None or t.done():
                            self._current_task[user.id] = self.loop.create_task(self.packet_timeout(user))

                    self.loop.call_soon_threadsafe(runner)

                # data.pcm은 48kHz, 16-bit, 2-channel PCM 형식
                audio_data = np.frombuffer(data.pcm, dtype=np.int16)
                if audio_data.size == 0:
                    return

                # 2채널 음성이 1차원 배열로 입력되기 때문에 2차원 배열로 변환
                audio_data = audio_data.reshape(-1, 2)

                # 모노로 변환
                mono_data = audio_data.mean(axis=1)

                # float32 형식으로 변환
                float_data = mono_data.astype(np.float32) / 32768.0

                # 버퍼에 추가
                if user.id not in self.buffers:
                    self.buffers[user.id] = []
                self.buffers[user.id].append(float_data)

                # 지아가 말하는 중에 사용자의 발화가 일정 시간 이상 지속되면 재생을 중단 (barge-in)
                if not self.only_hear and user.id not in self._interrupt_fired:
                    buffered_sec = sum(chunk.size for chunk in self.buffers[user.id]) / 48000.0
                    if buffered_sec >= config.voice_interrupt_speech_sec and self._tts_audio_active():
                        self._interrupt_fired.add(user.id)
                        self.loop.call_soon_threadsafe(self._interrupt_playback, user)

            def _tts_audio_active(self) -> bool:
                """이 길드에서 TTS 오디오가 재생 중이거나 재생 대기 중인지 확인합니다."""
                manager = bot.audio_stream_manager
                if not manager:
                    return False
                return bool(manager.playing.get(self.guild.id)) or any(manager.streams.get(self.guild.id, {}).values())

            def _interrupt_playback(self, user: discord.User):
                """지아의 음성 재생을 중단하고, 다음 LLM 호출에서 인터럽트를 인지하도록 표시합니다."""
                if not self._tts_audio_active():
                    return
                # 진행 중인 TTS 생성을 취소하고 foreground(TTS/효과음)만 비움. 배경 음악은 유지합니다.
                pipeline.cancel_tts_tasks(self.guild.id)
                pipeline.stop_foreground_audio(self.guild.id)
                pipeline.mark_playback_interrupted(self.guild.id)
                logger.info(f"[Discord:Interrupt] [{user.name}]의 발화가 이어져서 재생을 중단했어요.")

            async def packet_timeout(self, user: discord.User):
                try:
                    # 마지막 패킷 이후 voice_timeout_sec 동안 새 패킷이 없을 때까지 대기
                    while True:
                        timeout_sec = config.voice_timeout_sec
                        elapsed = time.monotonic() - self.last_packet_time.get(user.id, 0.0)
                        if elapsed >= timeout_sec:
                            break
                        await asyncio.sleep(timeout_sec - elapsed)
                    await self.send_vad_and_whisper(user)
                except asyncio.CancelledError:
                    pass

            async def send_vad_and_whisper(self, user: discord.User):
                # 사용자의 버퍼를 가져옴
                user_buffer = self.buffers.pop(user.id, [])
                self._interrupt_fired.discard(user.id)  # 다음 발화에서 다시 인터럽트 판정 가능하게 초기화
                if not user_buffer:
                    return

                try:
                    full_audio = np.concatenate(user_buffer)
                    # 파이프라인 작업 실행
                    pipeline.run_audio_task(user.name, self.guild.id, full_audio, self.only_hear, self.textchan_id)
                except Exception as e:
                    logger.error(f"[{user.name}] VAD(대화 감지) 또는 Whisper(텍스트화)에서 오류가 발생했어요. :( \n   -> {e}")

            def cleanup(self):
                # 대기 중인 모든 타임아웃 작업을 취소하고 버퍼를 비움
                for task in self._current_task.values():
                    if task and not task.done():
                        task.cancel()
                self._current_task.clear()
                self.buffers.clear()
                self._interrupt_fired.clear()
                # 다른 곳에서 LLM을 쓰는 중이면 언로드하지 않음 (오디오 수신 스레드에서 호출되므로 동기 호출 가능)
                if not llm_still_in_use(exclude_guild_id=self.guild.id):
                    unload_ollama_model(config.llmModel)

        async def get_channel_sure() -> discord.TextChannel | None:
            await bot.wait_until_ready()

            print("로그 채널 연결을 시도하고 있어요")
            if config.debug_text_channel:
                channel = bot.get_channel(config.debug_text_channel)
                if channel:
                    if isinstance(channel, discord.TextChannel):
                        return channel
                    else:
                        print("잘못된 채널 타입이에요")
                        return None
                else:
                    print("로그 채널 연결에 실패했어요")
                    return None
            else:
                print("디버그용 텍스트 채널이 설정되지 않았어요.")
                return None
            
        # ==== 커맨드 ====
        @bot.command(name="jiajoin", description="지아를 음성 채널에 초대해요")
        async def jiajoin(ctx):
            if ctx.author.voice is None:
                await ctx.send("먼저 음성 채널에 들어가주세요")
                if bot.def_channel:
                    await bot.def_channel.send(f"{ctx.channel.guild.name}/{ctx.channel.name} : 음성 채널 접속 실패 (요청자가 음성 채널 연결 상태가 아님)")
                return
            # RAG 인덱스 동기화
            get_rag_instance(ctx.guild.id).sync_all_metadata_to_faiss()
            # 모델 로드 동안 이벤트 루프가 멈추지 않도록 백그라운드 스레드에서 실행
            await asyncio.to_thread(load_ollama_model, config.llmModel)
            voice_channel = ctx.author.voice.channel
            # DAVE(E2EE) 호환 음성 수신 클라이언트로 연결
            voice_client = await voice_channel.connect(cls=voice_receive.VoiceRecvClient)
            logger.info(f"[Discord:Join] 음성 채널에 접속할게요 -> {voice_channel.name}")
            # 음성 수신을 위한 싱크 생성 및 리스닝 시작
            sink = TranscriptionSink(ctx.channel.guild, voice_channel)
            voice_client.listen(sink)
            # 채널이 조용할 때 먼저 말을 걸어보는 유휴 감시 시작 (proactive_idle_sec가 0이면 동작 안 함)
            pipeline.start_proactive_monitor(ctx.guild.id)
            if config.join_reply:
                await ctx.send("음성 채널에 접속할게요")
            if bot.def_channel:
                await bot.def_channel.send(f"{ctx.channel.guild.name}/{ctx.channel.name} : {ctx.author.voice.channel.guild.name}/{ctx.author.voice.channel.name}(으)로의 접속 요청")

        @bot.command(name="jialeave", description="지아를 음성 채널에서 내보네요")
        async def jialeave(ctx):
            if ctx.voice_client:
                # RAG 인덱스 저장
                get_rag_instance(ctx.guild.id).save_all()
                # 먼저 말 걸기 유휴 감시 중단
                pipeline.stop_proactive_monitor(ctx.guild.id)
                # 처리 대기 중인 발화 정리
                pipeline.clear_pending_utterances(ctx.guild.id)
                # 오디오 스트림 정리
                if bot.audio_stream_manager:
                    bot.audio_stream_manager.stop_all(ctx.guild.id)
                await ctx.voice_client.disconnect()
                if config.leave_reply:
                    await ctx.send("음성 채널에서 나갈게요")
                if bot.def_channel:
                    await bot.def_channel.send(f"{ctx.channel.guild.name}/{ctx.channel.name} : 음성 연결 해제 요청")
            else:
                await ctx.send("지금은 음성 채널에 접속한 상태가 아니에요")
                if bot.def_channel:
                    await bot.def_channel.send(f"{ctx.channel.guild.name}/{ctx.channel.name} : 음성 연결 해제 실패 (현재 연결 상태가 아님)")

        @bot.event
        async def on_ready():
            print('봇 온라인!')
            # 파이프라인에 봇 인스턴스 전달
            pipeline.set_bot(bot)
            
            bot.def_channel = await get_channel_sure()
            if bot.def_channel:
                print("로그 채널 연결에 성공했어요")
                await bot.def_channel.send("Project Jia 온라인!")

        @bot.command(name="jiaping", description="지아가 응답할 수 있는지 확인해요")
        async def jiaping(ctx):
            await ctx.send("pong!")

        @bot.command(name="jiareload", description="지아의 설정을 다시 불러와요")
        async def jiareload(ctx):
            if not await require_command_access(ctx, "owner", "/jiareload"):
                return
            try:
                from config.config_manager import config as global_config
                from LLM.langchain_llm import reload_llm
                from discord_interface.espnet_tts_output import reload_tts_model
                old_llm_model = global_config.llmModel
                global_config.reload()
                reload_whisper_model()  # Whisper 모델 재로딩
                reload_tts_model()  # TTS 모델 재로딩
                reload_llm()  # LLM과 시스템 프롬프트 재로딩, 에이전트 캐시 초기화
                if old_llm_model != global_config.llmModel:
                    # 모델이 바뀌었으면 이전 모델을 Ollama 메모리에서 내려요
                    await asyncio.to_thread(unload_ollama_model, old_llm_model)
                get_rag_instance(ctx.guild.id).save_all() # 변경된 사항이 있을 수 있으니 저장
                get_rag_instance(ctx.guild.id).sync_all_metadata_to_faiss() # RAG 인덱스 재동기화
                await ctx.send("설정이 성공적으로 재로딩되었어요")
                if bot.def_channel:
                    await bot.def_channel.send(f"{ctx.channel.guild.name}/{ctx.channel.name} : 설정 재로딩 요청")
                logger.info("[Discord:Reload] 설정을 다시 불러왔어요.")
            except Exception as e:
                await ctx.send(f"설정 재로딩 중 오류 발생: {e}")
                if bot.def_channel:
                    await bot.def_channel.send(f"{ctx.channel.guild.name}/{ctx.channel.name} : 설정 재로딩 실패 ({e})")
                logger.error(f"[Discord:Reload] 설정을 다시 불러오는 과정에 오류가 발생했어요. :(\n   -> {e}")

        @bot.command(name="jiasavesetting", description="지아의 설정을 저장해요")
        async def jiasavesetting(ctx):
            if not await require_command_access(ctx, "owner", "/jiasavesetting"):
                return
            try:
                from config.config_manager import config as global_config
                global_config.save_setting()
                await ctx.send("설정이 성공적으로 저장되었어요")
                if bot.def_channel:
                    await bot.def_channel.send(f"{ctx.channel.guild.name}/{ctx.channel.name} : 설정 저장 요청")
                logger.info("[Discord:Save_Setting] 설정을 저장했어요.")
            except Exception as e:
                await ctx.send(f"설정 저장 중 오류 발생: {e}")
                if bot.def_channel:
                    await bot.def_channel.send(f"{ctx.channel.guild.name}/{ctx.channel.name} : 설정 저장 실패 ({e})")
                logger.error(f"[Discord:Save_Setting] 설정을 저장하는 과정에 오류가 발생했어요. :(\n   -> {e}")

        @bot.command(name="jiaunloadmodel", description="지아의 LLM 모델을 메모리에서 내려요.")
        async def jiaunloadmodel(ctx):
            if not await require_command_access(ctx, "owner", "/jiaunloadmodel"):
                return
            try:
                model_name = config.llmModel
                # 백그라운드 스레드에서 모델 언로드 함수 실행
                await asyncio.to_thread(unload_ollama_model, model_name)
                await ctx.send(f"모델 '{model_name}'을(를) 메모리에서 언로드했어요. 다음 요청 시 다시 로드됩니다.")
                if bot.def_channel:
                    await bot.def_channel.send(f"{ctx.channel.guild.name}/{ctx.channel.name} : 모델 언로드 요청 ({model_name})")
            except Exception as e:
                await ctx.send(f"모델 언로드 중 오류 발생: {e}")
                if bot.def_channel:
                    await bot.def_channel.send(f"{ctx.channel.guild.name}/{ctx.channel.name} : 모델 언로드 실패 ({e})")
                logger.error(f"[Discord:Unload_Model] 모델을 언로드하는 과정에 오류가 발생했어요. :(\n   -> {e}")

        @bot.command(name="jiarestart", description="지아를 재시작해요. 재시작이 필요한 설정 변경(임베딩 모델 등)을 반영할 때 사용해요.")
        async def jiarestart(ctx):
            if not await require_command_access(ctx, "owner", "/jiarestart"):
                return
            try:
                await ctx.send("재시작할게요. 잠시 후에 다시 만나요!")
                if bot.def_channel:
                    await bot.def_channel.send(f"{ctx.channel.guild.name}/{ctx.channel.name} : 재시작 요청")
                logger.info("[Discord:Restart] 재시작 요청을 받아 프로세스를 다시 시작해요.")
                # 저장되지 않은 기억을 디스크에 저장
                from memory.RAG import rag_instances
                for rag in rag_instances.values():
                    rag.save_all()
                # 음성 연결을 정리
                for vc in list(bot.voice_clients):
                    try:
                        await vc.disconnect()
                    except Exception:
                        pass
                # 현재 프로세스를 같은 명령으로 교체해 재시작 (Windows에서는 새 프로세스 실행 후 현재 프로세스 종료)
                args = [sys.executable] + sys.argv
                if os.name == "nt":
                    args = [f'"{a}"' if " " in a else a for a in args]
                os.execv(sys.executable, args)
            except Exception as e:
                await ctx.send(f"재시작 중 오류 발생: {e}")
                if bot.def_channel:
                    await bot.def_channel.send(f"{ctx.channel.guild.name}/{ctx.channel.name} : 재시작 실패 ({e})")
                logger.error(f"[Discord:Restart] 재시작 과정에 오류가 발생했어요. :(\n   -> {e}")

        @bot.command(name="jia", description="지아와 대화해요")
        async def jia(ctx, *, prompt: str):
            try:
                async with ctx.channel.typing():
                    get_rag_instance(ctx.guild.id).sync_all_metadata_to_faiss()
                    textgen = TextGen(ctx)
                    textgen.text_gen_requestor(prompt)
            except Exception as e:
                await ctx.send(f"지아와 대화 중 오류 발생: {e}")
                if bot.def_channel:
                    await bot.def_channel.send(f"{ctx.channel.guild.name}/{ctx.channel.name} : 대화 오류 ({e})")
                logger.error(f"[Discord:Chat] 지아와 대화하는 과정에 오류가 발생했어요. :(\n   -> {e}")
                return
        
        @bot.command(name="지아", description="지아와 대화해요")
        async def 지아(ctx, *, prompt: str):
            try:
                async with ctx.channel.typing():
                    get_rag_instance(ctx.guild.id).sync_all_metadata_to_faiss()
                    textgen = TextGen(ctx)
                    textgen.text_gen_requestor(prompt)
            except Exception as e:
                await ctx.send(f"지아와 대화 중 오류 발생: {e}")
                if bot.def_channel:
                    await bot.def_channel.send(f"{ctx.channel.guild.name}/{ctx.channel.name} : 대화 오류 ({e})")
                logger.error(f"[Discord:Chat] 지아와 대화하는 과정에 오류가 발생했어요. :(\n   -> {e}")
                return

        @bot.command(name="jiasay", description="지아가 음성 채널에서 말해요")
        async def jiasay(ctx, *, text: str):
            if ctx.voice_client is None:
                await ctx.send("먼저 지아를 음성 채널에 초대해주세요")
                if bot.def_channel:
                    await bot.def_channel.send(f"{ctx.channel.guild.name}/{ctx.channel.name} : 음성 재생 실패 (지아가 음성 채널에 접속하지 않음)")
                return
            try:
                task_id = str(uuid.uuid4())
                threading.Thread(target=pipeline.tts_text_and_queue, args=(text, ctx.guild.id, task_id), daemon=True).start()
            except Exception as e:
                await ctx.send(f"음성 재생 중 오류 발생: {e}")
                if bot.def_channel:
                    await bot.def_channel.send(f"{ctx.channel.guild.name}/{ctx.channel.name} : 음성 재생 오류 ({e})")
                logger.error(f"[Discord:TTS] 음성을 재생하는 과정에 오류가 발생했어요. :(\n   -> {e}")
                return

        @bot.command(name="jiahear", description="지아가 음성 채널에서 듣고 텍스트로 변환해줘요")
        async def jiahear(ctx, textID: int = None):
            if not await require_command_access(ctx, "admin", "/jiahear"):
                return
            if ctx.author.voice is None:
                await ctx.send("먼저 음성 채널에 들어가주세요")
                if bot.def_channel:
                    await bot.def_channel.send(f"{ctx.channel.guild.name}/{ctx.channel.name} : 음성 채널 접속 실패 (요청자가 음성 채널 연결 상태가 아님)")
                return

            textchan = None
            if not textID:
                textchan = bot.def_channel
                if not textchan or not isinstance(textchan, discord.TextChannel):
                    await ctx.send("텍스트 채널 ID를 입력해주세요")
                    if bot.def_channel:
                        await bot.def_channel.send(f"{ctx.channel.guild.name}/{ctx.channel.name} : 음성 인식 실패 (텍스트 채널 ID 입력되지 않음)")
                    return
            else:
                textchan = bot.get_channel(textID)
                if not textchan or not isinstance(textchan, discord.TextChannel):
                    await ctx.send("텍스트 채널 ID가 올바르지 않아요")
                    if bot.def_channel:
                        await bot.def_channel.send(f"{ctx.channel.guild.name}/{ctx.channel.name} : 음성 인식 실패 (올바르지 않은 텍스트 채널 ID)")
                    return

            voice_channel = ctx.author.voice.channel
            # DAVE(E2EE) 호환 음성 수신 클라이언트로 연결
            voice_client = await voice_channel.connect(cls=voice_receive.VoiceRecvClient)
            logger.info(f"[Discord:Hear] 음성 채널에 접속할게요 -> {voice_channel.name}")
            # 음성 수신을 위한 싱크 생성 및 리스닝 시작 - textchan_id 전달
            sink = TranscriptionSink(ctx.channel.guild, voice_channel, only_hear=True, textchan_id=textchan.id)
            voice_client.listen(sink)
            if config.join_reply:
                await ctx.send("음성 채널에 접속할게요")
            if bot.def_channel:
                await bot.def_channel.send(f"{ctx.channel.guild.name}/{ctx.channel.name} : {ctx.author.voice.channel.guild.name}/{ctx.author.voice.channel.name}(으)로의 접속 요청")

        @bot.command(name="jiaplay", description="지아가 음성 채널에서 오디오 파일을 재생해요")
        async def jiaplay(ctx, file_path: str):
            if not config.allow_unsafe_jiaplay:
                await ctx.send(
                    "`/jiaplay`은 보안 문제로 차단되어 있어요. "
                    "`settings.toml`의 `[security] allow_unsafe_jiaplay` 설정에서 바꿀 수 있어요."
                )
                if bot.def_channel:
                    await bot.def_channel.send(f"{ctx.channel.guild.name}/{ctx.channel.name} : 오디오 재생 차단 (/jiaplay 비활성화)")
                return
            if not await require_command_access(ctx, "owner", "/jiaplay"):
                return
            voice_client = ctx.voice_client
            if not isinstance(voice_client, discord.VoiceClient):
                await ctx.send("먼저 지아를 음성 채널에 초대해주세요")
                if bot.def_channel:
                    await bot.def_channel.send(f"{ctx.channel.guild.name}/{ctx.channel.name} : 오디오 재생 실패 (지아가 음성 채널에 접속하지 않음)")
                return
            try:
                if not os.path.isfile(file_path):
                    await ctx.send("지정한 파일이 존재하지 않아요")
                    if bot.def_channel:
                        await bot.def_channel.send(f"{ctx.channel.guild.name}/{ctx.channel.name} : 오디오 재생 실패 (파일이 존재하지 않음)")
                    return
                ok, message = pipeline.play_sound_file(ctx.guild.id, file_path)
                await ctx.send(message)
                if not ok and bot.def_channel:
                    await bot.def_channel.send(f"{ctx.channel.guild.name}/{ctx.channel.name} : 오디오 재생 실패 ({message})")
            except Exception as e:
                await ctx.send(f"오디오 재생 중 오류 발생: {e}")
                if bot.def_channel:
                    await bot.def_channel.send(f"{ctx.channel.guild.name}/{ctx.channel.name} : 오디오 재생 오류 ({e})")
                logger.error(f"[Discord:Play] 오디오 파일을 재생하는 과정에 오류가 발생했어요. :(\n   -> {e}")
                return    

        @bot.command(name="jiamusic", description="배경 음악을 재생하거나 제어해요")
        async def jiamusic(ctx, action: str = "status", *, arg: str = ""):
            action = (action or "status").lower()
            voice_client = ctx.voice_client
            if not isinstance(voice_client, discord.VoiceClient):
                await ctx.send("먼저 지아를 음성 채널에 초대해주세요")
                if bot.def_channel:
                    await bot.def_channel.send(f"{ctx.channel.guild.name}/{ctx.channel.name} : 음악 제어 실패 (지아가 음성 채널에 접속하지 않음)")
                return

            try:
                if action in {"play", "queue", "add"}:
                    query = arg.strip().strip("\"'")
                    if not query:
                        await ctx.send("유튜브 URL, 재생목록 URL, 또는 검색어를 입력해주세요. 예) `/jiamusic play lofi hip hop radio`")
                        return
                    await ctx.send("유튜브 음악 정보를 불러오는 중이에요...")
                    try:
                        max_items = max(1, int(config.music_max_playlist_items))
                        tracks = await asyncio.to_thread(build_music_queue, query, max_items)
                    except Exception as e:
                        await ctx.send(f"유튜브 음악 정보를 불러오지 못했어요: {e}")
                        if bot.def_channel:
                            await bot.def_channel.send(f"{ctx.channel.guild.name}/{ctx.channel.name} : 유튜브 음악 정보 로딩 실패 ({e})")
                        return
                    if action == "play":
                        ok, message = pipeline.play_music(ctx.guild.id, tracks)
                    else:
                        ok, message = pipeline.queue_music(ctx.guild.id, tracks)
                    await ctx.send(message)
                    if not ok and bot.def_channel:
                        await bot.def_channel.send(f"{ctx.channel.guild.name}/{ctx.channel.name} : 음악 재생 실패 ({message})")

                elif action == "stop":
                    ok, message = pipeline.stop_music(ctx.guild.id)
                    await ctx.send(message)

                elif action == "pause":
                    ok, message = pipeline.pause_music(ctx.guild.id)
                    await ctx.send(message)

                elif action == "resume":
                    ok, message = pipeline.resume_music(ctx.guild.id)
                    await ctx.send(message)

                elif action == "skip":
                    ok, message = pipeline.skip_music(ctx.guild.id)
                    await ctx.send(message)

                elif action == "volume":
                    try:
                        raw_volume = float(arg.strip())
                    except ValueError:
                        await ctx.send("볼륨은 0.0부터 1.0 사이 숫자로 입력해주세요. 예) `/jiamusic volume 0.6`")
                        return
                    ok, message = pipeline.set_music_volume(ctx.guild.id, raw_volume)
                    await ctx.send(message)

                elif action == "status":
                    await ctx.send(pipeline.music_status(ctx.guild.id))

                else:
                    await ctx.send(
                        "**/jiamusic 사용법**\n"
                        "`/jiamusic play <유튜브 URL/재생목록/검색어>` — 기존 대기열을 바꾸고 재생해요\n"
                        "`/jiamusic queue <유튜브 URL/재생목록/검색어>` — 현재 대기열 뒤에 추가해요\n"
                        "`/jiamusic stop` — 배경 음악을 멈춰요\n"
                        "`/jiamusic skip` — 현재 곡을 건너뛰어요\n"
                        "`/jiamusic pause` — 배경 음악을 일시정지해요\n"
                        "`/jiamusic resume` — 배경 음악을 다시 재생해요\n"
                        "`/jiamusic volume <0.0~1.0>` — 음악 볼륨을 조절해요\n"
                        "`/jiamusic status` — 현재 음악 상태를 확인해요"
                    )
            except Exception as e:
                await ctx.send(f"음악 제어 중 오류 발생: {e}")
                if bot.def_channel:
                    await bot.def_channel.send(f"{ctx.channel.guild.name}/{ctx.channel.name} : 음악 제어 오류 ({e})")
                logger.error(f"[Discord:Music] 음악을 제어하는 과정에 오류가 발생했어요. :(\n   -> {e}")
                return
        
        @bot.command(name="jiastop", description="지아가 음성 채널에서 재생 중인 오디오를 멈춰요")
        async def jiastop(ctx):
            voice_client = ctx.voice_client
            if not isinstance(voice_client, discord.VoiceClient):
                await ctx.send("먼저 지아를 음성 채널에 초대해주세요")
                if bot.def_channel:
                    await bot.def_channel.send(f"{ctx.channel.guild.name}/{ctx.channel.name} : 오디오 정지 실패 (지아가 음성 채널에 접속하지 않음)")
                return
            try:
                # 아직 응답 생성이 시작되지 않은 대기 발화를 비워 새 응답이 시작되지 않게 함
                pipeline.clear_pending_utterances(ctx.guild.id)
                # 진행 중인 TTS 생성에 취소 신호를 먼저 보내 뒷문장이 새로 큐에 들어오지 않게 함
                cancelled = pipeline.cancel_tts_tasks(ctx.guild.id)
                # foreground(TTS/효과음)만 비웁니다. 배경 음악은 /jiamusic stop으로 따로 제어합니다.
                stopped = pipeline.stop_foreground_audio(ctx.guild.id)
                if stopped:
                    await ctx.send("재생을 멈췄어요")
                    if bot.def_channel:
                        await bot.def_channel.send(f"{ctx.channel.guild.name}/{ctx.channel.name} : 오디오 재생 정지 요청")
                elif cancelled:
                    # 재생 시작 전이지만 음성 생성이 진행 중이던 경우
                    await ctx.send("준비 중이던 음성 생성을 취소했어요")
                    if bot.def_channel:
                        await bot.def_channel.send(f"{ctx.channel.guild.name}/{ctx.channel.name} : 음성 생성 취소 요청")
                elif bot.audio_stream_manager and bot.audio_stream_manager.has_music(ctx.guild.id):
                    await ctx.send("지아가 말하는 중은 아니고, 배경 음악만 재생 중이에요. 음악을 멈추려면 `/jiamusic stop`을 사용해주세요.")
                else:
                    await ctx.send("지금은 재생 중이 아니에요")
                    if bot.def_channel:
                        await bot.def_channel.send(f"{ctx.channel.guild.name}/{ctx.channel.name} : 오디오 재생 정지 실패 (현재 재생 중이 아님)")
            except Exception as e:
                await ctx.send(f"오디오 정지 중 오류 발생: {e}")
                if bot.def_channel:
                    await bot.def_channel.send(f"{ctx.channel.guild.name}/{ctx.channel.name} : 오디오 정지 오류 ({e})")
                logger.error(f"[Discord:Stop] 오디오 재생을 멈추는 과정에 오류가 발생했어요. :(\n   -> {e}")
                return
            
        @bot.command(name="jiajoinnoagent", description="지아를 음성 채널에 초대해요 (음성 인식 없이)")
        async def jiajoinnoagent(ctx):
            if ctx.author.voice is None:
                await ctx.send("먼저 음성 채널에 들어가주세요")
                if bot.def_channel:
                    await bot.def_channel.send(f"{ctx.channel.guild.name}/{ctx.channel.name} : 음성 채널 접속 실패 (요청자가 음성 채널 연결 상태가 아님)")
                return
            voice_channel = ctx.author.voice.channel
            await voice_channel.connect()
            logger.info(f"[Discord:JoinNoAgent] 음성 채널에 접속할게요 -> {voice_channel.name}")
            if config.join_reply:
                await ctx.send("음성 채널에 접속할게요")
            if bot.def_channel:
                await bot.def_channel.send(f"{ctx.channel.guild.name}/{ctx.channel.name} : {ctx.author.voice.channel.guild.name}/{ctx.author.voice.channel.name}(으)로의 접속 요청 (음성 인식 없이)")

        @bot.command(name="jiamemory", description="지아의 기억을 관리해요")
        async def jiamemory(ctx, action: str = "help", *, arg: str = ""):
            try:
                rag = get_rag_instance(ctx.guild.id)
                action = action.lower()
                arg = arg.strip()
                if action in {"list", "search", "delete", "profile"}:
                    if not await require_command_access(ctx, "admin", f"/jiamemory {action}"):
                        return

                def format_rows(rows) -> str:
                    lines = []
                    for mem_id, username, summary, importance, timestamp in rows:
                        summary_short = summary if len(summary) <= 120 else summary[:120] + "…"
                        lines.append(f"`{mem_id[:8]}` [{timestamp[:16]}] ({username}, 중요도 {importance:.2f})\n　{summary_short}")
                    return "\n".join(lines)

                if action == "list":
                    page = int(arg) if arg.isdigit() else 1
                    total = rag.count_memories()
                    rows = rag.list_memories(page=page, page_size=5)
                    if not rows:
                        await ctx.send("이 페이지에는 기억이 없어요." if total else "아직 이 서버에 저장된 기억이 없어요.")
                        return
                    last_page = (total + 4) // 5
                    await ctx.send(f"**이 서버의 기억** (총 {total}개, {page}/{last_page} 페이지)\n{format_rows(rows)}\n\n삭제하려면 `/jiamemory delete <ID>`를 사용해주세요.")

                elif action == "search":
                    if not arg:
                        await ctx.send("검색어를 함께 입력해주세요. 예) `/jiamemory search 생일`")
                        return
                    rows = await asyncio.to_thread(rag.search_memories, arg)
                    if not rows:
                        await ctx.send("관련된 기억을 찾지 못했어요.")
                        return
                    await ctx.send(f"**'{arg}' 검색 결과**\n{format_rows(rows)}")

                elif action == "delete":
                    if not arg:
                        await ctx.send("삭제할 기억의 ID를 함께 입력해주세요. ID는 `/jiamemory list`에서 확인할 수 있어요.")
                        return
                    # ID는 uuid 형식의 일부만 허용 (LIKE 와일드카드 등으로 전체 삭제되는 것 방지)
                    if len(arg) < 4 or not arg.replace("-", "").isalnum():
                        await ctx.send("ID는 4자 이상으로, `/jiamemory list`에 표시된 형태 그대로 입력해주세요.")
                        return
                    deleted = rag.delete_memory(arg)
                    if deleted:
                        # 삭제된 기억이 검색되지 않도록 인덱스를 다시 생성
                        await asyncio.to_thread(rag.sync_all_metadata_to_faiss)
                        await asyncio.to_thread(rag.save_all)
                        await ctx.send(f"기억 {deleted}개를 삭제했어요.")
                        if bot.def_channel:
                            await bot.def_channel.send(f"{ctx.channel.guild.name}/{ctx.channel.name} : 기억 삭제 요청 ({deleted}개)")
                    else:
                        await ctx.send("해당 ID로 시작하는 기억을 찾지 못했어요.")

                elif action == "profile":
                    name = arg or ctx.author.name
                    facts = rag.get_profile_facts(name)
                    if not facts:
                        await ctx.send(f"{name}에 대해 따로 기억하고 있는 정보가 없어요.")
                        return
                    listed = "\n".join(f"- {f}" for f in facts)
                    await ctx.send(f"**{name}에 대해 기억하고 있는 것**\n{listed}")

                elif action == "optout":
                    if rag.is_opted_out(ctx.author.name):
                        await ctx.send("이미 기억 기능 사용을 거부한 상태예요.")
                        return
                    await asyncio.to_thread(rag.set_optout, ctx.author.name, True)
                    await asyncio.to_thread(rag.sync_all_metadata_to_faiss)
                    await ctx.send(f"{ctx.author.name}님의 대화와 정보를 이제 기억하지 않을게요. 기존 프로필과 단독 대화 기억도 삭제했어요.\n다시 켜려면 `/jiamemory optin`을 입력해주세요.")
                    if bot.def_channel:
                        await bot.def_channel.send(f"{ctx.channel.guild.name}/{ctx.channel.name} : {ctx.author.name} 기억 거부(opt-out) 설정")

                elif action == "optin":
                    if not rag.is_opted_out(ctx.author.name):
                        await ctx.send("지금도 기억 기능을 사용 중이에요.")
                        return
                    await asyncio.to_thread(rag.set_optout, ctx.author.name, False)
                    await ctx.send(f"이제부터 {ctx.author.name}님과의 대화를 다시 기억할게요.")
                    if bot.def_channel:
                        await bot.def_channel.send(f"{ctx.channel.guild.name}/{ctx.channel.name} : {ctx.author.name} 기억 거부 해제(opt-in)")

                elif action == "status":
                    opted_out = rag.is_opted_out(ctx.author.name)
                    facts = rag.get_profile_facts(ctx.author.name)
                    total = rag.count_memories()
                    state = "기억 안 함 (opt-out)" if opted_out else "기억 중"
                    if facts:
                        profile_text = "\n".join(f"- {fact}" for fact in facts)
                    else:
                        profile_text = "- 따로 저장된 프로필 사실이 없어요."
                    await ctx.send(
                        f"**{ctx.author.name}님의 기억 설정**\n"
                        f"- 상태: {state}\n"
                        f"- 프로필에 저장된 사실: {len(facts)}개\n"
                        f"- 이 서버의 전체 기억: {total}개\n\n"
                        f"**내 프로필**\n{profile_text}"
                    )

                else:
                    await ctx.send(
                        "**/jiamemory 사용법**\n"
                        "`/jiamemory list [페이지]` — 이 서버의 기억을 최신순으로 보여줘요\n"
                        "`/jiamemory search <검색어>` — 기억을 검색해요\n"
                        "`/jiamemory delete <ID>` — 해당 ID로 시작하는 기억을 삭제해요\n"
                        "`/jiamemory profile [이름]` — 사용자에 대해 기억하는 정보를 보여줘요 (관리자 전용)\n"
                        "`/jiamemory optout` — 내 대화와 정보를 기억하지 않게 해요 (기존 프로필/단독 기억도 삭제)\n"
                        "`/jiamemory optin` — 기억 기능을 다시 켜요\n"
                        "`/jiamemory status` — 내 기억 설정 상태와 프로필을 확인해요"
                    )
            except Exception as e:
                await ctx.send(f"기억 관리 중 오류 발생: {e}")
                if bot.def_channel:
                    await bot.def_channel.send(f"{ctx.channel.guild.name}/{ctx.channel.name} : 기억 관리 오류 ({e})")
                logger.error(f"[Discord:Memory] 기억을 관리하는 과정에 오류가 발생했어요. :(\n   -> {e}")

        @bot.command(name="jiatalk", description="지아와 대화를 시작해요. (대화에 /jia를 붙이지 않아도 괜찮아요.)")
        async def jiatalk(ctx):
            if not await require_command_access(ctx, "admin", "/jiatalk"):
                return
            # 길드에 대한 채널 set이 없으면 생성
            if ctx.guild.id not in bot.autotalk_channels:
                bot.autotalk_channels[ctx.guild.id] = set()

            if ctx.channel.id in bot.autotalk_channels[ctx.guild.id]:
                await ctx.send("이미 이 채널에서 대화 중이에요.")
                return
            
            bot.autotalk_channels[ctx.guild.id].add(ctx.channel.id)
            await ctx.send(f"이제부터 이 채널에서 `/jia` 없이 대화할 수 있어요. 이 채널에서 대화를 종료하려면 `/jiastoptalk`를 입력해주세요.")

        @bot.command(name="jiastoptalk", description="지아의 대화를 종료해요")
        async def jiastoptalk(ctx):
            if not await require_command_access(ctx, "admin", "/jiastoptalk"):
                return
            if ctx.guild.id not in bot.autotalk_channels or ctx.channel.id not in bot.autotalk_channels[ctx.guild.id]:
                await ctx.send("이 채널에서는 대화 기능이 활성화되어 있지 않아요.")
                return
            
            bot.autotalk_channels[ctx.guild.id].remove(ctx.channel.id)
            # 다른 자동 대화 채널이나 음성 연결이 없을 때만 언로드
            if not llm_still_in_use():
                await asyncio.to_thread(unload_ollama_model, config.llmModel)
            await ctx.send("이 채널의 대화 기능을 종료했어요.")

        @bot.event
        async def on_message(message):
            if message.author == bot.user:
                return
            
            # 커맨드를 처리하도록 명시적으로 호출
            await bot.process_commands(message)
            ctx = await bot.get_context(message)
            
            if ctx.command is not None:
                return
            
            # 해당 길드의 자동 대화 채널 목록에 현재 채널이 있는지 확인
            if message.guild.id in bot.autotalk_channels and message.channel.id in bot.autotalk_channels[message.guild.id] and not message.author.bot:
                await jia(ctx, prompt=message.content)

        # settings.toml 파일 변경을 감시해 재시작 없이 설정을 자동 반영 (모델 관련 설정은 해당 모델만 재로딩)
        config.start_auto_reload(on_change=on_config_changed)
        bot.run(config.bot_token)
    except Exception as e:
        logger.error(f"[Discord] discord 봇에서 에러 발생! 예외 처리되지 않은 문제에요.\n   -> {e}")
