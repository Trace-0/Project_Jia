import discord
from discord.ext import commands
from discord_interface import voice_receive
import logging
import numpy as np
from discord_interface.faster_whisper_output import reload_whisper_model
import asyncio
import threading
from config.config_manager import config
import os
from memory.RAG import get_rag_instance
from LLM.LLM_model_control import unload_ollama_model, load_ollama_model
from collections import deque
import uuid
from discord_interface import pipeline

for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger(__name__)

logger.setLevel(logging.INFO)

bot_client = None

class AudioStreamManager:
    """길드별 오디오 스트림을 관리하는 클래스"""
    def __init__(self, bot):
        self.streams: dict[int, dict[str, deque]] = {}
        self.playing: dict[int, bool] = {}
        self.bot = bot

    def add_to_queue(self, guild_id: int, task_id: str, audio_source):
        if guild_id not in self.streams:
            self.streams[guild_id] = {}
            self.playing[guild_id] = False
        if task_id not in self.streams[guild_id]:
            self.streams[guild_id][task_id] = deque()
        
        self.streams[guild_id][task_id].append(audio_source)
        logger.info(f"오디오 스트림 큐에 추가: 서버({guild_id})")
        
        if not self.playing.get(guild_id):
            self.bot.loop.create_task(self.play_next(guild_id))

    async def play_next(self, guild_id: int):
        if self.playing.get(guild_id):
            return

        self.playing[guild_id] = True
        guild = self.bot.get_guild(guild_id)
        voice_client = guild.voice_client if guild else None

        while any(self.streams.get(guild_id, {}).values()):
            task_id, queue = next((tid, q) for tid, q in self.streams.get(guild_id, {}).items() if q)
            
            if not queue:
                continue

            audio_source = queue.popleft()
            
            if isinstance(voice_client, discord.VoiceClient) and voice_client.is_connected():
                if isinstance(audio_source, str):
                    source = discord.FFmpegPCMAudio(audio_source)
                else:
                    source = discord.FFmpegPCMAudio(audio_source, pipe=True)
                voice_client.play(source, after=lambda e: self.bot.loop.call_soon_threadsafe(self.after_play, guild_id, audio_source, e))
                return # after_play 콜백이 play_next를 다시 호출

        self.playing[guild_id] = False

    def after_play(self, guild_id: int, audio_source, error):
        if error:
            logger.error(f"오디오 파일 재생 후 오류: {error}")
        if isinstance(audio_source, str) and os.path.exists(audio_source):
            os.remove(audio_source)
        elif hasattr(audio_source, 'close'):
            audio_source.close()
        self.playing[guild_id] = False
        self.bot.loop.create_task(self.play_next(guild_id))

class JiaBot(commands.Bot):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.textdict: dict = {} # 텍스트 작업의 task_id와 context를 매핑하는 딕셔너리
        self.autotalk_channels: dict[int, set[int]] = {} # 길드 ID와 자동 대화 채널 ID 집합을 매핑하는 딕셔너리
        self.def_channel: discord.TextChannel | None = None # 기본 로그 채널
        self.audio_stream_manager: AudioStreamManager | None = None

    async def setup_hook(self):
        self.audio_stream_manager = AudioStreamManager(self)

    def send_text_result_sync(self, task_id: str, result: str, user: str):
        """스레드에서 호출되어 봇의 루프에 메시지 전송 태스크를 생성하는 함수"""
        ctx = self.textdict.get(task_id)
        if ctx:
            # 비동기 함수인 ctx.send를 메인 루프에서 실행되도록 예약
            self.loop.create_task(ctx.send(f"[{user}]: {result}"))

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
        class TextGen():
            def __init__(self, ctx):
                self.ctx = ctx

            def text_gen_requestor(self, prompt):
                # 파이프라인 작업 실행
                task_id = str(uuid.uuid4())
                pipeline.run_text_task(task_id, self.ctx.author.name, self.ctx.channel.guild.id, prompt)
                return task_id

        # 음성 데이터를 수신하고 처리하는 클래스
        class TranscriptionSink(voice_receive.AudioSink):
            def __init__(self, guild: discord.Guild, voice_channel: discord.VoiceChannel, only_hear: bool = False, textchan_id: int = None):
                super().__init__()
                self.buffers: dict[int, list[np.ndarray]] = {} # 사용자별 오디오 청크를 저장하는 버퍼
                self._current_task: dict[int, asyncio.Task] = {} # 사용자별 타임아웃 작업을 저장하는 딕셔너리
                self.timeout_sec = 0.1 # 음성 입력 후 처리를 시작하기까지의 대기 시간
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

                # 이전 타임아웃 작업을 취소하여 타이머를 리셋
                def runner():
                    task = self._current_task.get(user.id)
                    if task and not task.done():
                        task.cancel()
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

            async def packet_timeout(self, user: discord.User):
                try:
                    await asyncio.sleep(self.timeout_sec)
                    await self.send_vad_and_whisper(user)
                except asyncio.CancelledError:
                    pass

            async def send_vad_and_whisper(self, user: discord.User):
                # 사용자의 버퍼를 가져옴
                user_buffer = self.buffers.pop(user.id, [])
                if not user_buffer:
                    return

                try:
                    full_audio = np.concatenate(user_buffer)
                    # 파이프라인 작업 실행
                    task_id = str(uuid.uuid4())
                    pipeline.run_audio_task(task_id, user.name, self.guild.id, full_audio, self.only_hear, self.textchan_id)
                except Exception as e:
                    logger.error(f"[{user.name}] VAD(대화 감지) 또는 Whisper(텍스트화)에서 오류가 발생했어요. :( \n   -> {e}")

            def cleanup(self):
                # 대기 중인 모든 타임아웃 작업을 취소하고 버퍼를 비움
                for task in self._current_task.values():
                    if task and not task.done():
                        task.cancel()
                self._current_task.clear()
                self.buffers.clear()
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
            load_ollama_model(config.llmModel)
            voice_channel = ctx.author.voice.channel
            # DAVE(E2EE) 호환 음성 수신 클라이언트로 연결
            voice_client = await voice_channel.connect(cls=voice_receive.VoiceRecvClient)
            logger.info(f"[Discord:Join] 음성 채널에 접속할게요 -> {voice_channel.name}")
            # 음성 수신을 위한 싱크 생성 및 리스닝 시작
            sink = TranscriptionSink(ctx.channel.guild, voice_channel)
            voice_client.listen(sink)
            if config.join_reply:
                await ctx.send("음성 채널에 접속할게요")
            if bot.def_channel:
                await bot.def_channel.send(f"{ctx.channel.guild.name}/{ctx.channel.name} : {ctx.author.voice.channel.guild.name}/{ctx.author.voice.channel.name}(으)로의 접속 요청")

        @bot.command(name="jialeave", description="지아를 음성 채널에서 내보네요")
        async def jialeave(ctx):
            if ctx.voice_client:
                # RAG 인덱스 저장
                get_rag_instance(ctx.guild.id).save_all()
                # 오디오 스트림 정리
                if bot.audio_stream_manager and ctx.guild.id in bot.audio_stream_manager.streams:
                    del bot.audio_stream_manager.streams[ctx.guild.id]
                    bot.audio_stream_manager.playing[ctx.guild.id] = False
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
            try:
                from config.config_manager import config as global_config
                global_config.reload()
                reload_whisper_model()  # Whisper 모델 재로딩
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

        @bot.command(name="jia", description="지아와 대화해요")
        async def jia(ctx, *, prompt: str):
            try:
                async with ctx.channel.typing():
                    get_rag_instance(ctx.guild.id).sync_all_metadata_to_faiss()
                    textgen = TextGen(ctx)
                    task_id = textgen.text_gen_requestor(prompt)
                    # task_id와 context를 매핑하여 나중에 응답을 보낼 채널을 찾음
                    bot.textdict[task_id] = ctx
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
                    task_id = textgen.text_gen_requestor(prompt)
                    bot.textdict[task_id] = ctx
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
                voice_client.play(discord.FFmpegPCMAudio(file_path), after=lambda e: logger.error(f"오디오 재생 중 오류 발생: {e}") if e else None)
            except Exception as e:
                await ctx.send(f"오디오 재생 중 오류 발생: {e}")
                if bot.def_channel:
                    await bot.def_channel.send(f"{ctx.channel.guild.name}/{ctx.channel.name} : 오디오 재생 오류 ({e})")
                logger.error(f"[Discord:Play] 오디오 파일을 재생하는 과정에 오류가 발생했어요. :(\n   -> {e}")
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
                if voice_client.is_playing():
                    voice_client.stop()
                    await ctx.send("재생을 멈췄어요")
                    if bot.def_channel:
                        await bot.def_channel.send(f"{ctx.channel.guild.name}/{ctx.channel.name} : 오디오 재생 정지 요청")
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

        @bot.command(name="jiatalk", description="지아와 대화를 시작해요. (대화에 /jia를 붙이지 않아도 괜찮아요.)")
        async def jiatalk(ctx):
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
            if ctx.guild.id not in bot.autotalk_channels or ctx.channel.id not in bot.autotalk_channels[ctx.guild.id]:
                await ctx.send("이 채널에서는 대화 기능이 활성화되어 있지 않아요.")
                return
            
            bot.autotalk_channels[ctx.guild.id].remove(ctx.channel.id)
            unload_ollama_model(config.llmModel)
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

        bot.run(config.bot_token)
    except Exception as e:
        logger.error(f"[Discord] discord 봇에서 에러 발생! 예외 처리되지 않은 문제에요.\n   -> {e}")