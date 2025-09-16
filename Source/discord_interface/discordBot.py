import discord
from discord.ext import commands
from discord.ext import voice_recv
import logging
import numpy as np
import torch
import torchaudio
from discord_interface.transformers_whisper import reload_whisper_model
import asyncio
from config.config_manager import config
import os
from datetime import datetime
from discord_interface.tts_output import reload_tts_model
from memory.RAG import RAG
import requests
from queue import Queue
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

class TextResult(BaseModel):
    task_id: str
    result: str

class AudioTask(BaseModel):
    guild_id: int
    file_path: str

for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

def bot():
    try:
        intents = discord.Intents.default()
        intents.message_content = True
        intents.voice_states = True
        intents.guilds = True
        intents.guild_messages = True

        bot = commands.Bot(command_prefix="/", intents=intents)
        bot.textdict = {}
        bot.rag = {}

        class TextGen():
            def __init__(self, ctx):
                self.ctx = ctx

            def text_gen_requestor(self, prompt):
                resp = requests.post(f"http://localhost:8000/run-text-process", json={
                    "user": self.ctx.author.name,
                    "guild": self.ctx.channel.guild.id,
                    "text": prompt
                })
                task_info = resp.json()
                task_id = task_info.get("task_id")
                return task_id

        class TranscriptionSink(voice_recv.AudioSink):
            def __init__(self, guild, voice_channel):
                super().__init__()
                self.buffers = {}  # 유저별 오디오 버퍼 (16 kHz)
                self.prev_tails = {} # 유저별 이전 음성 데이터
                self._current_task = {}
                self.timeout_sec = 0.3
                self.guild = guild
                self.loop = bot.loop
                self.resampler = torchaudio.transforms.Resample(orig_freq=48000, new_freq=16000, resampling_method="sinc_interpolation", lowpass_filter_width=128)
                self.voice_channel = voice_channel
                self.task_queue = Queue()

            def wants_opus(self) -> bool:
                # OPUS 패킷이 아닌 PCM 오디오(float)를 받아야 함
                return False

            def write(self, user: discord.User, data: voice_recv.VoiceData):
                if not user:
                    return
                if not user.bot:
                    def runner():
                        if user.id in self._current_task:
                            if self._current_task[user.id] != {} and not self._current_task[user.id].done():
                                self._current_task[user.id].cancel()
                        # 새 작업 등록 및 실행
                        self._current_task[user.id] = self.loop.create_task(self.packet_timeout(user))

                    # 루프 실행
                    self.loop.call_soon_threadsafe(runner)
                    
                    # data.pcm은 48000Hz의 16비트 PCM 형식
                    audio_data = np.frombuffer(data.pcm, dtype=np.int16)
                    # 2채널 음성 데이터가 1차원 배열로 전달되기 때문에 reshape 필요
                    audio_data = audio_data.reshape(-1, 2)

                    # 스테레오 -> 모노 전환
                    mono_data = audio_data.mean(axis=1)

                    # int16 -> float32 전환
                    tensor = torch.from_numpy(mono_data.astype(np.float32) / 32768.0).unsqueeze(0)

                    # 이전 텐서가 있으면 연결, 없으면 새로운 텐서 생성
                    prev_tail = self.prev_tails.get(user.id)
                    if prev_tail is not None:
                        tensor = torch.cat([prev_tail, tensor], dim=1)

                    # 16000Hz로 리샘플링
                    resampled = self.resampler(tensor)

                    # 길이가 256보다 작으면 패딩
                    ratio = resampled.shape[1] / tensor.shape[1]
                    cut = int(256 * ratio)
                    resampled = resampled[:, cut:]

                    # 마지막 256개 샘플을 이전 텐서로 업데이트
                    self.prev_tails[user.id] = tensor[:, -256:]

                    # 텐서를 numpy로 변환
                    resampled = resampled.squeeze(0).numpy()

                    # 버퍼에 누적
                    prev = self.buffers.get(user.id)
                    if prev is None:
                        buf = resampled
                    else:
                        buf = np.concatenate((prev, resampled))
                    self.buffers[user.id] = buf

            async def packet_timeout(self, user: discord.User):
                await asyncio.sleep(self.timeout_sec)
                self.send_vad_and_whisper(user)

            def send_vad_and_whisper(self, user: discord.User):
                try:
                    requests.post(f"http://localhost:8000/run-process", json={
                        "user": user.name,
                        "guild": self.guild.id,
                        "audio_data": self.buffers[user.id].tolist()  # numpy array를 리스트로 변환
                    })
                    self.buffers[user.id] = np.array([], dtype=np.float32)
                except Exception as e:
                    logger.error(f"[{user.name}] VAD(대화 감지) 또는 Whisper(텍스트화)에서 오류가 발생했어요. :( \n   -> {e}")

            def cleanup(self):
                # 음성 수신이 중단되면 요청; 버퍼 클리어
                self.buffers.clear()
                self.prev_tails.clear()

        async def get_channel_sure():
            await bot.wait_until_ready()

            print("로그 채널 연결을 시도하고 있어요")
            channel = bot.get_channel(config.debug_text_channel)
            if channel:
                return channel
            print("로그 채널 연결에 실패했어요")

        @app.post("/play-audio")
        async def play_audio_file(request: AudioTask):
            guild_id = request.guild_id
            file_path = request.file_path

            # 허용된 디렉토리 외부의 파일 재생을 방지
            allowed_path = os.path.abspath("Source/output_temp")
            requested_path = os.path.abspath(file_path)
            if not requested_path.startswith(allowed_path):
                logger.error(f"Path traversal 시도 감지. 재생 요청 차단: {file_path}")
                return {"status": "error", "message": "Access denied."}

            guild = bot.get_guild(guild_id)
            if not guild:
                logger.error(f"길드(서버)를 찾을 수 없어요: {guild_id}")
                # 재생되지 못한 오디오 파일이 남지 않도록 삭제
                if os.path.exists(file_path):
                    os.remove(file_path)
                return {"status": "error", "message": "Guild not found"}
            voice_client = guild.voice_client

            if isinstance(voice_client, discord.VoiceClient) and voice_client.is_connected():
                if not os.path.isfile(file_path):
                    logger.error(f"오디오 파일을 찾을 수 없어요: {file_path}")
                    return {"status": "error", "message": "File not found"}
                
                voice_client.play(discord.FFmpegPCMAudio(file_path), after=lambda e: os.remove(file_path) if e is None else logger.error(f"음성 재생 후 파일 삭제 중 오류 발생: {e}"))
                return {"status": "playing"}
            else:
                logger.warning(f"음성 클라이언트를 찾을 수 없거나 길드 {guild_id}에 연결되지 않았어요.")
                # 재생되지 못한 오디오 파일이 남지 않도록 삭제
                if os.path.exists(file_path):
                    os.remove(file_path)
                return {"status": "error", "message": "Voice client not connected"}

        @app.post("/text-send")
        async def task_processor(requested: TextResult):
            task_id = requested.task_id
            result = requested.result
            ctx = bot.textdict.get(task_id)
            if ctx is None:
                fut = asyncio.run_coroutine_threadsafe(
                    bot.def_channel.send("ctx를 찾을 수 없음"), bot.loop
                )
                fut.result()
                return {"status": "error", "message": f"task_id {task_id} not found"}
            try:
                # Discord 봇의 이벤트 루프에서 실행
                fut = asyncio.run_coroutine_threadsafe(ctx.send(result), bot.loop)
                fut.result()  # 예외 발생 시 잡기 위함
                return {"status": "received"}
            except Exception as e:
                fut = asyncio.run_coroutine_threadsafe(
                    ctx.send(f"결과 전송 중 오류 발생: {e}"), bot.loop
                )
                fut.result()
                return {"status": "error", "message": str(e)}

        # ==== 커맨드 흐름 ====
        @bot.command(name="jiajoin", description="지아를 음성 채널에 초대해요")
        async def jiajoin(ctx):
            if ctx.author.voice is None:
                await ctx.send("먼저 음성 채널에 들어가주세요")
                await bot.def_channel.send(f"{ctx.channel.guild.name}/{ctx.channel.name} : 음성 채널 접속 실패 (요청자가 음성 채널 연결 상태가 아님)")
                return
            bot.rag[ctx.channel.guild.id] = RAG(ctx.channel.guild.id)
            bot.rag[ctx.channel.guild.id].sync_all_metadata_to_faiss()
            voice_channel = ctx.author.voice.channel
            voice_client = await voice_channel.connect(cls=voice_recv.VoiceRecvClient)
            logger.info(f"[Discord:Join] 음성 채널에 접속할게요 -> {voice_channel.name}")
            sink = TranscriptionSink(ctx.channel.guild, voice_client)
            voice_client.listen(sink)
            if config.join_reply:
                await ctx.send("음성 채널에 접속할게요")
            await bot.def_channel.send(f"{ctx.channel.guild.name}/{ctx.channel.name} : {ctx.author.voice.channel.guild.name}/{ctx.author.voice.channel.name}(으)로의 접속 요청")

        @bot.command(name="jialeave", description="지아를 음성 채널에서 내보네요")
        async def jialeave(ctx):
            if ctx.voice_client:
                bot.rag[ctx.channel.guild.id].save_index()
                del bot.rag[ctx.channel.guild.id]
                await ctx.voice_client.disconnect()
                if config.leave_reply:
                    await ctx.send("음성 채널에서 나갈게요")
                await bot.def_channel.send(f"{ctx.channel.guild.name}/{ctx.channel.name} : 음성 연결 해제 요청")
            else:
                await ctx.send("지금은 음성 채널에 접속한 상태가 아니에요")
                await bot.def_channel.send(f"{ctx.channel.guild.name}/{ctx.channel.name} : 음성 연결 해제 실패 (현재 연결 상태가 아님)")

        @bot.event
        async def on_ready():
            print('봇 온라인!')
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
                new_config = type(global_config).load_setting()
                import config.config_manager
                config.config_manager.config = new_config
                reload_tts_model()  # TTS 모델 재로딩
                reload_whisper_model()  # Whisper 모델 재로딩
                rag = RAG(ctx.guild.id)
                rag.sync_all_metadata_to_faiss()  # FAISS 인덱스 동기화
                await ctx.send("설정이 성공적으로 재로딩되었어요")
                await bot.def_channel.send(f"{ctx.channel.guild.name}/{ctx.channel.name} : 설정 재로딩 요청")
                logger.info("[Discord:Reload] 설정을 다시 불러왔어요.")
            except Exception as e:
                await ctx.send(f"설정 재로딩 중 오류 발생: {e}")
                await bot.def_channel.send(f"{ctx.channel.guild.name}/{ctx.channel.name} : 설정 재로딩 실패 ({e})")
                logger.error(f"[Discord:Reload] 설정을 다시 불러오는 과정에 오류가 발생했어요. :(\n   -> {e}")

        @bot.command(name="jiasavesetting", description="지아의 설정을 저장해요")
        async def jiasavesetting(ctx):
            try:
                from config.config_manager import config as global_config
                global_config.save_setting()
                await ctx.send("설정이 성공적으로 저장되었어요")
                await bot.def_channel.send(f"{ctx.channel.guild.name}/{ctx.channel.name} : 설정 저장 요청")
                logger.info("[Discord:Save_Setting] 설정을 저장했어요.")
            except Exception as e:
                await ctx.send(f"설정 저장 중 오류 발생: {e}")
                await bot.def_channel.send(f"{ctx.channel.guild.name}/{ctx.channel.name} : 설정 저장 실패 ({e})")
                logger.error(f"[Discord:Save_Setting] 설정을 저장하는 과정에 오류가 발생했어요. :(\n   -> {e}")

        @bot.command(name="jiaaddslang", description="지아의 속어 사전을 업데이트해요")
        async def jiaaddslang(ctx, word: str, meaning: str, examples: str):
            try:
                rag = RAG(ctx.channel.guild.id)
                rag.save_slang_metadata(word, meaning, examples, ctx.author.name, datetime.now().isoformat())
                await ctx.send(f"속어 '{word}'이(가) 추가되었어요")
                await bot.def_channel.send(f"{ctx.channel.guild.name}/{ctx.channel.name} : 속어 '{word}' 추가 요청")
            except Exception as e:
                await ctx.send(f"속어 추가 중 오류 발생: {e}")
                await bot.def_channel.send(f"{ctx.channel.guild.name}/{ctx.channel.name} : 속어 추가 실패 ({e})")
                logger.error(f"[Discord:Add_Slang] 단어 사전을 추가하는 과정에 오류가 발생했어요. :(\n   -> {e}")

        @bot.command(name="jia", description="지아와 대화해요")
        async def jia(ctx, *, prompt: str):
            try:
                async with ctx.channel.typing():
                    rag = RAG(ctx.channel.guild.id)
                    rag.sync_all_metadata_to_faiss()
                    textgen = TextGen(ctx)
                    task_id = textgen.text_gen_requestor(prompt)
                    bot.textdict[task_id] = ctx
            except Exception as e:
                await ctx.send(f"지아와 대화 중 오류 발생: {e}")
                await bot.def_channel.send(f"{ctx.channel.guild.name}/{ctx.channel.name} : 대화 오류 ({e})")
                logger.error(f"[Discord:Chat] 지아와 대화하는 과정에 오류가 발생했어요. :(\n   -> {e}")
                return
        
        @bot.command(name="지아", description="지아와 대화해요")
        async def 지아(ctx, *, prompt: str):
            try:
                async with ctx.channel.typing():
                    textgen = TextGen(ctx)
                    task_id = textgen.text_gen_requestor(prompt)
                    bot.textdict[task_id] = ctx
            except Exception as e:
                await ctx.send(f"지아와 대화 중 오류 발생: {e}")
                await bot.def_channel.send(f"{ctx.channel.guild.name}/{ctx.channel.name} : 대화 오류 ({e})")
                logger.error(f"[Discord:Chat] 지아와 대화하는 과정에 오류가 발생했어요. :(\n   -> {e}")
                return

        bot.run(config.bot_token)
    except Exception as e:
        logger.error(f"[Discord] discord 봇에서 에러 발생! 예외 처리되지 않은 문제에요.\n   -> {e}")