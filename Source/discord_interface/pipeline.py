from fastapi import FastAPI
from pydantic import BaseModel
from queue import Queue
import threading
import time
import uuid
from discord_interface.vad import get_speech_timestamps_from_array
from discord_interface.tts_output import generate_tts
from discord_interface.transformers_whisper import transcribe_sync
from LLM.langchain_llm import generate_call_response, generate_response
import logging
import requests

app = FastAPI()

task_queue = Queue()
text_task_queue = Queue()

class Task(BaseModel):
    user: str
    guild: int
    audio_data: list

class TextTask(BaseModel):
    user: str
    guild: int
    text: str

def worker():
    while True:
        task, user, guild, audio_data = task_queue.get()
        if task is None:
            time.sleep(0.5)  # 대기 시간
            continue
        speech_timestamps = get_speech_timestamps_from_array(audio_data)
        if speech_timestamps:
            # 받은 음성 패킷에 대한 VAD 결과가 있다면
            last_segment = speech_timestamps[-1]
            end_idx = last_segment['end']
            logging.info(f"[{user}] 음성을 텍스트로 변환하고 있어요.")
            utterance = audio_data[:end_idx]
            text = transcribe_sync(utterance)
            logging.info(f"[{user}] 텍스트 변환 결과가 나왔어요. -> {text}")
            if text:
                reply = generate_call_response(user, guild, text)
                if reply:
                    file_name = generate_tts(reply)
                    try:
                        requests.post("http://localhost:8001/play-audio", json={
                            "guild_id": guild,
                            "file_path": file_name
                        })
                    except requests.exceptions.RequestException as e:
                        logging.error(f"음성 재생을 봇에 요청하는 데 실패했어요: {e}")
        # Mark the task as done
        task_queue.task_done()

def text_worker():
    while True:
        task, user, guild, text = text_task_queue.get()
        if task is None:
            time.sleep(0.5)
            continue
        result_text = generate_response(user, guild, text)
        requests.post(f"http://localhost:8001/text-send", json={"task_id": task, "result": result_text})
        text_task_queue.task_done()


threading.Thread(target=text_worker, daemon=True).start()
threading.Thread(target=worker, daemon=True).start()

@app.post("/run-process")
def submit_task(request: Task):
    """작업을 큐에 등록하고 작업 ID 반환"""
    task_id = str(uuid.uuid4())
    task_queue.put((task_id, request.user, request.guild, request.audio_data))
    return {"task_id": task_id, "status": "queued"}

@app.post("/run-text-process")
def run_text_process(request: TextTask):
    """텍스트 작업을 큐에 등록하고 작업 ID 반환"""
    task_id = str(uuid.uuid4())
    text_task_queue.put((task_id, request.user, request.guild, request.text))
    return {"task_id": task_id, "status": "queued"}
