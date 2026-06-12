from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field
from config.config_manager import config
import requests
import logging
import random
import time
import uuid

# 로컬 ComfyUI 서버로 이미지를 생성하는 선택 기능입니다.
# settings.toml의 [comfyui] url과 checkpoint가 모두 설정된 경우에만 도구가 에이전트에 등록됩니다.
# (ComfyUI가 없는 일반 사용자에게는 도구 자체가 보이지 않음)


def is_comfyui_enabled() -> bool:
    """ComfyUI 이미지 생성 기능 사용 여부 (url과 checkpoint가 모두 설정되어야 함)"""
    return bool(config.comfyui_url.strip()) and bool(config.comfyui_checkpoint.strip())


def _build_workflow(prompt: str, seed: int) -> dict:
    """기본 text-to-image 워크플로우(API 형식)를 구성합니다.

    CheckpointLoaderSimple -> CLIPTextEncode(긍정/부정) -> KSampler -> VAEDecode -> SaveImage.
    Flux Schnell 같은 단일 체크포인트 모델도 steps/cfg/scheduler 설정만 맞추면 그대로 동작합니다.
    """
    return {
        "4": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": config.comfyui_checkpoint}},
        "5": {"class_type": "EmptyLatentImage", "inputs": {
            "width": config.comfyui_width, "height": config.comfyui_height, "batch_size": 1}},
        "6": {"class_type": "CLIPTextEncode", "inputs": {"text": prompt, "clip": ["4", 1]}},
        "7": {"class_type": "CLIPTextEncode", "inputs": {"text": config.comfyui_negative_prompt, "clip": ["4", 1]}},
        "3": {"class_type": "KSampler", "inputs": {
            "seed": seed, "steps": config.comfyui_steps, "cfg": config.comfyui_cfg,
            "sampler_name": config.comfyui_sampler, "scheduler": config.comfyui_scheduler, "denoise": 1.0,
            "model": ["4", 0], "positive": ["6", 0], "negative": ["7", 0], "latent_image": ["5", 0]}},
        "8": {"class_type": "VAEDecode", "inputs": {"samples": ["3", 0], "vae": ["4", 2]}},
        "9": {"class_type": "SaveImage", "inputs": {"filename_prefix": "jia", "images": ["8", 0]}},
    }


def _generate_image_bytes(prompt: str) -> bytes:
    """ComfyUI API로 이미지를 생성하고 PNG 바이트를 반환합니다. 실패 시 예외를 던집니다."""
    base_url = config.comfyui_url.strip().rstrip("/")
    seed = random.randint(0, 2**31 - 1)
    workflow = _build_workflow(prompt, seed)

    # 1. 생성 작업 등록
    response = requests.post(
        f"{base_url}/prompt",
        json={"prompt": workflow, "client_id": str(uuid.uuid4())},
        timeout=10,
    )
    response.raise_for_status()
    prompt_id = response.json()["prompt_id"]
    logging.info(f"[ComfyUI] 이미지 생성을 시작했어요. (prompt_id={prompt_id}, seed={seed})")

    # 2. 완료될 때까지 폴링
    deadline = time.monotonic() + config.comfyui_timeout_sec
    outputs = None
    while time.monotonic() < deadline:
        time.sleep(1.0)
        history = requests.get(f"{base_url}/history/{prompt_id}", timeout=10).json()
        entry = history.get(prompt_id)
        if not entry:
            continue
        status = entry.get("status", {})
        if status.get("status_str") == "error":
            raise RuntimeError(f"ComfyUI에서 생성 작업이 실패했습니다: {status}")
        if entry.get("outputs"):
            outputs = entry["outputs"]
            break
    if outputs is None:
        raise TimeoutError(f"{config.comfyui_timeout_sec}초 안에 이미지 생성이 끝나지 않았습니다.")

    # 3. 출력 노드에서 이미지 정보를 찾아 다운로드
    for node_output in outputs.values():
        for image in node_output.get("images", []):
            view = requests.get(
                f"{base_url}/view",
                params={"filename": image["filename"], "subfolder": image.get("subfolder", ""), "type": image.get("type", "output")},
                timeout=30,
            )
            view.raise_for_status()
            return view.content
    raise RuntimeError("생성 결과에서 이미지를 찾지 못했습니다.")


class GenerateImageInput(BaseModel):
    """generate_image 도구의 입력"""
    prompt: str = Field(description="생성할 장면을 묘사하는 영어 프롬프트. 쉼표로 구분된 키워드 또는 짧은 문장.")
    wait_message: str = Field(default="", description="이미지를 그리는 동안 채널에 먼저 보여줄 짧은 한국어 안내 문구. 평소 대화하던 말투로 작성한다. (예: 좋아, 잠깐만 기다려봐. 금방 그려올게.)")


def comfyui_image_tool(guild_id: int, prefer_voice_channel: bool) -> StructuredTool:
    """ComfyUI로 이미지를 생성해 디스코드 채널에 올리는 LangChain 도구

    먼저 대기 안내 문구를 채널에 보내두고, 생성이 끝나면 그 메시지를 수정해 이미지를 첨부합니다.
    prefer_voice_channel: 음성 대화용 에이전트면 True. 접속 중인 음성 채널의 채팅에 우선 전송합니다.
    """

    def _generate(prompt: str, wait_message: str = "") -> str:
        prompt = (prompt or "").strip().strip("\"'")
        if not prompt:
            return "생성할 이미지에 대한 영어 프롬프트를 입력해야 합니다."
        if not is_comfyui_enabled():
            return "이미지 생성 기능이 설정되어 있지 않습니다. (settings.toml의 [comfyui] url/checkpoint 필요)"

        # 순환 임포트 방지를 위해 사용 직전에 임포트 (pipeline -> langchain_llm -> comfyui_image)
        from discord_interface import pipeline

        # 1. 생성을 시작하기 전에 대기 안내 문구를 먼저 보냄 (생성이 끝나면 이 메시지를 수정해 이미지를 첨부)
        wait_text = (wait_message or "").strip() or "잠깐만, 그림 그리는 중이야."
        placeholder, note = pipeline.send_placeholder_message(guild_id, wait_text, prefer_voice_channel=prefer_voice_channel)
        if placeholder is None:
            return f"이미지를 보낼 채널을 찾지 못해 생성을 시작하지 않았습니다: {note}"

        # 2. 이미지 생성
        try:
            image_bytes = _generate_image_bytes(prompt)
        except requests.ConnectionError:
            logging.error(f"[ComfyUI] 서버({config.comfyui_url})에 연결할 수 없어요.")
            pipeline.edit_message_text(placeholder, "이미지 생성 서버에 연결하지 못했어요.")
            return "ComfyUI 서버에 연결할 수 없습니다. 서버가 실행 중인지 확인이 필요하다고 사용자에게 알려주세요."
        except Exception as e:
            logging.error(f"[ComfyUI] 이미지 생성 중 오류 발생: {e}")
            pipeline.edit_message_text(placeholder, "그림을 그리다가 문제가 생겼어요.")
            return f"이미지 생성에 실패했습니다: {e}"

        # 3. 보내둔 안내 메시지를 수정해 이미지를 첨부
        filename = f"jia_{uuid.uuid4().hex[:8]}.png"
        ok, message = pipeline.edit_message_attach_image(placeholder, image_bytes, filename)
        if not ok:
            # 수정에 실패하면 새 메시지로 전송 시도
            ok, message = pipeline.send_image_to_guild(guild_id, image_bytes, filename, prefer_voice_channel=prefer_voice_channel)
        if ok:
            return "이미지를 생성해서 채널에 올렸습니다. 사용자에게 이미지를 확인해보라고 자연스럽게 알려주세요."
        return f"이미지는 생성했지만 채널에 올리지 못했습니다: {message}"

    return StructuredTool.from_function(
        func=_generate,
        name="generate_image",
        args_schema=GenerateImageInput,
        description=(
            "텍스트 프롬프트로 이미지를 생성해서 지금 대화 중인 디스코드 채널에 올립니다. "
            "사용자가 그림이나 이미지를 그려달라고 할 때 사용하세요. "
            "prompt에는 생성할 장면을 묘사하는 영어 프롬프트를, "
            "wait_message에는 그리는 동안 채널에 먼저 보여줄 너의 말투의 짧은 한국어 안내 문구를 넣으세요. "
            "안내 문구가 먼저 올라가고, 생성이 끝나면 그 메시지가 이미지로 바뀝니다."
        ),
    )
