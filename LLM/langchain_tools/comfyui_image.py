from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field
from dataclasses import dataclass
from config.config_manager import config
from config.config import SETTINGS_PATH
import requests
import logging
import random
import time
import uuid
import re
import tomlkit

# 로컬 ComfyUI 서버로 이미지를 생성하는 선택 기능입니다.
# settings.toml의 [comfyui] url과 checkpoint 또는 [comfyui.models.*]가 설정된 경우에만 도구가 에이전트에 등록됩니다.
# (ComfyUI가 없는 일반 사용자에게는 도구 자체가 보이지 않음)


@dataclass(frozen=True)
class ComfyUIModelProfile:
    model_id: str
    checkpoint: str
    use_when: str = ""
    tags: tuple[str, ...] = ()
    steps: int = 20
    cfg: float = 7.0
    width: int = 1024
    height: int = 1024
    sampler: str = "euler"
    scheduler: str = "normal"
    negative_prompt: str = ""


def _unwrap_toml(value):
    return value.unwrap() if hasattr(value, "unwrap") else value


def _as_tags(value) -> tuple[str, ...]:
    raw = _unwrap_toml(value)
    if raw is None:
        return ()
    if isinstance(raw, str):
        return tuple(t.strip().lower() for t in raw.split(",") if t.strip())
    if isinstance(raw, (list, tuple)):
        return tuple(str(t).strip().lower() for t in raw if str(t).strip())
    return ()


def _as_int(value, default: int) -> int:
    try:
        return int(_unwrap_toml(value))
    except (TypeError, ValueError):
        return default


def _as_float(value, default: float) -> float:
    try:
        return float(_unwrap_toml(value))
    except (TypeError, ValueError):
        return default


def _default_model_profile() -> ComfyUIModelProfile | None:
    checkpoint = config.comfyui_checkpoint.strip()
    if not checkpoint:
        return None
    return ComfyUIModelProfile(
        model_id="default",
        checkpoint=checkpoint,
        use_when="기본 이미지 생성 모델. 별도 모델을 고르기 애매할 때 사용합니다.",
        steps=config.comfyui_steps,
        cfg=config.comfyui_cfg,
        width=config.comfyui_width,
        height=config.comfyui_height,
        sampler=config.comfyui_sampler,
        scheduler=config.comfyui_scheduler,
        negative_prompt=config.comfyui_negative_prompt,
    )


def _profile_from_mapping(model_id: str, raw_value) -> ComfyUIModelProfile | None:
    raw = _unwrap_toml(raw_value)
    if not isinstance(raw, dict):
        return None
    checkpoint = str(raw.get("checkpoint", "")).strip()
    if not checkpoint:
        return None
    default = _default_model_profile()
    return ComfyUIModelProfile(
        model_id=model_id,
        checkpoint=checkpoint,
        use_when=str(raw.get("use_when", raw.get("description", ""))).strip(),
        tags=_as_tags(raw.get("tags", ())),
        steps=_as_int(raw.get("steps"), default.steps if default else config.comfyui_steps),
        cfg=_as_float(raw.get("cfg"), default.cfg if default else config.comfyui_cfg),
        width=_as_int(raw.get("width"), default.width if default else config.comfyui_width),
        height=_as_int(raw.get("height"), default.height if default else config.comfyui_height),
        sampler=str(raw.get("sampler", default.sampler if default else config.comfyui_sampler)).strip(),
        scheduler=str(raw.get("scheduler", default.scheduler if default else config.comfyui_scheduler)).strip(),
        negative_prompt=str(raw.get("negative_prompt", default.negative_prompt if default else config.comfyui_negative_prompt)).strip(),
    )


def get_comfyui_model_profiles() -> dict[str, ComfyUIModelProfile]:
    """settings.toml에서 사용할 수 있는 ComfyUI 모델 프로필을 읽습니다.

    기존 단일 [comfyui] checkpoint 설정은 "default" 프로필로 유지하고,
    추가로 [comfyui.models.<model_id>] 테이블을 상황별 모델 후보로 등록합니다.
    """
    profiles: dict[str, ComfyUIModelProfile] = {}
    default = _default_model_profile()
    if default:
        profiles[default.model_id] = default

    if not SETTINGS_PATH.exists():
        return profiles
    try:
        doc = tomlkit.parse(SETTINGS_PATH.read_text(encoding="utf-8"))
        models = doc.get("comfyui", {}).get("models", {})
    except Exception as e:
        logging.error(f"[ComfyUI] settings.toml의 모델 프로필을 읽지 못했어요: {e}")
        return profiles

    if not hasattr(models, "items"):
        return profiles
    for model_id, raw_model in models.items():
        profile = _profile_from_mapping(str(model_id), raw_model)
        if profile:
            profiles[profile.model_id] = profile
    return profiles


def is_comfyui_enabled() -> bool:
    """ComfyUI 이미지 생성 기능 사용 여부 (url과 사용할 모델 프로필이 모두 있어야 함)"""
    return bool(config.comfyui_url.strip()) and bool(get_comfyui_model_profiles())


def _build_workflow(prompt: str, seed: int, profile: ComfyUIModelProfile) -> dict:
    """기본 text-to-image 워크플로우(API 형식)를 구성합니다.

    CheckpointLoaderSimple -> CLIPTextEncode(긍정/부정) -> KSampler -> VAEDecode -> SaveImage.
    Flux Schnell 같은 단일 체크포인트 모델도 steps/cfg/scheduler 설정만 맞추면 그대로 동작합니다.
    """
    return {
        "4": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": profile.checkpoint}},
        "5": {"class_type": "EmptyLatentImage", "inputs": {
            "width": profile.width, "height": profile.height, "batch_size": 1}},
        "6": {"class_type": "CLIPTextEncode", "inputs": {"text": prompt, "clip": ["4", 1]}},
        "7": {"class_type": "CLIPTextEncode", "inputs": {"text": profile.negative_prompt, "clip": ["4", 1]}},
        "3": {"class_type": "KSampler", "inputs": {
            "seed": seed, "steps": profile.steps, "cfg": profile.cfg,
            "sampler_name": profile.sampler, "scheduler": profile.scheduler, "denoise": 1.0,
            "model": ["4", 0], "positive": ["6", 0], "negative": ["7", 0], "latent_image": ["5", 0]}},
        "8": {"class_type": "VAEDecode", "inputs": {"samples": ["3", 0], "vae": ["4", 2]}},
        "9": {"class_type": "SaveImage", "inputs": {"filename_prefix": "jia", "images": ["8", 0]}},
    }


def _tokens(text: str) -> set[str]:
    return set(re.findall(r"[0-9a-zA-Z가-힣_]+", (text or "").lower()))


def _select_model_profile(prompt: str, requested_model_id: str = "") -> ComfyUIModelProfile:
    profiles = get_comfyui_model_profiles()
    if not profiles:
        raise RuntimeError("사용 가능한 ComfyUI 모델 프로필이 없습니다.")

    requested = (requested_model_id or "").strip()
    if requested and requested in profiles:
        return profiles[requested]

    if len(profiles) == 1:
        return next(iter(profiles.values()))

    prompt_text = (prompt or "").lower()
    prompt_tokens = _tokens(prompt)
    scored = []
    for profile in profiles.values():
        score = 0
        for tag in profile.tags:
            tag_text = tag.lower()
            if tag_text in prompt_tokens or tag_text in prompt_text:
                score += 4
        for token in _tokens(profile.use_when):
            if len(token) >= 2 and token in prompt_text:
                score += 1
        if profile.model_id == "default":
            score -= 1
        scored.append((score, profile))
    scored.sort(key=lambda item: item[0], reverse=True)
    if scored and scored[0][0] > 0:
        return scored[0][1]
    return profiles.get("default") or next(iter(profiles.values()))


def _format_model_profiles(profiles: dict[str, ComfyUIModelProfile]) -> str:
    if not profiles:
        return "(등록된 모델 프로필 없음)"
    lines = []
    for profile in profiles.values():
        details = []
        if profile.use_when:
            details.append(profile.use_when)
        if profile.tags:
            details.append(f"tags={', '.join(profile.tags)}")
        details.append(f"checkpoint={profile.checkpoint}")
        lines.append(f"- {profile.model_id}: {' / '.join(details)}")
    return "\n".join(lines)


def _generate_image_bytes(prompt: str, profile: ComfyUIModelProfile) -> bytes:
    """ComfyUI API로 이미지를 생성하고 PNG 바이트를 반환합니다. 실패 시 예외를 던집니다."""
    base_url = config.comfyui_url.strip().rstrip("/")
    seed = random.randint(0, 2**31 - 1)
    workflow = _build_workflow(prompt, seed, profile)

    # 1. 생성 작업 등록
    response = requests.post(
        f"{base_url}/prompt",
        json={"prompt": workflow, "client_id": str(uuid.uuid4())},
        timeout=10,
    )
    response.raise_for_status()
    prompt_id = response.json()["prompt_id"]
    logging.info(f"[ComfyUI] 이미지 생성을 시작했어요. (model={profile.model_id}, checkpoint={profile.checkpoint}, prompt_id={prompt_id}, seed={seed})")

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
    model_id: str = Field(default="", description="사용할 ComfyUI 모델 프로필 ID. 상황에 맞는 ID를 고르고, 애매하면 비워둔다.")


def comfyui_image_tool(guild_id: int, prefer_voice_channel: bool) -> StructuredTool:
    """ComfyUI로 이미지를 생성해 디스코드 채널에 올리는 LangChain 도구

    먼저 대기 안내 문구를 채널에 보내두고, 생성이 끝나면 그 메시지를 수정해 이미지를 첨부합니다.
    prefer_voice_channel: 음성 대화용 에이전트면 True. 접속 중인 음성 채널의 채팅에 우선 전송합니다.
    """

    def _generate(prompt: str, wait_message: str = "", model_id: str = "") -> str:
        prompt = (prompt or "").strip().strip("\"'")
        if not prompt:
            return "생성할 이미지에 대한 영어 프롬프트를 입력해야 합니다."
        if not is_comfyui_enabled():
            return "이미지 생성 기능이 설정되어 있지 않습니다. (settings.toml의 [comfyui] url과 checkpoint 또는 [comfyui.models.*] 필요)"

        try:
            profile = _select_model_profile(prompt, model_id)
        except Exception as e:
            return f"이미지 생성 모델을 선택하지 못했습니다: {e}"

        # 순환 임포트 방지를 위해 사용 직전에 임포트 (pipeline -> langchain_llm -> comfyui_image)
        from discord_interface import pipeline

        # 1. 생성을 시작하기 전에 대기 안내 문구를 먼저 보냄 (생성이 끝나면 이 메시지를 수정해 이미지를 첨부)
        wait_text = (wait_message or "").strip() or "잠깐만, 그림 그리는 중이야."
        placeholder, note = pipeline.send_placeholder_message(guild_id, wait_text, prefer_voice_channel=prefer_voice_channel)
        if placeholder is None:
            return f"이미지를 보낼 채널을 찾지 못해 생성을 시작하지 않았습니다: {note}"

        # 2. 이미지 생성
        try:
            image_bytes = _generate_image_bytes(prompt, profile)
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
            return f"이미지를 생성해서 채널에 올렸습니다. 사용한 모델 프로필은 '{profile.model_id}'입니다. 사용자에게 이미지를 확인해보라고 자연스럽게 알려주세요."
        return f"이미지는 생성했지만 채널에 올리지 못했습니다: {message}"

    profiles = get_comfyui_model_profiles()
    return StructuredTool.from_function(
        func=_generate,
        name="generate_image",
        args_schema=GenerateImageInput,
        description=(
            "텍스트 프롬프트로 이미지를 생성해서 지금 대화 중인 디스코드 채널에 올립니다. "
            "사용자가 그림이나 이미지를 그려달라고 할 때 사용하세요. "
            "prompt에는 생성할 장면을 묘사하는 영어 프롬프트를, "
            "wait_message에는 그리는 동안 채널에 먼저 보여줄 너의 말투의 짧은 한국어 안내 문구를 넣으세요. "
            "model_id에는 아래 모델 프로필 중 상황에 맞는 ID를 넣으세요. 애매하면 비워두면 됩니다. "
            "안내 문구가 먼저 올라가고, 생성이 끝나면 그 메시지가 이미지로 바뀝니다.\n"
            f"사용 가능한 모델 프로필:\n{_format_model_profiles(profiles)}"
        ),
    )
