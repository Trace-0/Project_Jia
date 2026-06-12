from langchain_core.tools import Tool
from pathlib import Path
import logging
import tomlkit

# 사운드보드 전용 폴더. LLM은 이 폴더 안의 오디오 파일만 재생할 수 있습니다.
# (임의 경로의 파일을 읽거나 재생하는 것을 막기 위해 위치를 코드에 고정합니다)
SOUNDBOARD_DIR = Path(__file__).resolve().parents[2] / "soundboard"
REGISTRY_PATH = SOUNDBOARD_DIR / "sounds.toml"
ALLOWED_EXTENSIONS = {".mp3", ".wav", ".ogg", ".flac", ".m4a", ".opus", ".webm", ".aac"}


def _ensure_soundboard_dir():
    """soundboard 폴더와 설명 파일(sounds.toml)이 없으면 만듭니다."""
    SOUNDBOARD_DIR.mkdir(exist_ok=True)
    if not REGISTRY_PATH.exists():
        doc = tomlkit.document()
        doc.add(tomlkit.comment("사운드보드 효과음 설명 파일"))
        doc.add(tomlkit.comment("이 폴더에 오디오 파일을 넣으면 아래에 항목이 자동으로 추가돼요."))
        doc.add(tomlkit.comment("각 파일의 설명을 적어주면 지아(LLM)가 어떤 효과음인지 이해하고 상황에 맞게 재생해요."))
        doc.add(tomlkit.comment('예: "tada.mp3" = "축하하거나 무언가에 성공했을 때 쓰는 빰빠밤 팡파레 효과음"'))
        doc.add(tomlkit.nl())
        REGISTRY_PATH.write_text(tomlkit.dumps(doc), encoding="utf-8")


def _scan_audio_files() -> list[str]:
    """soundboard 폴더 바로 아래의 허용된 확장자 오디오 파일 이름 목록을 반환합니다."""
    return sorted(
        p.name for p in SOUNDBOARD_DIR.iterdir()
        if p.is_file() and p.suffix.lower() in ALLOWED_EXTENSIONS
    )


def load_sound_registry() -> dict[str, str]:
    """폴더를 스캔해 {파일명: 설명}을 반환합니다.

    설명이 아직 없는 새 파일은 sounds.toml에 빈 항목으로 자동 등록해,
    사용자가 파일을 넣은 뒤 설명만 채우면 되도록 합니다.
    """
    _ensure_soundboard_dir()
    try:
        doc = tomlkit.parse(REGISTRY_PATH.read_text(encoding="utf-8"))
    except Exception as e:
        logging.error(f"[Soundboard] sounds.toml을 읽지 못했어요: {e}")
        doc = tomlkit.document()
    files = _scan_audio_files()
    new_files = [name for name in files if name not in doc]
    if new_files:
        for name in new_files:
            doc[name] = ""
        try:
            REGISTRY_PATH.write_text(tomlkit.dumps(doc), encoding="utf-8")
            logging.info(f"[Soundboard] 새 효과음 {len(new_files)}개를 sounds.toml에 등록했어요: {new_files}")
        except Exception as e:
            logging.error(f"[Soundboard] sounds.toml 저장 중 오류 발생: {e}")
    # 파일이 삭제된 항목은 목록에서 제외 (sounds.toml의 항목 자체는 보존)
    return {name: str(doc.get(name, "")).strip() for name in files}


def _format_sound_list(sounds: dict[str, str]) -> str:
    return "\n".join(
        f"- {name}: {desc if desc else '(설명 없음)'}"
        for name, desc in sounds.items()
    )


def _resolve_sound_path(file_name: str) -> Path | None:
    """파일 이름을 soundboard 폴더 안의 경로로 안전하게 변환합니다.

    폴더 밖을 가리키는 경로(상위 디렉터리 탈출, 절대 경로 등)는 None을 반환합니다.
    """
    candidate = (SOUNDBOARD_DIR / file_name).resolve()
    if candidate.parent != SOUNDBOARD_DIR.resolve():
        return None
    if not candidate.is_file() or candidate.suffix.lower() not in ALLOWED_EXTENSIONS:
        return None
    return candidate


def soundboard_tool(guild_id: int) -> Tool:
    """soundboard 폴더의 효과음을 음성 채널에서 재생하는 LangChain 도구"""

    def _play(query: str) -> str:
        # 순환 임포트 방지 (pipeline -> langchain_llm -> soundboard)
        from discord_interface import pipeline

        name = (query or "").strip().strip("\"'")
        sounds = load_sound_registry()
        if not sounds:
            return "사운드보드에 등록된 효과음이 없습니다. 사용자가 soundboard 폴더에 오디오 파일을 넣어야 사용할 수 있습니다."

        # 파일 이름 매칭 (확장자 생략, 대소문자 무시 허용)
        matched = next(
            (fname for fname in sounds
             if name.lower() in (fname.lower(), Path(fname).stem.lower())),
            None,
        )
        if matched is None:
            return (
                f"'{name}' 효과음을 찾지 못했습니다. 사용 가능한 효과음 목록:\n"
                f"{_format_sound_list(sounds)}\n"
                "재생하려면 이 중 하나의 파일 이름을 정확히 입력해 다시 시도하세요."
            )

        path = _resolve_sound_path(matched)
        if path is None:
            return f"'{matched}' 효과음은 재생할 수 없는 파일입니다."

        logging.info(f"[Soundboard] 효과음 재생 요청: {matched} (guild={guild_id})")
        ok, message = pipeline.play_sound_file(guild_id, str(path))
        if ok:
            return f"'{matched}' 효과음을 재생했습니다."
        return f"효과음을 재생하지 못했습니다: {message}"

    sounds = load_sound_registry()
    listing = _format_sound_list(sounds) if sounds else "(아직 등록된 효과음이 없습니다)"
    return Tool(
        name="play_soundboard",
        func=_play,
        description=(
            "음성 채널에서 효과음을 재생합니다. 대화 상황에 어울리는 효과음이 있을 때 사용하세요. "
            "재생할 효과음의 파일 이름을 입력해야 합니다.\n"
            f"사용 가능한 효과음 목록:\n{listing}"
        ),
    )
