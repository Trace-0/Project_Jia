from langchain_core.tools import Tool
from dataclasses import dataclass
from pathlib import Path
import logging
import random
import re
import threading
import time
import tomlkit
from config.config_manager import config

# 사운드보드 전용 폴더. LLM은 이 폴더 안의 오디오 파일만 재생할 수 있습니다.
# (임의 경로의 파일을 읽거나 재생하는 것을 막기 위해 위치를 코드에 고정합니다)
SOUNDBOARD_DIR = Path(__file__).resolve().parents[2] / "soundboard"
REGISTRY_PATH = SOUNDBOARD_DIR / "sounds.toml"
ALLOWED_EXTENSIONS = {".mp3", ".wav", ".ogg", ".flac", ".m4a", ".opus", ".webm", ".aac"}

AUTO_REACTION_KEYWORDS = {
    "success": ("성공", "해냈", "됐다", "이겼", "승리", "클리어", "합격", "완료", "대박", "좋았", "축하", "success", "win"),
    "celebrate": ("축하", "생일", "기념", "파티", "만세", "축배", "celebrate", "party", "birthday"),
    "fail": ("실패", "망했다", "틀렸", "졌어", "패배", "아깝", "실수", "fail", "lose"),
    "awkward": ("정적", "어색", "민망", "갑분싸", "침묵", "awkward", "silence"),
    "laugh": ("웃겨", "웃기", "ㅋㅋ", "하하", "농담", "개그", "드립", "laugh", "funny", "joke"),
    "surprise": ("뭐야", "헉", "깜짝", "놀랐", "진짜", "미쳤", "surprise", "wow"),
    "sad": ("슬퍼", "아쉽", "우울", "눈물", "sad", "cry"),
    "angry": ("화나", "빡", "짜증", "열받", "분노", "angry"),
}

_auto_reaction_lock = threading.Lock()
_auto_reaction_last_played: dict[tuple[int, str], float] = {}


@dataclass(frozen=True)
class SoundEntry:
    file_name: str
    description: str = ""
    tags: tuple[str, ...] = ()
    auto: bool = True
    cooldown_sec: float | None = None
    chance: float | None = None

    def display_description(self) -> str:
        parts = []
        if self.description:
            parts.append(self.description)
        if self.tags:
            parts.append(f"tags={', '.join(self.tags)}")
        return " / ".join(parts) if parts else "(설명 없음)"


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


def _as_float(value, default: float | None = None) -> float | None:
    if value is None:
        return default
    try:
        return float(_unwrap_toml(value))
    except (TypeError, ValueError):
        return default


def _entry_from_toml(file_name: str, raw_value) -> SoundEntry:
    raw = _unwrap_toml(raw_value)
    if isinstance(raw, dict):
        description = str(raw.get("desc", raw.get("description", ""))).strip()
        chance = _as_float(raw.get("chance"))
        if chance is not None:
            chance = max(0.0, min(1.0, chance))
        return SoundEntry(
            file_name=file_name,
            description=description,
            tags=_as_tags(raw.get("tags", ())),
            auto=bool(raw.get("auto", True)),
            cooldown_sec=_as_float(raw.get("cooldown", raw.get("cooldown_sec"))),
            chance=chance,
        )
    return SoundEntry(file_name=file_name, description=str(raw_value or "").strip())


def load_sound_registry() -> dict[str, SoundEntry]:
    """폴더를 스캔해 {파일명: SoundEntry}를 반환합니다.

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
    return {name: _entry_from_toml(name, doc.get(name, "")) for name in files}


def _format_sound_list(sounds: dict[str, SoundEntry]) -> str:
    return "\n".join(
        f"- {name}: {entry.display_description()}"
        for name, entry in sounds.items()
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


def _normalize_text(text: str) -> str:
    return (text or "").lower()


def _words(text: str) -> set[str]:
    return set(re.findall(r"[0-9a-zA-Z가-힣_]+", _normalize_text(text)))


def _score_auto_reaction(entry: SoundEntry, context: str) -> int:
    text = _normalize_text(context)
    words = _words(context)
    score = 0

    for tag in entry.tags:
        if tag in words or tag in text:
            score += 4
        for keyword in AUTO_REACTION_KEYWORDS.get(tag, ()):
            if keyword.lower() in text:
                score += 5

    for word in _words(entry.description):
        if len(word) >= 2 and word in text:
            score += 1

    stem = Path(entry.file_name).stem.lower()
    if stem and stem in text:
        score += 2
    return score


def maybe_play_auto_reaction(guild_id: int, context: str) -> tuple[bool, str]:
    """대화 문맥에 맞는 효과음을 자동으로 골라 재생합니다.

    자동 반응은 settings.toml의 [soundboard] auto_react가 켜져 있을 때만 동작합니다.
    각 효과음은 sounds.toml에서 auto=false, cooldown, chance로 개별 제어할 수 있습니다.
    """
    if not config.soundboard_auto_react:
        return False, "사운드보드 자동 반응이 꺼져 있습니다."
    sounds = load_sound_registry()
    if not sounds:
        return False, "등록된 효과음이 없습니다."

    candidates = [
        (score, entry)
        for entry in sounds.values()
        if entry.auto and (score := _score_auto_reaction(entry, context)) > 0
    ]
    if not candidates:
        return False, "문맥에 맞는 효과음 후보가 없습니다."

    candidates.sort(key=lambda item: item[0], reverse=True)
    best_score = candidates[0][0]
    top_candidates = [entry for score, entry in candidates if score == best_score]
    entry = random.choice(top_candidates)

    chance = entry.chance if entry.chance is not None else config.soundboard_auto_react_chance
    chance = max(0.0, min(1.0, chance))
    if random.random() > chance:
        return False, f"자동 반응 확률로 '{entry.file_name}' 재생을 건너뜁니다."

    cooldown = entry.cooldown_sec if entry.cooldown_sec is not None else config.soundboard_auto_react_cooldown_sec
    now = time.monotonic()
    key = (guild_id, entry.file_name)
    with _auto_reaction_lock:
        last_played = _auto_reaction_last_played.get(key, 0.0)
        if cooldown > 0 and now - last_played < cooldown:
            return False, f"'{entry.file_name}' 효과음은 쿨다운 중입니다."

    path = _resolve_sound_path(entry.file_name)
    if path is None:
        return False, f"'{entry.file_name}' 효과음은 재생할 수 없는 파일입니다."

    from discord_interface import pipeline

    logging.info(f"[Soundboard] 자동 반응 효과음 재생: {entry.file_name} (guild={guild_id}, score={best_score})")
    ok, message = pipeline.play_sound_file(guild_id, str(path))
    if ok:
        with _auto_reaction_lock:
            _auto_reaction_last_played[key] = time.monotonic()
    return ok, message


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
