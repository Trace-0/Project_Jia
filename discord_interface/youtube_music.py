from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any


try:
    import yt_dlp
except ImportError:  # pragma: no cover - runtime dependency check
    yt_dlp = None


YOUTUBE_FORMAT = "bestaudio/best"


@dataclass(frozen=True)
class MusicTrack:
    title: str
    webpage_url: str
    stream_url: str | None = None
    duration: int | None = None


def _require_ytdlp():
    if yt_dlp is None:
        raise RuntimeError("yt-dlp가 설치되어 있지 않습니다. requirements.txt를 다시 설치해주세요.")
    return yt_dlp


def _is_url(query: str) -> bool:
    return query.startswith("http://") or query.startswith("https://")


def _entry_url(entry: dict[str, Any]) -> str | None:
    url = entry.get("webpage_url") or entry.get("original_url") or entry.get("url")
    if not url:
        return None
    if isinstance(url, str) and url.startswith(("http://", "https://")):
        return url
    if entry.get("ie_key") == "Youtube":
        return f"https://www.youtube.com/watch?v={url}"
    return str(url)


def _entry_title(entry: dict[str, Any], fallback_url: str) -> str:
    return str(entry.get("title") or entry.get("fulltitle") or fallback_url)


def _base_options() -> dict[str, Any]:
    return {
        "format": YOUTUBE_FORMAT,
        "quiet": True,
        "no_warnings": True,
        "ignoreerrors": True,
        "skip_download": True,
        "source_address": "0.0.0.0",
    }


def _extract_query(query: str) -> str:
    query = query.strip()
    if _is_url(query):
        return query
    return f"ytsearch1:{query}"


def build_music_queue(query: str, max_items: int) -> list[MusicTrack]:
    """Resolve a YouTube URL, playlist URL, or search query into queueable tracks."""
    ytdlp = _require_ytdlp()
    query = query.strip()
    if not query:
        return []

    limit = max(1, max_items)
    options = _base_options() | {
        "extract_flat": "in_playlist",
        "playlistend": limit,
        "noplaylist": False,
    }
    with ytdlp.YoutubeDL(options) as ydl:
        info = ydl.extract_info(_extract_query(query), download=False)

    if not info:
        return []

    if info.get("entries"):
        tracks: list[MusicTrack] = []
        for entry in info.get("entries") or []:
            if not entry:
                continue
            webpage_url = _entry_url(entry)
            if not webpage_url:
                continue
            tracks.append(
                MusicTrack(
                    title=_entry_title(entry, webpage_url),
                    webpage_url=webpage_url,
                    duration=entry.get("duration"),
                )
            )
            if len(tracks) >= limit:
                break
        return tracks

    webpage_url = _entry_url(info)
    if not webpage_url:
        return []
    return [
        MusicTrack(
            title=_entry_title(info, webpage_url),
            webpage_url=webpage_url,
            stream_url=info.get("url"),
            duration=info.get("duration"),
        )
    ]


def resolve_music_track(track: MusicTrack) -> MusicTrack:
    """Resolve a queue item into a playable audio stream URL."""
    if track.stream_url:
        return track

    ytdlp = _require_ytdlp()
    options = _base_options() | {
        "noplaylist": True,
    }
    with ytdlp.YoutubeDL(options) as ydl:
        info = ydl.extract_info(track.webpage_url, download=False)

    if not info:
        raise RuntimeError(f"음악 정보를 불러오지 못했습니다: {track.title}")

    stream_url = info.get("url")
    if not stream_url:
        formats = info.get("formats") or []
        audio_formats = [fmt for fmt in formats if fmt.get("url") and fmt.get("acodec") != "none"]
        if audio_formats:
            stream_url = audio_formats[-1].get("url")

    if not stream_url:
        raise RuntimeError(f"재생 가능한 오디오 스트림을 찾지 못했습니다: {track.title}")

    return replace(
        track,
        title=_entry_title(info, track.title),
        webpage_url=info.get("webpage_url") or track.webpage_url,
        stream_url=stream_url,
        duration=info.get("duration") or track.duration,
    )
