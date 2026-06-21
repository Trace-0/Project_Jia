from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass, replace
from typing import Any
from urllib.parse import urlparse


try:
    import yt_dlp
except ImportError:  # pragma: no cover - runtime dependency check
    yt_dlp = None


YOUTUBE_FORMAT = "bestaudio/best"
URL_SCHEME_RE = re.compile(r"^[A-Za-z][A-Za-z0-9+.-]*://")
DEFAULT_ALLOWED_MUSIC_URL_HOSTS = ("youtube.com", "youtu.be")


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


def _normalise_allowed_hosts(allowed_url_hosts: Iterable[str] | None) -> tuple[str, ...]:
    if allowed_url_hosts is None:
        allowed_url_hosts = DEFAULT_ALLOWED_MUSIC_URL_HOSTS
    if not allowed_url_hosts:
        return ()
    hosts = []
    for host in allowed_url_hosts:
        normalised = str(host).strip().lower().strip(".")
        if normalised:
            hosts.append(normalised)
    return tuple(dict.fromkeys(hosts))


def _host_matches(hostname: str, allowed_host: str) -> bool:
    return hostname == allowed_host or hostname.endswith(f".{allowed_host}")


def _format_allowed_hosts(allowed_url_hosts: Iterable[str] | None) -> str:
    hosts = _normalise_allowed_hosts(allowed_url_hosts)
    return ", ".join(hosts) if hosts else "없음"


def is_allowed_music_url(url: str, allowed_url_hosts: Iterable[str] | None) -> bool:
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        return False
    hostname = parsed.hostname.lower().strip(".")
    return any(_host_matches(hostname, allowed) for allowed in _normalise_allowed_hosts(allowed_url_hosts))


def _has_bare_allowed_host(query: str, allowed_url_hosts: Iterable[str] | None) -> bool:
    if any(char.isspace() for char in query):
        return False
    hostname = query.split("/", 1)[0].split("?", 1)[0].split("#", 1)[0].lower().strip(".")
    return any(_host_matches(hostname, allowed) for allowed in _normalise_allowed_hosts(allowed_url_hosts))


def validate_music_query(query: str, allowed_url_hosts: Iterable[str] | None) -> str:
    """Return a safe yt-dlp input. Search terms are allowed; URLs are host-restricted."""
    query = query.strip()
    if not query:
        return query

    candidate = query
    if query.lower().startswith("www.") or _has_bare_allowed_host(query, allowed_url_hosts):
        candidate = f"https://{query}"

    if not URL_SCHEME_RE.match(candidate):
        return query

    parsed = urlparse(candidate)
    if parsed.scheme not in {"http", "https"}:
        raise ValueError("음악 URL은 http/https 형식만 사용할 수 있어요. 검색어로 재생하려면 URL 대신 검색어를 입력해주세요.")
    if not parsed.hostname:
        raise ValueError("음악 URL의 호스트를 확인할 수 없어요.")
    if not is_allowed_music_url(candidate, allowed_url_hosts):
        raise ValueError(
            "허용되지 않은 음악 URL이에요. "
            "검색어 또는 허용된 도메인의 URL만 사용할 수 있어요. "
            f"현재 허용: {_format_allowed_hosts(allowed_url_hosts)}"
        )
    return candidate


def _is_url(query: str) -> bool:
    return query.lower().startswith(("http://", "https://"))


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


def _extract_query(query: str, allowed_url_hosts: Iterable[str] | None) -> str:
    query = validate_music_query(query, allowed_url_hosts)
    if _is_url(query):
        return query
    return f"ytsearch1:{query}"


def build_music_queue(query: str, max_items: int, allowed_url_hosts: Iterable[str] | None = None) -> list[MusicTrack]:
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
        info = ydl.extract_info(_extract_query(query, allowed_url_hosts), download=False)

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
            if not is_allowed_music_url(webpage_url, allowed_url_hosts):
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
    if not is_allowed_music_url(webpage_url, allowed_url_hosts):
        raise ValueError(
            "허용되지 않은 음악 URL이에요. "
            f"현재 허용: {_format_allowed_hosts(allowed_url_hosts)}"
        )
    return [
        MusicTrack(
            title=_entry_title(info, webpage_url),
            webpage_url=webpage_url,
            stream_url=info.get("url"),
            duration=info.get("duration"),
        )
    ]


def resolve_music_track(track: MusicTrack, allowed_url_hosts: Iterable[str] | None = None) -> MusicTrack:
    """Resolve a queue item into a playable audio stream URL."""
    if not is_allowed_music_url(track.webpage_url, allowed_url_hosts):
        raise RuntimeError(
            "허용되지 않은 음악 URL이라 재생하지 않았어요. "
            f"현재 허용: {_format_allowed_hosts(allowed_url_hosts)}"
        )
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
        webpage_url=validate_music_query(info.get("webpage_url") or track.webpage_url, allowed_url_hosts),
        stream_url=stream_url,
        duration=info.get("duration") or track.duration,
    )
