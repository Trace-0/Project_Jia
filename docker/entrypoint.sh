#!/bin/sh
# 컨테이너를 다시 만들어도 설정/기억/효과음/모델이 남도록 모두 /app/data(호스트 폴더)에 두고 연결합니다.
set -e

DATA=/app/data
mkdir -p "$DATA/memory" "$DATA/soundboard" "$DATA/models" "$DATA/cache"

# 설정 파일이 없으면 빈 파일을 만들어 두면, 지아가 처음 실행할 때 기본값과 설명을 채워 넣습니다.
[ -f "$DATA/settings.toml" ] || touch "$DATA/settings.toml"
ln -sf "$DATA/settings.toml" /app/settings.toml

rm -rf /app/memory/data /app/soundboard
ln -s "$DATA/memory" /app/memory/data
ln -s "$DATA/soundboard" /app/soundboard

# Whisper/임베딩/VAD/문장 끝 판정 모델과 uvx 캐시도 data 폴더에 보관 (다시 받지 않도록)
export HF_HOME="$DATA/cache/huggingface"
export TORCH_HOME="$DATA/cache/torch"
export UV_CACHE_DIR="$DATA/cache/uv"

exec "$@"
