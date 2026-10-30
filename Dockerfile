# Project Jia 도커 이미지 (Linux + NVIDIA GPU)
# 빌드/실행: docker compose up -d --build   (자세한 내용은 README의 "Docker로 실행하기")

# faster-whisper(CTranslate2)가 CUDA 12 + cuDNN 9를 쓰므로 cuDNN 런타임 이미지를 사용합니다.
FROM nvidia/cuda:12.6.3-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_INSTALL_DIR=/opt/python \
    UV_PROJECT_ENVIRONMENT=/opt/venv \
    PATH=/opt/venv/bin:$PATH

# ffmpeg: 음악/효과음 재생, libopus: 디스코드 음성 인코딩, libsndfile: soundfile
RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg libopus0 libsndfile1 ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# uv(패키지 설치)와 uvx(기본 검색 MCP 서버 실행)
COPY --from=ghcr.io/astral-sh/uv:0.9.9 /uv /uvx /usr/local/bin/

WORKDIR /app

# 의존성만 먼저 설치해서 코드가 바뀌어도 이 층은 캐시를 재사용
COPY pyproject.toml uv.lock .python-version ./
RUN --mount=type=cache,target=/root/.cache/uv uv sync --frozen --no-install-project

COPY . .
RUN sed -i 's/\r$//' docker/entrypoint.sh && chmod +x docker/entrypoint.sh

# 설정, 기억, 효과음, 모델 캐시는 모두 /app/data에 저장됩니다. (docker-compose.yml에서 폴더로 연결)
VOLUME ["/app/data"]
ENTRYPOINT ["/app/docker/entrypoint.sh"]
CMD ["python", "main.py"]
