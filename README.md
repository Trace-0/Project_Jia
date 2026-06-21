# Project_Jia
AI bot that can chat and converse on Discord

디스코드에서 지아와 함께 채팅 또는 대화를 하며 대화해봅시다!

[설치](https://github.com/Trace-0/Project_Jia?tab=readme-ov-file#1-설치) / [커맨드](https://github.com/Trace-0/Project_Jia?tab=readme-ov-file#2-지아의-커맨드) / [변경사항](https://github.com/Trace-0/Project_Jia?tab=readme-ov-file#5-변경-내용)


## 지아가 할 수 있는 것

- 음성을 듣고 대답할 수 있어요.
- 채팅을 보고 대답할 수 있어요.
- 모르는 내용이 있으면 언제든 인터넷에서 정보를 검색할 수 있어요.
- 이전 대화를 기억하고 활용할 수 있어요. (서버별로 기억해서 다른 서버에서 기억에 대해 물어보면 곤란해할 수 있어요.)
- 사람처럼 기억을 잊기도 해요. 중요하지 않은 기억은 시간이 지나면 흐려지다 사라지고, 자주 꺼내 본 기억은 더 오래 남아요.
- 대화하면서 알게 된 여러분에 대한 사실(취향, 관계 등)을 사람별로 기억해요. 내 기억 상태와 프로필은 `/jiamemory status`로 확인할 수 있어요.
- 기억되는 게 싫다면 `/jiamemory optout`으로 거부할 수 있어요. 거부한 사용자는 대화 기록과 프로필 저장에서 제외돼요.
- 음성 채널이 한동안 조용하면 지아가 먼저 말을 걸어볼 수도 있어요. (기본은 꺼져 있고, `proactive_idle_sec` 설정으로 켤 수 있어요.)
- 사운드보드 효과음을 대화 상황에 맞춰 직접 고르거나, 설정에 따라 자동으로 짧게 반응하게 할 수 있어요.
- 음성 채널에서 유튜브 음악이나 재생목록을 재생하고, 지아가 말할 때는 음악 볼륨을 자동으로 낮출 수 있어요.
- (선택 기능) 로컬에서 ComfyUI를 사용하고 있다면, 지아에게 그림을 그려달라고 할 수 있어요! 상황별 모델 프로필을 등록해두면 그림 종류에 맞는 모델을 골라 사용할 수 있어요. (`settings.toml`의 `[comfyui]` 항목을 설정해야 해요. 설정하지 않으면 이 기능은 완전히 비활성화돼요.)
- 이 모든걸 로컬 환경에서 작동할 수 있어요. 외부로의 데이터 이동이 없어 유출 걱정없이 사용할 수 있어요.
- (선택) 원한다면 OpenAI·Anthropic 같은 외부 LLM API나 Ollama Cloud를 대신 사용할 수도 있어요. (`settings.toml`의 `[llm]` 항목을 바꾸면 돼요. 다만 이 경우 대화 내용이 외부로 전송돼요.)

<details>
  
  <summary>어떻게 이게 가능한가요?</summary>
  
  ### 음성 대화의 경우:
  
  1. ~~discord-ext-voice-recv 라이브러리를~~ 자체 음성 수신 모듈(discord_interface/voice_receive.py)을 사용하여 디스코드 음성을 실시간으로 입력받아요. (discord.py 내장 DAVE E2EE 세션을 활용해 종단간 암호화된 음성도 복호화할 수 있어요.) (2026년 3월 Discord의 A/V E2EE(DAVE) 프로토콜 사용 강제 업데이트로 인해 이전 라이브러리를 제거했어요.)
  2. Silero-VAD를 사용하여 실제로 말한 내용인지 확인해요. (여기서 지연이 너무 많이 발생하는 문제가 생겨, 어쩔 수 없이 마이크가 계속 열려있는 분은 Silero-VAD 단계를 넘어갈 수 없어요. :( )
  3. Silero-VAD가 실제로 말했다고 생각되는 부분을 적당한 여유를 남겨 잘라내요.
  4. ~~Whisper~~ Faster Whisper 라이브러리를 이용해 말한 내용을 텍스트로 변환해요. (1.1.0 업데이트에서 라이브러리가 변경되었어요. Faster Whisper를 사용하면 훨씬 더 빠른 결과를 받을 수 있기 때문이에요.)
  5. 변환된 텍스트는 LLM의 판단에 따라 RAG를 사용할지 말지를 결정해요. (1.0.1 업데이트에서 추가되었어요.)
  6. RAG를 사용하기로 결정됐다면 먼저 Faiss라는 유사도 검색 벡터에 넣어 현재 입력된 텍스트와 가장 유사한 기억을 최대 3개까지 불러와요.
  7. 유사한 기억과 변환된 텍스트를 LangGraph에 넣어 LLM이 응답을 생성하도록 만들어요.
  8. LLM이 필요에 따라 MCP 서버를 사용할 수도 있어요. ~~(지연이 너무 발생하지 않을까 걱정되어, 텍스트 채팅에 비해 적은 기능을 가지고 있어요.)~~ (1.1.0 업데이트에서 MCP를 사용할 수 있도록 했어요.)
  9. ~~LLM이 응답을 생성해내면 g2pk 라이브러리가 LLM의 응답을 발음하기 편한 형태로 바꿔줘요.~~ (1.0.1 업데이트에서 삭제되었어요. g2pk 라이브러리가 포함되면 오히려 더 부자연스러운 음성을 생성했기 때문이에요.)
  10. 생성된 응답을 . , ! ? 을 기준으로 실시간으로 분리해요. (예시: 안녕! 나는 지아야. 만나서 반가워. 오늘은 어때? -> 안녕 / 나는 지아야 / 만나서 반가워 / 오늘은 어때) (1.1.0 업데이트에서 추가되었어요. 음성을 실시간으로 생성하여 지연시간을 최대한 줄일 수 있기 때문이에요.)
  11. 분리된 문장은 ESPnet2가 다운로드된 TTS 모델을 이용해 실시간으로 대화 음성을 만들어내요. (1.1.0 업데이트에서 추가되었어요. 음성을 실시간으로 생성할 수 있도록 코드를 수정했어요.)
  12. discord.py를 이용해 만들어진 대화 음성을 출력해요.
  13. 말했던 내용과 LLM의 응답을 합하여 LLM이 어떤 대화를 했는지 요약하도록 해요.
  14. 요약된 내용을 Faiss 벡터에 추가하고 DB에도 저장해요.
  
  <img alt="image" src=".\voice_process.png" />
  
  (직접 설계한 구조에요. 음성 출력 이후 LLM을 이용한 요약 저장 아이디어는 음성 출력 시간과 사용자가 다시 말을 거는 시간을 활용할 수 있다는 점에서 착안되었어요.)
  
  ### 텍스트 채팅의 경우:
  
  1. discord.py가 슬래쉬 커멘드로 입력된 내용을 받아요.
  2. 입력된 내용은 LLM의 판단에 따라 RAG를 사용할지 말지를 결정해요. (1.0.1 업데이트에서 추가되었어요.)
  3. RAG를 사용하기로 결정됐다면 먼저 Faiss라는 유사도 검색 벡터에 넣어 입력된 텍스트와 가장 유사한 기억을 최대 3개까지 불러와요.
  4. 유사한 기억과 입력된 텍스트를 LangGraph에 넣어 LLM이 응답을 생성하도록 만들어요.
  5. LLM이 필요에 따라 MCP 서버를 사용할 수 있어요. (기본적으로 DuckDuckGoSearch MCP 서버가 연결되어 있어요.)
  6. LLM이 응답을 생성해내면 discord.py가 응답을 채팅으로 출력해요.
   
</details>


# 1. 설치

### 1-1. 설치하기전에 필요한 것이 있어요.

1. Ollama
2. KSS기반 TTS 모델 (ESPnet에서 사용할 수 있는지 확인해야 해요.)
3. 디스코드 봇 토큰
4. CUDA Toolkit 12.6

## 1-1-1. Ollama

ollama는 [여기](https://ollama.com/) 에서 다운로드 받을 수 있어요.

## 1-1-2. Vit기반 TTS 모델

[Hugging Face](https://huggingface.co/)에서 'vit ko'로 검색하면 모델이 몇 가지 나오는데 여기에서 ESPnet 예제가 있는지 확인해야해요. ESPnet 예제가 있다면 다운로드하여 압축 파일(.zip)형태로 준비해주세요.

## 1-1-3. 디스코드 봇 토큰

디스코드 봇 토큰은 [여기](https://discord.com/developers/applications)에서 생성할 수 있어요. 자세한 내용은 인터넷을 검색하는걸 추천드려요.

## 1-1-4. CUDA Toolkit 12.6

> [!important]
> CUDA Toolkit은 반드시 12.6.x 버전으로 설치해주세요.

CUDA Toolkit은 [여기](https://developer.nvidia.com/cuda-toolkit-archive)에서 다운로드 받을 수 있어요.

(Embedded Python 버전이 CUDA Toolkit 12.6 기준으로 라이브러리를 다운로드하기 때문이에요.)

# 1-2. Embedded python 버전으로 받기

1. [여기](https://github.com/Trace-0/Project_Jia/releases/tag/v1.1.2)에서 파일을 다운로드 받아요.
2. 받은 파일의 압축을 해제해주세요.
3. TTS 모델을 파일 안으로 옮겨주세요.
4. 'settings.toml' 파일을 열고 봇 토큰 등 설정을 입력해주세요. (파일이 없다면 한 번 실행하면 자동으로 생성돼요.)
5. 'RUN_FIRST.bat' 파일을 실행해주세요.
6. 'RUN_SECOND.bat' 파일을 실행해주세요. 관리자 권한을 요구하는 창이 나온다면 '예'를 선택해주세요. (ESPnet, pywin32의 문제로 TTS모델이 변경되면 관리자 권한이 필요합니다.)
7. (다음부터는 'start.bat' 파일로 실행할 수 있어요. 하지만, TTS 모델이 바뀐다면 'RUN_SECOND.bat' 파일로 한 번은 실행해야해요.)
8. 이제 지아와 대화해보죠!

* * *

# 2. 지아의 커맨드

| 커맨드 | 설명 |
|---|---|
| `/jia [내용]` , `/지아 [내용]` | 이 커맨드 뒤에 적히는 내용은 지아가 대답해줘요! 예) /지아 안녕? |
| `/jiajoin` | 지아가 통화방에 들어와요! |
| `/jialeave` | 지아가 통화방에서 나가요. :( |
| `/jiareload` | 지아의 설정을 다시 불러와요. 봇 소유자/화이트리스트 전용이에요. |
| `/jiasavesetting` | 현재 지아의 설정을 `settings.toml`에 저장해요. 봇 소유자/화이트리스트 전용이에요. |
| `/jiaunloadmodel` | 현재 로드된 LLM 모델을 언로드해요. 봇 소유자/화이트리스트 전용이에요. |
| `/jiaping` | 지아가 "pong!" 메시지를 보내요. |
| `/jiasay [문장]` | 지아가 [문장] 부분을 읽어줘요. |
| `/jiahear (채널ID)` | 지아가 듣는 내용을 (채널ID)에 적어줘요. (채널ID)를 입력하지 않으면 debug text channel에 적어줘요. 서버 관리자/화이트리스트 전용이에요. |
| `/jiastop` | 현재 지아가 재생중인 음성을 멈춰요. |
| `/jiaplay <로컬 파일 경로>` | 봇 PC의 로컬 오디오 파일을 재생해요. 보안상 위험해서 기본 비활성화되어 있고, 설정을 켠 뒤에도 봇 소유자/화이트리스트 전용이에요. |
| `/jiajoinnoagent` | 지아가 통화방에 들어와요. 하지만, 아무 기능도 작동하지 않아요. |
| `/jiatalk` | 해당 채널에서 작성하는 모든 대화는 지아가 대답해줘요. 이제 `/jia`나 `/지아`를 입력하지 않아도 괜찮아요. 서버 관리자/화이트리스트 전용이에요. |
| `/jiastoptalk` | `/jiatalk` 기능을 멈춰요. 서버 관리자/화이트리스트 전용이에요. |
| `/jiamusic play <유튜브 URL/재생목록/검색어>` | 기존 대기열을 바꾸고 유튜브 음악을 재생해요. 지아가 말할 때는 자동으로 음악 볼륨이 낮아져요. |
| `/jiamusic queue <유튜브 URL/재생목록/검색어>` | 현재 음악 대기열 뒤에 곡이나 재생목록을 추가해요. |
| `/jiamusic stop/skip/pause/resume/status` | 배경 음악을 멈추거나 다음 곡으로 넘기고, 일시정지/재개/상태 확인을 해요. |
| `/jiamusic volume <0.0~1.0>` | 배경 음악 볼륨을 조절해요. |
| `/jiarestart` | 지아가 재시작돼요. 재시작이 필요한 설정을 바꿨을 때 사용해요. 봇 소유자/화이트리스트 전용이에요. |
| `/jiamemory list (페이지)` | 이 서버에 저장된 기억을 최신순으로 보여줘요. 서버 관리자/화이트리스트 전용이에요. |
| `/jiamemory search [검색어]` | 저장된 기억을 유사도로 검색해요. 서버 관리자/화이트리스트 전용이에요. |
| `/jiamemory delete [ID]` | 해당 ID로 시작하는 기억을 삭제해요. (ID는 list/search에서 확인) 서버 관리자/화이트리스트 전용이에요. |
| `/jiamemory profile (이름)` | 지아가 그 사용자에 대해 기억하고 있는 사실을 보여줘요. 서버 관리자/화이트리스트 전용이에요. |
| `/jiamemory optout` | 지아가 나에 대한 대화와 정보를 기억하지 않게 해요. 기존 프로필과 단독 대화 기억도 함께 삭제돼요. |
| `/jiamemory optin` | 기억 기능을 다시 켜요. |
| `/jiamemory status` | 내 기억 설정 상태와 내 프로필 내용을 확인해요. |


# 2-1. 설정(settings.toml) 항목

지아의 모든 설정(봇 토큰 포함)은 프로젝트 루트의 `settings.toml` 파일에서 조절할 수 있어요. 파일이 없다면 최초 실행 때 기본값으로 자동 생성되고, 항목마다 설명 주석이 함께 적혀 있어요.

기존에 `.env`로 설정을 관리하던 경우, 실행하면 `JIA_` 설정값이 자동으로 `settings.toml`로 옮겨지고 원본은 `.env.bak`에 백업돼요.

`settings.toml`을 수정하고 저장하면 지아가 자동으로 변경을 감지해서 **재시작 없이** 반영해요. 모델 관련 설정이 바뀌면 해당 모델만 자동으로 다시 로드돼요. (수동으로 다시 불러오고 싶다면 `/jiareload`를 사용할 수 있어요.)

| 키 | 기본값 | 설명 | 반영 방식 |
|---|---|---|---|
| `[bot]` `token` | (없음) | 디스코드 봇 토큰 | 재시작 필요 |
| `[bot]` `debug_text_channel` | `0` | 로그를 보낼 디버그 텍스트 채널 ID (0이면 사용 안 함) | 재시작 필요 |
| `[bot]` `join_reply` | `true` | 음성 채널 접속 시 안내 메시지 전송 여부 | 즉시 |
| `[bot]` `leave_reply` | `true` | 음성 채널 퇴장 시 안내 메시지 전송 여부 | 즉시 |
| `[voice]` `timeout_sec` | `0.1` | 마지막 음성 패킷 이후 이 시간 동안 패킷이 없으면 발화가 끝났다고 판단해요 | 즉시 |
| `[voice]` `interrupt_speech_sec` | `0.5` | 지아가 말하는 중 사용자의 발화가 이 시간 이상 이어지면 재생을 중단해요(barge-in) | 즉시 |
| `[voice]` `proactive_idle_sec` | `0` | 음성 채널에서 이 시간(초) 동안 아무도 말이 없으면 지아가 먼저 말을 걸어봐요 (0이면 사용 안 함) | 즉시 |
| `[soundboard]` `auto_react` | `false` | 대화 상황에 맞는 효과음을 자동으로 재생할지 여부 | 즉시 |
| `[soundboard]` `auto_react_cooldown_sec` | `20` | 같은 효과음이 자동으로 다시 재생되기까지의 최소 간격 | 즉시 |
| `[soundboard]` `auto_react_chance` | `0.35` | 자동 반응 후보가 잡혔을 때 실제로 재생할 확률 | 즉시 |
| `[music]` `volume` | `0.7` | 배경 음악 기본 볼륨 | 즉시 |
| `[music]` `duck_volume` | `0.25` | 지아가 말하거나 효과음이 재생될 때 낮출 음악 볼륨 | 즉시 |
| `[music]` `max_playlist_items` | `50` | 유튜브 재생목록에서 한 번에 추가할 최대 곡 수 | 즉시 |
| `[security]` `command_whitelist_user_ids` | `[]` | 보호 명령 권한을 우회할 Discord 유저 ID 목록 | 즉시 |
| `[security]` `allow_unsafe_jiaplay` | `false` | 위험 기능인 `/jiaplay` 로컬 파일 재생을 허용할지 여부 | 즉시 |
| `[vad]` `threshold` | `0.7` | 발화로 판정할 확률 임계값 (0.0~1.0, 낮을수록 민감) | 즉시 |
| `[vad]` `min_speech_ms` | `150` | 이보다 짧은 발화 구간은 무시해요 | 즉시 |
| `[vad]` `min_silence_ms` | `1000` | 발화 구간 분리에 필요한 최소 무음 시간 | 즉시 |
| `[vad]` `max_speech_sec` | `30` | 발화 구간 하나의 최대 길이 | 즉시 |
| `[vad]` `padding_ms` | `200` | 발화 구간 시작 지점 앞에 붙이는 여유 시간 | 즉시 |
| `[whisper]` `model` | `turbo` | STT(음성 인식) 모델 | 모델 자동 재로딩 |
| `[whisper]` `device` | `cuda` | Whisper 실행 디바이스 | 모델 자동 재로딩 |
| `[whisper]` `compute_type` | `float16` | Whisper 연산 정밀도 | 모델 자동 재로딩 |
| `[whisper]` `beam_size` | `5` | STT beam size (클수록 정확하지만 느려요) | 즉시 |
| `[tts]` `model` | (없음) | TTS(음성 합성) 모델 경로 | 모델 자동 재로딩 |
| `[llm]` `model` | `gemma4:latest` | LLM 모델 이름. `provider`가 `ollama`면 Ollama 모델, 외부 API면 그 API의 모델 이름 | 모델 자동 재로딩 |
| `[llm]` `provider` | `ollama` | LLM 제공자. `ollama`(기본, 로컬) 또는 `openai`/`anthropic`/`google_genai`/`groq` 등. 아래 '외부 LLM API 사용' 항목 참고 | 모델 자동 재로딩 |
| `[llm]` `api_key` | (없음) | API 키. 외부 API 또는 Ollama Cloud 사용 시 입력. `provider=ollama`인데 키만 넣으면 Ollama Cloud로 연결돼요 | 모델 자동 재로딩 |
| `[llm]` `api_base` | (없음) | LLM 서버/API 주소 재정의 (선택). `ollama`면 원격 Ollama 서버 주소, 외부 API면 OpenAI 호환 서버 등 | 모델 자동 재로딩 |
| `[llm]` `num_ctx` | `16384` | LLM 컨텍스트 윈도우 크기(토큰). `provider`가 `ollama`일 때만 적용 | 모델 자동 재로딩 |
| `[llm]` `system_prompt` | (내장) | 지아의 성격/말투를 정의하는 시스템 프롬프트 | 모델 자동 재로딩 |
| `[llm]` `tools` | DuckDuckGo 검색 | 연결할 MCP 서버 목록. 아래 'MCP 서버 연결' 항목 참고 | 즉시 (자동 재연결) |
| `[llm]` `response_reserve_tokens` | `2048` | 컨텍스트 윈도우에서 응답 생성용으로 남겨둘 토큰 여유분 | 즉시 |
| `[rag]` `embedding_model` | `dragonkue/BGE-m3-ko` | 기억 검색용 임베딩 모델 | 재시작 필요 |
| `[rag]` `faiss_threshold` | `0.5` | 기억 검색 결과로 인정할 최소 유사도 점수 | 즉시 |
| `[rag]` `top_k` | `3` | 기억 검색 시 가져올 최대 개수 | 즉시 |
| `[rag]` `forgettable_importance` | `0.8` | 이 중요도 미만은 잊어버릴 수 있는 기억으로 저장해요 | 즉시 |
| `[rag]` `warn_importance` | `0.5` | 이 중요도 미만의 기억은 부정확할 수 있다는 경고와 함께 사용해요 | 즉시 |
| `[rag]` `save_importance_min` | `0.1` | 이 중요도 이하의 대화는 기억으로 저장하지 않아요 | 즉시 |
| `[rag]` `forget_decay_per_day` | `0.02` | 잊어버릴 수 있는 기억의 하루당 중요도 감쇠량 (0이면 망각 기능 끔) | 즉시 |
| `[rag]` `forget_threshold` | `0.15` | 감쇠된 중요도가 이 값 미만이 되면 기억을 삭제해요 | 즉시 |
| `[rag]` `retrieval_boost` | `0.05` | 기억이 검색에 사용될 때마다 중요도를 이만큼 올려요 (자주 쓰는 기억은 오래 유지) | 즉시 |
| `[rag]` `profile_max_facts` | `20` | 사용자별 프로필에 보관할 최대 사실 개수 (초과 시 오래된 것부터 삭제) | 즉시 |
| `[comfyui]` `url` | (없음) | ComfyUI 서버 주소 (예: `http://127.0.0.1:8188`). 비워두면 이미지 생성 기능을 사용하지 않아요 (선택 기능) | 즉시 |
| `[comfyui]` `checkpoint` | (없음) | 사용할 체크포인트 파일 이름 (ComfyUI의 `models/checkpoints` 안 파일명) | 즉시 |
| `[comfyui]` `steps` | `20` | 이미지 생성 스텝 수 (Flux Schnell은 4, 일반 SD 모델은 20~30 권장) | 즉시 |
| `[comfyui]` `cfg` | `7.0` | CFG 스케일 (Flux Schnell은 1.0, 일반 SD 모델은 7.0 권장) | 즉시 |
| `[comfyui]` `width` | `1024` | 생성 이미지 가로 크기 | 즉시 |
| `[comfyui]` `height` | `1024` | 생성 이미지 세로 크기 | 즉시 |
| `[comfyui]` `sampler` | `euler` | 샘플러 이름 | 즉시 |
| `[comfyui]` `scheduler` | `normal` | 스케줄러 이름 (Flux 계열은 `simple` 권장) | 즉시 |
| `[comfyui]` `negative_prompt` | (없음) | 네거티브 프롬프트 (Flux 계열은 비워둠) | 즉시 |
| `[comfyui]` `timeout_sec` | `120` | 이미지 생성 대기 제한 시간(초) | 즉시 |
| `[comfyui.models.<ID>]` `checkpoint` | (없음) | 상황별 이미지 생성 모델 프로필의 체크포인트 파일 이름 | `/jiareload` 권장 |
| `[comfyui.models.<ID>]` `use_when` / `tags` | (없음) | 지아가 이 모델을 언제 고를지 판단할 설명과 태그 | `/jiareload` 권장 |
| `[settings]` `watch_interval_sec` | `2.0` | `settings.toml` 변경 감지 주기(초) | 재시작 필요 |

> [!tip]
> 프로그램을 업데이트해서 새 설정 항목이 생기면, 다음 실행 때 기본값이 `settings.toml`에 자동으로 채워져요. 직접 적어둔 주석과 키 순서는 그대로 보존돼요.

> [!caution]
> `settings.toml`에는 디스코드 봇 토큰과 (사용하는 경우) 외부 LLM API 키가 들어 있으니 다른 사람과 공유하지 마세요.

## 외부 LLM API 사용

기본적으로 지아는 로컬 Ollama로 동작하지만, 원한다면 OpenAI·Anthropic 같은 외부 LLM API를 사용할 수도 있어요. `settings.toml`의 `[llm]` 항목에서 바꿀 수 있어요.

```toml
[llm]
provider = "openai"        # ollama(기본) / openai / anthropic / google_genai / groq 등
model = "gpt-4o-mini"      # 해당 제공자의 모델 이름
api_key = "sk-..."         # 외부 API 키
# api_base = "..."         # (선택) OpenAI 호환 서버 등 주소를 직접 지정할 때만
```

- 외부 제공자를 쓰려면 해당 langchain 통합 패키지를 먼저 설치해야 해요. 예: OpenAI는 `pip install langchain-openai`, Anthropic은 `pip install langchain-anthropic`. (설치가 안 되어 있으면 실행 시 어떤 패키지를 깔아야 하는지 안내해줘요.)
- `api_key`는 비워두고 환경 변수(`OPENAI_API_KEY` 등)로 넣어도 돼요.
- `provider`를 다시 `ollama`로 바꾸면 로컬 모델로 돌아가요. 이때는 `api_key`가 필요 없어요.

다른 컴퓨터나 다른 포트에서 돌아가는 **Ollama 서버**에 연결하고 싶다면, `provider`는 `ollama`로 둔 채 `api_base`에 주소만 적으면 돼요. (이 경우 `api_key`는 필요 없어요)

```toml
[llm]
provider = "ollama"
model = "gemma4:latest"
api_base = "http://192.168.0.10:11434"   # 원격 Ollama 서버 주소
```

[Ollama Cloud](https://docs.ollama.com/cloud)를 사용하면 강력한 로컬 GPU 없이도 큰 모델을 돌릴 수 있어요. `provider`는 `ollama`로 둔 채 `api_key`에 [Ollama API 키](https://ollama.com/settings/keys)를 넣으면 돼요. 주소는 자동으로 `https://ollama.com`에 연결돼요.

```toml
[llm]
provider = "ollama"
model = "gpt-oss:120b"   # Ollama Cloud 모델 이름
api_key = "..."          # https://ollama.com/settings/keys 에서 발급
```

> [!caution]
> 외부 LLM API나 Ollama Cloud를 사용하면 대화 내용이 해당 업체의 서버로 전송돼요. "모든 처리가 로컬에서 이루어진다"는 장점은 이 경우 적용되지 않으니, 데이터 유출이 걱정된다면 기본값인 로컬 `ollama`(키 없이)를 사용하세요.

## MCP 서버 연결

지아에게 MCP 서버를 연결해서 도구를 늘려줄 수 있어요. `settings.toml`에 `[llm.tools.서버이름]` 테이블을 추가하면 돼요. (형식은 langchain의 `MultiServerMCPClient`와 같아요)

```toml
# 기본으로 연결되는 DuckDuckGo 검색 MCP 서버 (stdio 방식)
[llm.tools.ddg-search]
command = "uvx"
args = ["duckduckgo-mcp-server"]
transport = "stdio"

# 원격/로컬 HTTP 서버를 연결하는 예시
[llm.tools.my-server]
url = "http://localhost:9000/mcp"
transport = "streamable_http"
```

- 저장하면 재시작 없이 자동으로 다시 연결돼요.
- MCP를 아예 사용하지 않으려면 `tools = {}`로 적어주세요.
- 서버 연결에 실패해도 지아의 대화는 정상 동작하고, 해당 도구만 빠져요.


# 2-2. ComfyUI 이미지 생성 모델 프로필

기본 설정처럼 `[comfyui] checkpoint` 하나만 적어도 이미지 생성은 동작해요. 이 경우 지아는 그 모델을 `default` 프로필로 사용합니다.

그림 종류에 따라 다른 체크포인트를 쓰고 싶다면 `[comfyui.models.<ID>]` 테이블을 추가해 주세요. 지아는 이미지 생성 도구를 사용할 때 `use_when`과 `tags`를 보고 상황에 맞는 `model_id`를 고릅니다. 애매하면 `default`나 첫 번째 모델을 사용해요.

```toml
[comfyui]
url = "http://127.0.0.1:8188"
checkpoint = "default_model.safetensors" # fallback default 모델
steps = 20
cfg = 7.0
width = 1024
height = 1024

[comfyui.models.illust]
checkpoint = "anime_illust.safetensors"
use_when = "캐릭터, 애니풍 일러스트, 감정 표현, 귀여운 장면"
tags = ["character", "anime", "illust"]
steps = 24
cfg = 7.0

[comfyui.models.photo]
checkpoint = "realistic_photo.safetensors"
use_when = "현실 사진, 음식, 장소, 물건, 제품처럼 사실적인 이미지"
tags = ["photo", "realistic", "food", "place", "product"]

[comfyui.models.meme]
checkpoint = "meme_style.safetensors"
use_when = "밈, 웃긴 상황, 과장된 리액션 이미지"
tags = ["meme", "funny", "reaction"]
```

프로필별로 `steps`, `cfg`, `width`, `height`, `sampler`, `scheduler`, `negative_prompt`를 따로 지정할 수 있어요. 생략한 값은 `[comfyui]`의 기본값을 따라갑니다.

> [!tip]
> 모델 프로필을 바꾼 뒤 지아가 새 모델 목록을 바로 인지하게 하려면 `/jiareload`를 실행해주세요. 이미지 생성 시점에는 설정 파일을 다시 읽지만, LLM 도구 설명의 모델 목록은 에이전트 재로딩 후 가장 정확해요.


# 2-3. 사운드보드

지아가 대화 상황에 어울리는 효과음을 직접 골라 음성 채널에서 재생할 수 있어요.

사용 방법은 간단해요:

1. 프로젝트 루트의 `soundboard` 폴더에 오디오 파일을 넣어주세요. (폴더가 없다면 한 번 실행하면 자동으로 생성돼요. mp3, wav, ogg, flac, m4a, opus, webm, aac 지원)
2. 지아를 실행하면 새 파일이 `soundboard/sounds.toml`에 자동으로 등록돼요.
3. `sounds.toml`을 열고 각 파일이 어떤 효과음인지 설명을 적어주세요. 지아는 이 설명을 읽고 어떤 상황에 어떤 효과음을 틀지 판단해요.

```toml
# soundboard/sounds.toml
"tada.mp3" = "축하하거나 무언가에 성공했을 때 쓰는 빰빠밤 팡파레 효과음"
"dog.wav" = "강아지가 멍멍 짖는 소리"
```

자동 반응을 더 세밀하게 조절하고 싶다면 아래처럼 효과음별 설정을 적을 수 있어요. 기존 문자열 형식도 계속 지원됩니다.

```toml
# settings.toml
[soundboard]
auto_react = true
auto_react_cooldown_sec = 20
auto_react_chance = 0.35

# soundboard/sounds.toml
"tada.mp3" = { desc = "축하하거나 성공했을 때 쓰는 팡파레", tags = ["success", "celebrate"], cooldown = 20, chance = 0.8 }
"fail.wav" = { desc = "실패하거나 아쉬운 상황의 효과음", tags = ["fail", "awkward"], cooldown = 30, chance = 0.5 }
"laugh.wav" = { desc = "웃긴 드립이나 농담에 쓰는 웃음 효과음", tags = ["laugh"], auto = true }
"secret.wav" = { desc = "수동으로만 쓰고 싶은 효과음", auto = false }
```

- `tags`: 자동 반응에 사용할 상황 태그예요. `success`, `celebrate`, `fail`, `awkward`, `laugh`, `surprise`, `sad`, `angry` 같은 태그를 인식해요.
- `cooldown`: 해당 효과음이 자동으로 다시 재생되기까지의 최소 간격이에요. 생략하면 `[soundboard] auto_react_cooldown_sec`를 따라가요.
- `chance`: 후보로 잡혔을 때 실제로 재생할 확률이에요. 생략하면 `[soundboard] auto_react_chance`를 따라가요.
- `auto = false`: 자동 반응에서는 제외하고, 지아가 직접 `play_soundboard` 도구를 쓸 때만 재생할 수 있게 해요.

> [!note]
> 보안을 위해 지아(LLM)는 `soundboard` 폴더 바로 아래에 있는 허용된 확장자의 오디오 파일만 재생할 수 있어요. 폴더 밖의 파일을 읽거나 재생하는 것은 차단돼요.

> [!tip]
> 파일을 새로 넣으면 다음 도구 호출 때 바로 재생할 수 있지만, 지아가 효과음 목록을 새로 인지하게 하려면 `/jiareload`를 한 번 실행해주는 것이 좋아요.


# 2-4. 보호 명령과 화이트리스트

일부 명령은 서버 전체 동작, 기억 조회/삭제, 로컬 파일 접근에 영향을 주므로 권한이 필요해요. `owner` 보호 명령은 Discord 봇 소유자 또는 `[security] command_whitelist_user_ids`에 등록된 유저만 사용할 수 있고, `admin` 보호 명령은 서버 관리자/서버 관리 권한자 또는 화이트리스트 유저가 사용할 수 있어요.

```toml
[security]
command_whitelist_user_ids = [123456789012345678]
```

- `owner` 보호 명령: `/jiareload`, `/jiasavesetting`, `/jiaunloadmodel`, `/jiarestart`, `/jiaplay`
- `admin` 보호 명령: `/jiahear`, `/jiatalk`, `/jiastoptalk`, `/jiamemory list`, `/jiamemory search`, `/jiamemory delete`, `/jiamemory profile`
- 일반 사용자 명령: `/jia`, `/지아`, `/jiajoin`, `/jialeave`, `/jiajoinnoagent`, `/jiaping`, `/jiasay`, `/jiastop`, `/jiamusic ...`, `/jiamemory optout`, `/jiamemory optin`, `/jiamemory status`

화이트리스트는 Discord 서버 권한과 무관하게 보호 명령을 허용하는 우회 목록이므로, 실제로 봇 운영을 맡길 사람의 ID만 등록하는 것을 권장해요. `/jiaplay`는 화이트리스트나 봇 소유자 권한이 있어도 `[security] allow_unsafe_jiaplay = true`를 켜야 실제로 재생됩니다.


# 2-5. 위험 기능: /jiaplay 로컬 파일 재생

`/jiaplay <로컬 파일 경로>`는 봇이 실행 중인 컴퓨터의 로컬 오디오 파일을 직접 읽어서 음성 채널에 재생하는 기능이에요. 이 기능은 편하지만 보안상 위험해서 기본값으로 꺼져 있습니다.

```toml
[security]
allow_unsafe_jiaplay = true
```

이 설정을 켜면 Discord에서 명령을 입력할 수 있는 사람이 봇 프로세스 권한으로 접근 가능한 로컬 파일 경로를 지정할 수 있어요. 오디오 파일에 사적인 내용이 들어 있으면 그 내용이 음성 채널로 그대로 나가고, 아주 큰 파일을 지정하면 파일 전체를 메모리에 읽으면서 봇이 느려지거나 중단될 수 있어요. 또한 신뢰할 수 없는 미디어 파일을 FFmpeg가 해석하게 되므로, 디코더 취약점이나 비정상 파일로 인한 오류 가능성도 생깁니다.

따라서 이 설정은 개인 서버나 완전히 신뢰하는 사용자만 명령을 쓸 수 있는 환경에서만 켜는 것을 권장해요. 일반적인 음악 재생은 `/jiamusic`의 유튜브 재생 기능을 사용하는 편이 더 안전합니다.


# 2-6. 배경 음악과 덕킹

`/jiamusic` 명령어로 음성 채널에 유튜브 음악을 틀 수 있어요. `yt-dlp`로 유튜브 단일 영상, 재생목록 URL, 검색어를 받아 재생 큐로 만들고, 각 곡을 재생할 때 실제 오디오 스트림 URL을 가져옵니다. 음악이 재생되는 동안 지아가 TTS로 대답하거나 사운드보드 효과음을 재생하면, 음악 볼륨이 자동으로 `[music] duck_volume`까지 낮아지고 foreground 재생이 끝나면 원래 볼륨으로 돌아갑니다.

```text
/jiamusic play https://www.youtube.com/watch?v=...
/jiamusic play https://www.youtube.com/playlist?list=...
/jiamusic play lofi hip hop radio
/jiamusic queue 신나는 게임 bgm
/jiamusic skip
/jiamusic volume 0.6
/jiamusic pause
/jiamusic resume
/jiamusic stop
/jiamusic status
```

음악과 지아의 목소리는 내부 오디오 믹서에서 함께 PCM으로 합쳐져 Discord에 전송돼요. 그래서 음악이 흐르는 중에도 대화 응답이 큐에 막히지 않고, 지아가 말할 때 음악이 작아지는 라디오 진행자 같은 동작을 할 수 있어요.

재생목록은 기본적으로 최대 50곡까지만 한 번에 추가해요. 더 길게 받고 싶다면 `settings.toml`의 `[music] max_playlist_items` 값을 바꿔주세요.

> [!note]
> `/jiastop`은 지아의 TTS와 효과음 같은 foreground 재생만 멈춰요. 배경 음악을 멈추려면 `/jiamusic stop`을 사용해주세요.


# 3. 현재 발견된 문제

마이크가 항상 열려있는 사용자는 지연 시간 문제로 음성을 처리하지 않는 문제

ESPnet의 문제로 관리자 권한을 요구하는 문제

pywin32의 문제로 관리자 권한을 요구하는 문제

간혹 opus 에러가 발생하는 문제


# 4. 변경 계획

- [ ] TTS, STT 라이브러리를 선택할 수 있도록 수정할 예정입니다.
- [x] (가능하다면) LLM을 외부 API를 사용할 수 있도록 수정할 예정입니다.
- [x] 설정과 관련된 수정을 할 예정입니다.
- [x] /jia 또는 /지아 없이 대화할 수 있는 명령어를 추가할 예정입니다.
- [x] 사운드보드를 추가할 예정입니다.
- [x] 다인 대화에 최적화된 모습을 보여줄 수 있도록 수정할 예정입니다.


# 5. 변경 내용

[음성]은 음성 대화에 영향을 주는 변경사항이며 [채팅]은 채팅 대화에 영향을 주는 변경사항입니다. 아무것도 적혀있지 않다면 공통으로 적용되는 변경사항입니다.

## 1.2.0 (beta)

이번 버전은 음성 대화를 다인(여러 명) 대화에 최적화하는 방향으로 수정되었습니다.

이제 여러 명이 함께 통화하는 자리에서도 지아가 대화의 흐름을 읽고 자연스럽게 끼어들거나 조용히 들을 수 있습니다.

1. [음성] 발화를 길드별로 모아 화자 이름 라벨("이름: 내용")과 함께 한 번의 LLM 호출로 처리하도록 변경했습니다.
2. [음성] 지아가 응답을 생성/재생하는 동안 들어온 발화는 버려지지 않고 다음 배치로 묶여 한꺼번에 처리됩니다. 발화마다 응답이 연달아 재생되는 일이 사라집니다.
3. [음성] LLM 호출과 응답 재생이 길드별 단일 워커로 직렬화되었습니다. 여러 명이 동시에 말할 때 대화 기록이 깨지거나 응답 음성이 뒤섞여 재생되던 문제가 해결됩니다.
4. [음성] 사람들끼리 대화하는 중이라 응답할 상황이 아니라고 LLM이 판단하면 지아가 말을 하지 않습니다.
5. [음성] 현재 음성 채널 참가자 목록(봇 제외)을 LLM 호출마다 함께 전달합니다. 지아가 발화가 누구를 향한 것인지 판단하는 근거로 쓰며, 1대1 상황에서는 응답 없이 침묵하는 일이 줄어듭니다.
6. [음성] 지아의 음성 재생 중에 사용자의 발화가 0.5초 이상 지속되면 재생을 중단합니다. 중단되었다는 사실은 다음 LLM 호출에 함께 전달되어 지아가 중단을 인지하고 자연스럽게 대화를 이어갑니다.
7. 대화 기억 구조를 최대 3쌍 제한에서 컨텍스트 윈도우 기반 유지로 변경했습니다. 이제 컨텍스트 윈도우 예산 안에서 기록을 최대한 유지하고, 한도를 넘으면 오래된 메시지부터 제거합니다.
8. LLM 컨텍스트 윈도우 크기를 설정 항목으로 추가했습니다. (settings.toml의 `[llm] num_ctx`, 기본값 16384) GPU 메모리에 여유가 있다면 이 값을 올려 더 많은 대화를 기억하게 할 수 있습니다.
9. [음성] `/jiastop`, `/jialeave`가 아직 응답 생성이 시작되지 않은 대기 발화도 함께 정리하도록 변경했습니다.
10. 설정 파일을 `.env`에서 `settings.toml`로 변경했습니다. 지아의 모든 설정(봇 토큰 포함)은 항목별 설명 주석과 함께 `settings.toml`에서 관리되며, `.env`는 LangSmith 같은 개발용 키 전용이 되어 일반 사용자는 더 이상 다룰 필요가 없습니다. 기존 `.env`의 `JIA_` 설정값은 최초 실행 시 자동으로 `settings.toml`로 옮겨집니다. (원본은 `.env.bak`에 백업)
11. 코드 곳곳에 하드코딩되어 있던 세부 튜닝값을 모두 설정 항목으로 옮겼습니다. (발화 종료 대기 시간, barge-in 판정 시간, VAD 파라미터, Whisper 디바이스/빔 크기, RAG 중요도 임계값/검색 개수, 임베딩 모델, LLM 응답 여유 토큰 등. 전체 목록은 [2-1. 설정(settings.toml) 항목](#2-1-설정settingstoml-항목) 참고)
12. `settings.toml` 파일 변경을 자동으로 감지해 재시작 없이 설정을 반영합니다. 모델 관련 설정이 바뀌면 해당 모델만 자동으로 다시 로드되고, 이전 LLM 모델은 Ollama 메모리에서 내려갑니다.
13. `/jiarestart` 명령어를 추가했습니다. 재시작이 필요한 설정(`[rag] embedding_model` 등)을 바꿨을 때 사용할 수 있습니다.
14. 프로그램 업데이트로 새 설정 항목이 추가되면 다음 실행 때 기본값이 `settings.toml`에 자동으로 채워집니다.
15. 기억을 사람처럼 잊는 망각 시스템을 추가했습니다. 중요도가 낮은 기억은 마지막으로 사용된 뒤 시간이 지날수록 흐려지다가 기준 미만이 되면 삭제되고, 검색에 실제로 쓰인 기억은 중요도가 조금씩 올라가 더 오래 유지됩니다. (감쇠량/삭제 기준/보정량은 `[rag]` 설정으로 조절 가능)
16. `/jiamemory` 명령어를 추가했습니다. 서버에 저장된 기억을 목록으로 보거나(`list`) 검색하고(`search`) 삭제할 수 있으며(`delete`), 사용자별 프로필 조회(`profile`)와 기억 설정 상태 확인(`status`)도 가능합니다. 현재 `profile`은 서버 관리자 전용이고, 일반 사용자는 `status`로 본인 프로필을 확인합니다.
17. 대화에서 알게 된 사용자에 대한 장기적인 사실(취향, 관계 등)을 사람별 프로필로 기억하고, 이후 대화에 자동으로 참고합니다. 대화 요약 과정에서 함께 추출되어 추가 LLM 호출 없이 저장됩니다.
18. 기억 사용을 거부할 수 있는 기능을 추가했습니다. `/jiamemory optout`을 입력한 사용자는 이후 대화 저장과 프로필 추출에서 제외되며, 기존 프로필과 단독 대화 기억도 삭제됩니다. (`optin`으로 다시 켤 수 있습니다)
19. [음성] 먼저 말 걸기 기능을 추가했습니다. 음성 채널이 일정 시간(`[voice] proactive_idle_sec`, 기본 0=꺼짐) 동안 조용하면 지아가 이전 대화나 기억을 활용해 먼저 말을 걸어봅니다. 기존 발화 처리 구조를 그대로 따르므로 실제 대화와 충돌하지 않습니다.
20. 로컬 ComfyUI를 사용한 이미지 생성 기능을 추가했습니다. (선택 기능) `settings.toml`의 `[comfyui] url`과 `checkpoint`를 설정하면 지아에게 그림을 그려달라고 할 수 있고, 설정하지 않으면 기능이 완전히 비활성화되어 일반 사용자에게는 영향이 없습니다.
21. 이미지 생성처럼 시간이 걸리는 작업은 먼저 대기 안내 메시지를 보낸 뒤, 완성되면 그 메시지를 수정해 이미지를 첨부하도록 했습니다. 생성이 진행 중인지 확인할 수 있습니다.
22. 시스템 프롬프트를 정리하고, 사용자를 "야"·"님" 같은 호칭 대신 발화에 표시된 사용자 이름으로 부르도록 규칙을 추가했습니다.
23. 시간이 오래 걸리는 도구(인터넷 검색 등)를 사용할 때, 먼저 기다려 달라는 안내를 전달한 뒤 결과를 이어서 전달하도록 했습니다. ([음성]은 안내 음성으로, [채팅]은 안내 메시지로) 즉시 끝나는 도구에는 적용되지 않습니다.
24. 연결할 MCP 서버를 `settings.toml`의 `[llm.tools]`에서 직접 지정할 수 있도록 했습니다. 저장하면 재시작 없이 자동으로 다시 연결되며, 자세한 형식은 [MCP 서버 연결](#mcp-서버-연결) 항목을 참고하세요.
25. [음성] 음성 대화도 텍스트와 동일하게 MCP 도구를 사용하도록 통일하되, 지연을 줄이기 위해 음성에서는 정말 필요할 때만 도구를 쓰도록 안내합니다. (잡담이나 이미 아는 내용은 도구 없이 바로 응답하고, 최신 정보나 모르는 사실을 물을 때만 검색)
26. LLM을 로컬 Ollama 대신 외부 API(OpenAI, Anthropic, Google 등)로도 사용할 수 있게 했습니다. `settings.toml`의 `[llm] provider`/`api_key`/`api_base`로 설정하며, 기본값은 로컬 `ollama`라 기존 사용자에게는 영향이 없습니다. 자세한 방법은 [외부 LLM API 사용](#외부-llm-api-사용) 항목을 참고하세요. (단, 외부 API 사용 시 대화 내용이 외부로 전송됩니다)
27. `provider`를 `ollama`로 둔 채 `[llm] api_base`에 주소를 적으면, 다른 컴퓨터나 다른 포트에서 돌아가는 원격 Ollama 서버에도 연결할 수 있습니다. 모델 로드/언로드도 해당 서버를 향합니다.
28. Ollama Cloud(https://ollama.com)를 지원합니다. `provider`를 `ollama`로 둔 채 `[llm] api_key`에 Ollama API 키만 넣으면 자동으로 클라우드(`https://ollama.com`)에 Bearer 인증으로 연결되어, 강력한 로컬 GPU 없이도 큰 모델을 쓸 수 있습니다. (이 경우 모델 로드/언로드는 클라우드가 관리하므로 건너뜁니다)
29. 사운드보드 자동 반응 기능을 추가했습니다. `[soundboard] auto_react`를 켜면 대화 문맥과 `sounds.toml`의 태그를 바탕으로 효과음을 짧게 자동 재생하며, 효과음별 `cooldown`, `chance`, `auto` 설정으로 과한 재생을 막을 수 있습니다.
30. ComfyUI 이미지 생성에서 상황별 모델 프로필을 지원합니다. `[comfyui.models.<ID>]`에 체크포인트와 `use_when`/`tags`를 등록하면 지아가 이미지 종류에 맞는 `model_id`를 선택해 생성합니다.
31. [음성] 배경 음악 재생 명령어 `/jiamusic`를 추가했습니다. stop/pause/resume/volume/status 제어를 지원합니다.
32. [음성] 오디오 믹서를 추가해 배경 음악과 TTS/효과음을 함께 출력합니다. 지아가 말하거나 효과음을 재생하는 동안 음악 볼륨을 `[music] duck_volume`으로 낮추고, 끝나면 기존 음악 볼륨으로 복구합니다.
33. [음성] `/jiamusic`가 로컬 파일 직접 재생 대신 `yt-dlp` 기반 유튜브 재생을 사용합니다. 유튜브 단일 영상, 검색어, 재생목록을 지원하고 `queue`/`skip`으로 대기열을 제어할 수 있습니다.
34. [보안] `/jiaplay` 로컬 파일 재생은 기본 비활성화했습니다. Discord 명령으로 봇 PC의 로컬 파일을 읽어 음성 채널에 내보낼 수 있는 위험 기능이므로 `[security] allow_unsafe_jiaplay = true`를 명시적으로 설정한 경우에만 작동합니다.
35. [보안] `/jiamemory profile`은 서버 관리자 전용으로 변경하고, 일반 사용자는 `/jiamemory status`에서 본인의 기억 설정과 프로필 내용을 함께 확인하도록 변경했습니다.
36. [보안] 보호 명령 권한 체계를 추가했습니다. owner/admin 보호 명령은 권한을 확인하고, `[security] command_whitelist_user_ids`에 등록된 Discord 유저는 서버 권한과 무관하게 보호 명령을 사용할 수 있습니다.

---

<details>

  <summary>이전 버전의 변경사항</summary>
  
  ## 1.1.2

  이번 버전은 모델 로드/언로드와 음성·텍스트 파이프라인에서 불필요한 작업을 제거하는 리팩토링이 중점적으로 이루어진 수정입니다.

  아래의 수정을 통해 음성 인식이나 모델 로드 중에도 봇이 멈추지 않으며, 숨어있던 버그도 여럿 수정되었습니다.

  1. Ollama 모델 로드/언로드를 실제 토큰 생성 없이 빈 요청(keep_alive 설정)만으로 처리하도록 변경했습니다. 특히 이미 내려가 있는 모델을 언로드할 때 모델을 다시 로드하고 추론까지 하던 낭비가 제거되었습니다.
  2. [음성] 모델 로드/언로드를 백그라운드 스레드에서 실행하도록 변경했습니다. 이제 `/jiajoin`의 모델 로드 중에도 봇이 다른 명령에 응답할 수 있습니다.
  3. 다른 채널이나 음성 연결에서 LLM을 사용 중이면 언로드를 건너뛰도록 변경했습니다. (`/jiastoptalk`, 음성 연결 종료 시)
  4. [채팅] `/jia`, `/지아`의 응답이 내부 오류(인자 불일치)로 전송되지 않던 문제를 수정했습니다.
  5. [음성] 음성 인식(리샘플링/VAD/Whisper)이 봇 이벤트 루프를 블로킹하던 문제를 수정했습니다. 이제 음성 인식 중에도 봇이 멈추지 않습니다.
  6. [음성] TTS 생성에서 짧은 문장이 긴 문장을 추월해 순서가 뒤바뀐 채 재생될 수 있던 문제가 수정되었습니다.
  7. [음성] `/jiastop`이 음성 수신까지 영구히 중단시키던 문제를 수정했습니다.
  8. [음성] `/jiastop`이 대기 중인 오디오 큐를 비우고 진행 중인 TTS 생성에도 취소 신호를 보내도록 변경했습니다.
  9. [음성] 발화 종료 감지의 비효율적인 부분을 효율적인 코드로 변경하였습니다. 다만, 감지 타이밍은 같습니다.
  10. [음성] `/jiasay`가 음성 생성 파이프라인 코드를 재사용합니다. 문장 전체를 음성으로 생성하지 않고 적절히 끊어 생성하므로 매우 빠른 속도로 음성 재생이 시작됩니다.
  11. [채팅] 텍스트 응답 채널 매핑이 전송 후에도 계속 쌓이던 메모리 누수를 수정했습니다.
  12. 과거 FastAPI 구조의 흔적이던 불필요한 task_id 전달 경로를 정리했습니다.
  13. `/jiareload` 명령어를 사용해도 일부 설정이 반영되지 않던 문제를 수정했습니다.


  ## 1.1.1

  이번 버전은 Discord의 DAVE 프로토콜 사용 강제에 대응이 중점적으로 이루어진 수정입니다.

  아래의 수정을 통해 음성 연결을 정상적으로 사용할 수 있으며 다른 세세한 수정도 포함되었습니다.

  1. [음성] discord.ext.voice.recv 라이브러리를 제거하였습니다.
  2. [음성] 현재 코드에 대응되며 DAVE 암호화 프로토콜에 대응되는 신규 모듈을 작성하였습니다.
  3. 기본 LLM 모델을 `Gemma4:latest`로 변경하였습니다.
  4. Source 폴더 안에 있던 코드를 루트 위치로 평탄화 했습니다.
  5. 코드 내부의 Source 폴더 관련 하드코딩된 부분을 수정했습니다.
  6. 설정 파일을 `.env` 파일이 대신할 수 있도록 위임했습니다.


  ## 1.1.0

  이번 버전은 최대한 응답의 지연을 줄이기 위한 방향으로 수정되었습니다.

  아래의 수정을 통해 체감할 수 있을 정도로 빠르게 응답을 받을 수 있습니다.

  1. 개발자의 비동기 이해도가 떨어져 이전 작성된 버전에 FastAPI가 사용되었지만, 현재 버전부터 FastAPI를 사용한 부분을 전부 제거하였습니다.
  2. [음성] 입력 감지 기간을 0.3초에서 0.1초로 변경하였습니다. 0.1초간 음성이 입력되지 않으면 말이 끝났다고 판단하고 처리를 시작합니다.
  3. 단어 사전 기능을 삭제했습니다. 단어 사전 기능은 괜찮은 기능으로 보였지만, LLM이 사전/대화기록을 선택적으로 접근하게 되면서 용도가 불분명해졌기 때문입니다. 이후 업데이트로 재등장할 가능성은 있지만, 1.0.0 버전만큼 중요성이 높지 않을것으로 예상됩니다.
  4. [음성] STT(ASR) 라이브러리를 Transformers(Whisper)에서 Faster Whisper로 변경했습니다.
  5. [음성] LLM -> TTS 과정을 비동기로 만들어 LLM이 응답을 생성하는 과정에도 자연스러운 TTS를 실시간으로 만들어 낼 수 있도록 로직을 변경했습니다.
  6. 최근 디스코드 메시지를 최대 10개까지 불러올 수 있도록 LLM Tool에 추가했습니다. 이는 이미지도 포함됩니다. (이미지를 인식하고 대화를 나누기 위해선 **반드시** VLM 모델로 변경해야 합니다.)
  7. LLM의 기억 구조를 변경했습니다. 이제 LLM은 최대 3쌍(사용자 3 + AI 3)의 대화를 기억합니다.(이전에는 무제한으로 저장했습니다. 이는 맥락 오염 방지를 위한 설계이기도 하지만, VRAM 절약에도 도움이 됩니다. :) ) 이로써 LLM은 RAG의 대화 기억 기능을 자주 활용하게 됩니다.
  8. 대화 요약 저장 과정을 1.0.0 버전으로 롤백하고, 이 코드를 수정하여 더욱 안전하고 정확하고 예상할 수 있는 결과를 받을 수 있도록 변경하였습니다.
  9. [음성] LLM 모델을 항상 메모리에 로드시켜두어 LLM 모델이 다시 로드되며 발생하는 지연을 제거했습니다.
  10. [음성] 생성된 TTS 음성을 디스크에 저장하지 않고 메모리에 잠시 저장하도록 변경하여 IO 지연을 제거했습니다.
  11. [음성] RAG의 DB를 음성 대화 중에는 메모리에 로드해두도록 변경하여 IO 지연을 제거했습니다.
  12. [채팅] 이전 변경 계획이었던 `/jiatalk` 명령어를 추가했습니다.
  13. 친구들과 놀기 위해 수정한 코드도 이번 업데이트에 적용되었습니다. (이번 업데이트로 추가된 명령어 대부분은 이렇게 만들어졌습니다. :) )


  ## 1.0.1

  대화를 저장하고 과거의 대화를 불러오는 과정을 LLM이 스스로 판단하여 결정할 수 있도록 변경하였습니다.

  해당 변경으로 기억 관련된 문제 대부분이 해결되었으며 도구를 사용하지 않을때의 응답속도가 개선되었습니다. (RTX 3090으로 5초 정도의 지연만 발생했습니다. :) )

  음성 대화 부분을 langgraph로 전환했습니다.

  fastAPI의 비효율적인 호출을 일부 제거하여 미약하지만 응답속도가 개선되었습니다.

  프롬프트에 도구를 명시하여 직접적으로 말하지 않아도 도구를 사용할 수 있도록 하였습니다.

</details>
