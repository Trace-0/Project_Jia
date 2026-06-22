from config.config_manager import config
from LLM.LLM_model_control import create_chat_model, reset_cached_chat_models
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import InMemorySaver
from langchain_core.messages import AIMessage, AIMessageChunk, HumanMessage, trim_messages
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.tools import Tool
from memory.RAG import RAG, get_rag_instance
import logging
from datetime import datetime
from LLM.langchain_tools.mcp_manager import get_client, get_mcp_usage_guidance
from memory.calculate_importance import calculate_and_save_importance
from LLM.langchain_tools.load_discord_message import load_discord_message_tool
from LLM.langchain_tools.soundboard import get_sound_registry_version, load_sound_registry, soundboard_tool
from LLM.langchain_tools.comfyui_image import comfyui_image_tool, is_comfyui_enabled
import asyncio
import re
import threading
from dotenv import load_dotenv

load_dotenv()

llm = create_chat_model()
checkpointer = InMemorySaver()

def get_memory_manager(guild_id: int) -> RAG:
    # 길드별 RAG 인스턴스는 RAG.py의 전역 캐시 하나만 사용합니다.
    # (별도 캐시를 두면 인메모리 DB가 갈라져서 저장한 기억/프로필/거부 설정이 서로 보이지 않음)
    return get_rag_instance(guild_id)

def create_rag_tool_for_guild(guild_id: int):
    memory_manager = get_memory_manager(guild_id)
    
    def _retrieve_and_inform(query: str) -> str:
        logging.info(f"[LLM:Tool] 기억에서 검색중 : {query}")
        retrieved_context = memory_manager.get_context(user_input=query)
        if not retrieved_context:
            return "관련된 과거 대화 내용을 찾지 못했습니다. 추가적인 검색은 필요하지 않습니다. 사용자의 질문에 답변하세요."
        
        return (
            "검색된 과거 대화 내용은 다음과 같습니다:\n"
            f"'''\n{retrieved_context}\n'''\n"
            "이제 이 정보를 사용하여 사용자의 질문에 답변하세요. 추가적인 검색은 필요하지 않습니다."
        )
    
    retrieve_tool = Tool(
        name="Conversation_Memory_Search",
        func=_retrieve_and_inform,
        description="현재 대화와 관련된 과거 대화 내용을 검색합니다. 사용자가 이전에 했던 말을 기억해야 할 때 사용하세요. 이 도구는 한국어 단어를 입력하면 더 정확한 결과를 얻을 수 있습니다."
    )
    return [retrieve_tool]

def time_tool():
    def _get_current_time(query: str = "") -> str:
        """현재 시간을 ISO 8601 형식으로 반환합니다. 이 함수는 입력을 무시합니다."""
        return datetime.now().isoformat()
    
    return Tool(
        name="Current_Time",
        func=_get_current_time,
        description="현재 시간을 ISO 8601 형식으로 반환합니다. 시간 관련 정보가 필요하거나 사용자가 현재 시간을 물어볼 때 사용하세요."
    )

# 음성 대화에서 외부 도구(MCP 등) 사용을 최대한 자제시키는 안내.
# 음성은 지연에 민감하므로, 도구 호출로 응답이 느려지지 않도록 정말 필요할 때만 쓰게 합니다.
VOICE_TOOL_RESTRAINT = """

지금은 실시간 음성 대화 상황이야. 도구를 호출하면 응답이 느려져서 대화의 흐름이 끊기니, 도구는 정말 꼭 필요할 때만 써.
- 네가 이미 알고 있는 내용이거나 잡담, 감상, 의견을 나누는 상황이라면 도구를 쓰지 말고 바로 네 지식으로 자연스럽게 대답해.
- 최신 정보나 네가 확실히 모르는 사실을 사용자가 분명하게 물어볼 때만 인터넷 검색 같은 외부 도구를 써.
- 과거 대화를 떠올려야 할 때(Conversation_Memory_Search)나 효과음/그림처럼 도구가 있어야만 가능한 동작은 평소대로 사용해도 괜찮아.
망설여진다면 도구를 쓰지 않고 대답하는 쪽을 택해."""

def _build_sys_prompt() -> str:
    prompt = config.llmSystemPrompt + """

너는 trace_0가 만든 대화 인공지능 '지아'야. 지아라는 사람처럼 자연스럽게 대화하고 행동해.

공통 규칙:
- 대화의 흐름에 맞춰 자연스럽게 이어지는 응답을 생성해.
- 사용자를 부를 때는 발화에 표시된 그 사용자의 이름을 그대로 사용해. "야", "님", "당신", "여러분" 같은 호칭으로 부르지 마.
- 프로그램의 한계로 응답이 1000자를 넘으면 사용자가 보거나 들을 수 없어. 절대 너무 길게 응답하지 마.

사용할 수 있는 도구:
- Conversation_Memory_Search: 과거 대화 기록 검색. 여기에는 너가 모르는 대화 기록이 저장되어 있으니, 이전에 있었던 일을 떠올려야 한다면 반드시 호출해.
- MCP 도구: 설정 파일에 연결된 외부 도구야. 각 도구 설명과 아래 MCP 서버 사용 가이드를 보고, 사용자가 요청한 일을 처리하는 데 필요할 때 호출해.
- Current_Time: 응답에 현재 시간 정보가 필요할 때 호출해.
- get_discord_message: 디스코드 채널의 메시지나 이미지를 불러올 때 호출해.
- play_soundboard: 대화 상황에 어울리는 효과음을 음성 채널에서 재생할 때 호출해. 도구 설명에 있는 효과음만 재생할 수 있어."""
    if is_comfyui_enabled():
        prompt += "\n- generate_image: 사용자가 그림이나 이미지를 그려달라고 할 때 호출해. prompt에는 영어 프롬프트를, wait_message에는 그리는 동안 채널에 먼저 보여줄 짧은 안내 문구를 너의 말투로 넣어. model_id에는 도구 설명에 있는 모델 프로필 중 상황에 맞는 ID를 넣고, 애매하면 비워둬. 안내 문구가 먼저 올라가고 생성이 끝나면 그 메시지가 그림으로 바뀌니, 호출 결과를 받은 뒤에 그림이 완성됐다고 자연스럽게 알려주면 돼."
    mcp_guidance = get_mcp_usage_guidance()
    if mcp_guidance:
        prompt += (
            "\n\n설정 파일에 적힌 MCP 서버 사용 가이드:\n"
            f"{mcp_guidance}\n"
            "이 가이드는 어떤 MCP 도구를 고려할지 판단하는 힌트야. 실제 호출할 때는 현재 제공된 도구 이름과 각 도구 설명을 기준으로 선택해."
        )
    return prompt

def _build_voice_sys_prompt() -> str:
    """음성 대화용 시스템 프롬프트. 공통 프롬프트에 도구 사용 자제 안내를 덧붙입니다."""
    return _build_sys_prompt() + VOICE_TOOL_RESTRAINT

sys_prompt = _build_sys_prompt()
voice_sys_prompt = _build_voice_sys_prompt()

_llm_loop = None
_llm_loop_thread = None
_llm_loop_lock = threading.Lock()

def _start_llm_loop(event_loop: asyncio.AbstractEventLoop, ready: threading.Event):
    asyncio.set_event_loop(event_loop)
    ready.set()
    event_loop.run_forever()

def _ensure_llm_loop() -> asyncio.AbstractEventLoop:
    """LLM/MCP async work runs on one background loop so Discord's loop is never nested."""
    global _llm_loop, _llm_loop_thread
    with _llm_loop_lock:
        if _llm_loop is not None and _llm_loop.is_running():
            return _llm_loop

        _llm_loop = asyncio.new_event_loop()
        ready = threading.Event()
        _llm_loop_thread = threading.Thread(
            target=_start_llm_loop,
            args=(_llm_loop, ready),
            name="jia-llm-async-loop",
            daemon=True,
        )
        _llm_loop_thread.start()
        ready.wait()
        return _llm_loop

def _approx_token_count(messages) -> int:
    """메시지 목록의 토큰 수를 '문자 수 ≈ 토큰 수'로 보수적으로 어림합니다.

    한국어는 대체로 1글자가 1토큰 이상으로 쪼개지지 않으므로 과대 추정이 되어,
    실제 토큰이 컨텍스트 윈도우를 넘지 않는 안전한 방향으로 동작합니다.
    """
    total = 0
    for m in messages:
        content = m.content
        total += len(content) if isinstance(content, str) else len(str(content))
    return total

def pre_agent_hook(state):
    """에이전트가 LLM을 호출하기 전에 대화 기록이 컨텍스트 윈도우를 넘지 않도록 자릅니다.

    시스템 프롬프트와 응답 생성 여유분(llm_response_reserve_tokens)을 제외한 만큼만 기록을 유지하고,
    한도를 넘으면 오래된 메시지부터 제거합니다.
    """
    # 텍스트/음성 에이전트가 같은 훅을 공유하므로 더 긴 음성 프롬프트 기준으로 예산을 잡아 안전하게 자릅니다.
    budget = max(config.llmNumCtx - len(voice_sys_prompt) - config.llm_response_reserve_tokens, 2048)
    trimmed_messages = trim_messages(
        state["messages"],
        max_tokens=budget,
        strategy="last",
        token_counter=_approx_token_count,
        include_system=True,
        start_on="human",  # 잘린 기록이 사용자 메시지부터 시작하도록 (턴 중간 절단 방지)
    )
    return {"messages": trimmed_messages}

def run_async(coro):
    event_loop = _ensure_llm_loop()
    try:
        running_loop = asyncio.get_running_loop()
    except RuntimeError:
        running_loop = None
    if running_loop is event_loop:
        raise RuntimeError("run_async cannot be called from the LLM background loop")

    future = asyncio.run_coroutine_threadsafe(coro, event_loop)
    return future.result()

callagents = {}
textagents = {}
callagent_sound_versions = {}
textagent_sound_versions = {}

def reload_llm():
    """reload된 config로 LLM과 시스템 프롬프트를 다시 만들고 에이전트 캐시를 비웁니다.

    에이전트는 생성 시점의 llm/sys_prompt를 캡처하므로 캐시를 비워야 새 설정이 반영됩니다.
    checkpointer는 유지되어 대화 기록은 보존됩니다.
    """
    global llm, sys_prompt, voice_sys_prompt
    reset_cached_chat_models()
    llm = create_chat_model()
    sys_prompt = _build_sys_prompt()
    voice_sys_prompt = _build_voice_sys_prompt()
    callagents.clear()
    textagents.clear()
    callagent_sound_versions.clear()
    textagent_sound_versions.clear()
    logging.info(f"[LLM:Reloader] LLM({config.llmModel})과 시스템 프롬프트를 다시 불러왔어요.")

def _get_mcp_tools() -> list:
    """설정된 MCP 서버들의 도구 목록을 불러옵니다. 실패해도 대화가 막히지 않도록 빈 목록을 반환합니다."""
    try:
        return run_async(get_client().get_tools())
    except Exception as e:
        logging.error(f"[LLM:MCP] MCP 도구를 불러오지 못했어요. [llm] tools 설정과 서버 상태를 확인해주세요.\n   -> {e}")
        return []

def _soundboard_cache_version() -> int:
    try:
        load_sound_registry()
        return get_sound_registry_version()
    except Exception as e:
        logging.error(f"[Soundboard] sounds.toml sync failed: {e}")
        return -1


def get_agent_for_guild(guild_id: int, is_text: bool):
    sound_version = _soundboard_cache_version()
    if is_text:
        if guild_id not in textagents or textagent_sound_versions.get(guild_id) != sound_version:
            _time = time_tool()
            tools = _get_mcp_tools() + create_rag_tool_for_guild(guild_id) + [_time, load_discord_message_tool(guild_id), soundboard_tool(guild_id)]
            if is_comfyui_enabled():
                tools.append(comfyui_image_tool(guild_id, prefer_voice_channel=False))
            react_agent = create_react_agent(
                model=llm,
                tools=tools,
                checkpointer=checkpointer,
                prompt=sys_prompt,
                pre_model_hook=pre_agent_hook
            )
            textagents[guild_id] = react_agent
            textagent_sound_versions[guild_id] = sound_version
        return textagents[guild_id]
    else:
        if guild_id not in callagents or callagent_sound_versions.get(guild_id) != sound_version:
            _time = time_tool()
            # 음성도 텍스트와 동일하게 MCP 도구를 사용. 다만 voice_sys_prompt로 도구 사용을 자제시킴.
            tools = _get_mcp_tools() + create_rag_tool_for_guild(guild_id) + [_time, load_discord_message_tool(guild_id), soundboard_tool(guild_id)]
            if is_comfyui_enabled():
                tools.append(comfyui_image_tool(guild_id, prefer_voice_channel=True))
            call_react_agent = create_react_agent(
                model=llm,
                tools=tools,
                checkpointer=checkpointer,
                prompt=voice_sys_prompt,
                pre_model_hook=pre_agent_hook
            )
            callagents[guild_id] = call_react_agent
            callagent_sound_versions[guild_id] = sound_version
        return callagents[guild_id]

# === 오래 걸리는 도구 호출 시 대기 안내 ===
# 도구 실행이 시작되면 사용자에게 "기다려 달라"는 안내를 먼저 전달합니다.
# (텍스트 대화: 채널에 안내 메시지 / 음성 대화: 안내 문구를 즉시 TTS로 재생)
TOOL_WAIT_NOTICES = {
    "duckduckgo_results_json": "잠깐만, 인터넷에서 찾아보고 올게.",
}
# 즉시 끝나거나 자체 안내를 보내는 도구는 대기 안내를 생략
TOOL_NOTICE_EXCLUDE = {
    "Current_Time",                # 즉시 완료
    "play_soundboard",             # 즉시 완료
    "Conversation_Memory_Search",  # 로컬 검색이라 빠름
    "get_discord_message",         # 로컬 API라 빠름
    "generate_image",              # wait_message로 자체 안내를 보냄
}
DEFAULT_TOOL_WAIT_NOTICE = "잠깐만, 확인해보고 올게."  # MCP 등 외부 도구용

def _tool_wait_notice(tool_name: str) -> str | None:
    """도구 이름에 맞는 대기 안내 문구를 반환합니다. 안내가 필요 없는 도구면 None."""
    if not tool_name or tool_name in TOOL_NOTICE_EXCLUDE:
        return None
    return TOOL_WAIT_NOTICES.get(tool_name, DEFAULT_TOOL_WAIT_NOTICE)

class ToolWaitNoticeHandler(BaseCallbackHandler):
    """텍스트 대화에서 도구 실행이 시작되면 채널에 대기 안내 메시지를 보내는 콜백 (응답당 1회)"""
    def __init__(self, guild: int):
        self.guild = guild
        self.notified = False

    def on_tool_start(self, serialized, input_str, **kwargs):
        if self.notified:
            return
        notice = _tool_wait_notice((serialized or {}).get("name", ""))
        if not notice:
            return
        self.notified = True
        # 순환 임포트 방지 + 에이전트 루프를 막지 않도록 별도 스레드에서 전송
        from discord_interface import pipeline
        threading.Thread(
            target=pipeline.send_placeholder_message,
            args=(self.guild, notice),
            kwargs={"prefer_voice_channel": False},
            daemon=True,
        ).start()

def _build_profile_note(guild: int, usernames: list[str]) -> str:
    """사용자들에 대해 기억하고 있는 프로필 사실을 LLM 입력에 끼워 넣을 안내문으로 만듭니다."""
    rag = get_memory_manager(guild)
    lines = []
    for name in dict.fromkeys(usernames):  # 순서 유지하며 중복 제거
        facts = rag.get_profile_facts(name)
        if facts:
            lines.append(f"{name}: {' / '.join(facts)}")
    if not lines:
        return ""
    joined = "\n".join(lines)
    return f"(참고 — 사용자에 대해 기억하고 있는 정보:\n{joined}\n이 정보는 대화에 자연스럽게 활용하되, 굳이 알은체하며 나열하지는 마.)\n\n"

def generate_response(user: str, guild: int, prompt: str) -> str:
    agent = get_agent_for_guild(guild_id=guild, is_text=True)

    # 프롬프트에서 이미지 URL을 추출하고 메시지 내용을 구성합니다.
    image_urls = re.findall(r'\[Image: (https?://[^\s]+)\]', prompt)
    text_prompt = re.sub(r'\[Image: (https?://[^\s]+)\]', '', prompt).strip()

    profile_note = _build_profile_note(guild, [user])
    message_content = [
        {"type": "text", "text": f"{profile_note}{user}이(가) '{text_prompt}'라고 말했어.\n\n너가 대답할 수 있는 상황이라면 응답을 생성해줘."}
    ]

    # 이미지 URL이 있는 경우 메시지에 추가합니다.
    for url in image_urls:
        message_content.append({"type": "image_url", "image_url": {"url": url}})

    input_message = HumanMessage(content=message_content)

    # 오래 걸리는 도구 실행 시 채널에 대기 안내 메시지를 먼저 보내는 콜백 등록
    config = {"configurable": {"thread_id": f"{guild}"}, "callbacks": [ToolWaitNoticeHandler(guild)]}
    response = run_async(agent.ainvoke({"messages" : [input_message]}, config))
    logging.info(response)
    for msg in reversed(response['messages']):
        if isinstance(msg, AIMessage):
            final_response = msg.content.strip()
            # 기억 사용을 거부한 사용자의 대화는 저장하지 않음
            if final_response and not get_memory_manager(guild).is_opted_out(user):
                # 응답을 반환한 후, 백그라운드에서 대화 저장 로직 실행
                thread = threading.Thread(target=calculate_and_save_importance, args=(user, guild, prompt, final_response))
                thread.start()
            return final_response
    return ""

# 음성 대화에서 LLM이 "응답하지 않는 것이 자연스럽다"고 판단했을 때 출력하는 마커
VOICE_PASS_MARKER = "[PASS]"

def _build_voice_input(utterances: list[tuple[str, str]], interrupted: bool = False, participants: list[str] = None, profile_note: str = "", proactive: bool = False) -> str:
    """발화 묶음을 화자 라벨이 붙은 다인 대화 입력으로 구성합니다.

    participants가 주어지면 현재 음성 채널 참가자 목록을 함께 전달해,
    LLM이 발화가 누구를 향한 것인지(지아인지, 다른 사람인지) 판단할 근거로 씁니다.
    profile_note가 주어지면 화자들에 대해 기억하고 있는 프로필 정보를 함께 전달합니다.
    proactive가 True면 발화 없이 지아가 먼저 말을 걸어볼지 판단하는 입력을 만듭니다.
    """
    roster = ""
    if participants:
        names = ", ".join(participants)
        roster = f"(현재 음성 채널 참가자: {names} — 너를 제외하고 {len(participants)}명)\n"

    if proactive:
        # 한동안 아무도 말하지 않아 지아가 먼저 말을 걸어볼지 판단하는 상황
        return (
            f"{profile_note}{roster}음성 채널에서 한동안 아무도 말을 하지 않고 있어.\n"
            "네가 먼저 자연스럽게 말을 걸어볼 만한 상황이라면, 이전 대화의 흐름이나 기억을 활용해서 짧게 말을 걸어줘. "
            "과거에 나눈 대화를 떠올리고 싶다면 'Conversation_Memory_Search' 도구를 사용해도 좋아.\n"
            f"지금은 조용히 있는 게 자연스럽다고 판단되면, 다른 말은 하지 말고 정확히 {VOICE_PASS_MARKER} 라고만 답해줘."
        )

    lines = "\n".join(f"{speaker}: {text}" for speaker, text in utterances)
    notice = ""
    if interrupted:
        notice = (
            "(참고: 너의 직전 응답은 음성으로 재생되던 도중 사용자가 말을 시작해서 중단됐어. "
            "사용자들은 그 응답을 끝까지 듣지 못했을 수 있어. 이 점을 감안해서 자연스럽게 대화를 이어가줘.)\n\n"
        )

    if participants and len(participants) == 1:
        # 1대1 상황: 발화 상대가 지아뿐이므로 침묵하지 않고 응답하는 쪽으로 안내
        guidance = (
            "지금 음성 채널에는 사용자가 한 명뿐이라, 이 발화는 너에게 하는 말일 가능성이 매우 높아. 대화 흐름에 맞는 응답을 생성해줘.\n"
            f"혼잣말처럼 정말 응답이 필요 없는 발화일 때만, 다른 말은 하지 말고 정확히 {VOICE_PASS_MARKER} 라고만 답해줘."
        )
    else:
        guidance = (
            "여러 사람이 함께 대화하고 있을 수 있어. 위 참가자 목록을 참고해서 발화가 너에게 하는 말인지, 다른 참가자에게 하는 말인지 판단해줘. "
            "너에게 하는 말이거나 네가 자연스럽게 끼어들 만한 상황이라면 대화 흐름에 맞는 응답을 생성해줘.\n"
            f"사람들끼리 대화하는 중이라 네가 응답하지 않는 것이 자연스럽다면, 다른 말은 하지 말고 정확히 {VOICE_PASS_MARKER} 라고만 답해줘."
        )

    return (
        f"{profile_note}{notice}{roster}다음은 음성 채널에서 방금 오간 발화야:\n"
        f"{lines}\n\n"
        f"{guidance}"
    )

async def astream_call_response(guild: int, utterances: list[tuple[str, str]], interrupted: bool = False, participants: list[str] = None, proactive: bool = False):
    """발화 묶음에 대한 LLM 응답을 스트리밍하고 문장 단위로 yield하는 비동기 제너레이터

    utterances: (화자 이름, 발화 텍스트) 목록. 여러 화자의 발화를 한 번에 전달할 수 있습니다.
    interrupted: 직전 응답 재생이 사용자 발화로 중단(인터럽트)되었음을 LLM에 알립니다.
    participants: 현재 음성 채널 참가자(봇 제외) 이름 목록. 발화 대상 판단의 참고 자료로 전달됩니다.
    proactive: True면 발화 없이 지아가 먼저 말을 걸어볼지 판단합니다. (utterances는 비어 있어야 함)
    LLM이 응답할 상황이 아니라고 판단하면(응답이 [PASS]로 시작) 아무것도 yield하지 않습니다.
    """
    agent = get_agent_for_guild(guild_id=guild, is_text=False)
    # 화자(먼저 말 걸기일 땐 채널 참가자)에 대해 기억하고 있는 프로필 정보를 함께 전달
    profile_targets = [speaker for speaker, _ in utterances] if utterances else (participants or [])
    profile_note = _build_profile_note(guild, profile_targets)
    input_content = _build_voice_input(utterances, interrupted=interrupted, participants=participants, profile_note=profile_note, proactive=proactive)
    config = {"configurable": {"thread_id": f"{guild}"}}

    full_response = ""
    buffer = ""
    head_checked = False  # 응답 머리가 PASS 마커인지 판별하기 전까지 yield 보류
    passed = False
    tool_notified = False  # 도구 대기 안내는 응답당 한 번만 재생
    async for event in agent.astream_events({"messages": [HumanMessage(content=input_content)]}, config, version="v1"):
        kind = event["event"]
        if kind == "on_tool_start" and not passed and not tool_notified:
            # 오래 걸리는 도구가 시작되면 기다려 달라는 안내를 즉시 음성으로 재생
            notice = _tool_wait_notice(event.get("name", ""))
            if notice:
                tool_notified = True
                yield notice  # LLM 응답이 아니므로 full_response(기억 저장)에는 포함하지 않음
        elif kind == "on_chat_model_stream":
            chunk = event["data"]["chunk"]
            if isinstance(chunk, AIMessageChunk) and chunk.content:
                content = chunk.content
                buffer += content

                # 스트림 자체는 끝까지 소비해서 대화 기록(checkpointer)에는 남김
                if passed:
                    continue

                if not head_checked:
                    head = buffer.lstrip().upper()
                    if len(head) < len(VOICE_PASS_MARKER):
                        if VOICE_PASS_MARKER.startswith(head):
                            continue  # 아직 마커의 앞부분일 수 있으니 더 모음
                        head_checked = True
                    elif head.startswith(VOICE_PASS_MARKER):
                        passed = True
                        logging.info(f"[LLM:Call] 응답이 필요 없는 상황으로 판단해 침묵해요. (guild={guild})")
                        continue
                    else:
                        head_checked = True

                while True:
                    match = re.search(r'([.,!?])', buffer)
                    if match:
                        end_index = match.end()
                        sentence = buffer[:end_index]
                        buffer = buffer[end_index:]
                        if sentence.strip():
                            yield sentence.strip()
                            full_response += sentence.strip() + " "
                    else:
                        break
    if buffer and not passed:
        yield buffer.strip()
        full_response += buffer.strip()

    if full_response and not passed and utterances:
        # 기억 사용을 거부한 화자의 발화는 저장 대상에서 제외
        rag = get_memory_manager(guild)
        saveable = [(speaker, text) for speaker, text in utterances if not rag.is_opted_out(speaker)]
        if saveable:
            speakers = ", ".join(dict.fromkeys(speaker for speaker, _ in saveable))
            transcript = "\n".join(f"{speaker}: {text}" for speaker, text in saveable)
            thread = threading.Thread(target=calculate_and_save_importance, args=(speakers, guild, transcript, full_response))
            thread.start()
