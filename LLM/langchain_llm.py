from config.config_manager import config
from langchain_ollama.chat_models import ChatOllama
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import InMemorySaver
from langchain_community.tools import DuckDuckGoSearchResults
from langchain_core.messages import AIMessage, AIMessageChunk, HumanMessage, trim_messages
from langchain_core.tools import Tool
from memory.RAG import RAG
import logging
from datetime import datetime
from LLM.langchain_tools.mcp_manager import client
from memory.calculate_importance import calculate_and_save_importance
from LLM.langchain_tools.load_discord_message import load_discord_message_tool
from LLM.langchain_tools.soundboard import soundboard_tool
import asyncio
import re
import threading
from dotenv import load_dotenv

load_dotenv()

llm = ChatOllama(model=config.llmModel, keep_alive=-1, num_ctx=config.llmNumCtx)
checkpointer = InMemorySaver()
memory_managers = {}

def get_memory_manager(guild_id: int) -> RAG:
    if guild_id not in memory_managers:
        memory_managers[guild_id] = RAG(guild_id=guild_id)
    return memory_managers[guild_id]

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

calltool = [DuckDuckGoSearchResults()]

def _build_sys_prompt() -> str:
    return config.llmSystemPrompt + f"""\n\n너는 trace_0가 만든 대화 인공지능 '지아'야. 너는 지아라는 사람처럼 대화하고 행동해야 해.\n\n대화의 흐름에 맞춰서 자연스럽게 이어지는 응답을 생성해줘.\n다만, 프로그램의 한계로 너의 응답이 1000자를 넘으면 너의 응답을 사용자가 보거나 들을 수 없게 돼. 그러니 절대 너무 길게 응답을 생성하지마.\n\n만약 사용자가 이전에 있었던 일에 대해 떠올리길 원한다면 'Conversation_Memory_Search' 도구를 호출해줘. 여기에는 너가 모르는 대화 기록이 저장되어 있으니 과거의 일을 떠올려야 한다면 반드시 이 도구를 호출해.\n인터넷 검색이 필요하다면 'DuckDuckGoSearchResults' 도구를 호출해줘.\n응답에 시간 정보가 필요하다면 'Current_Time' 도구를 호출해줘.\n디스코드 메시지나 이미지를 불러오고 싶다면 'get_discord_message' 도구를 호출해줘.\n대화 상황에 어울리는 효과음을 음성 채널에서 재생하고 싶다면 'play_soundboard' 도구를 호출해줘. 도구 설명에 있는 효과음만 재생할 수 있어."""

sys_prompt = _build_sys_prompt()

loop = asyncio.new_event_loop()
asyncio.set_event_loop(loop)

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
    budget = max(config.llmNumCtx - len(sys_prompt) - config.llm_response_reserve_tokens, 2048)
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
    result = loop.run_until_complete(coro)
    return result

callagents = {}
textagents = {}

def reload_llm():
    """reload된 config로 LLM과 시스템 프롬프트를 다시 만들고 에이전트 캐시를 비웁니다.

    에이전트는 생성 시점의 llm/sys_prompt를 캡처하므로 캐시를 비워야 새 설정이 반영됩니다.
    checkpointer는 유지되어 대화 기록은 보존됩니다.
    """
    global llm, sys_prompt
    llm = ChatOllama(model=config.llmModel, keep_alive=-1, num_ctx=config.llmNumCtx)
    sys_prompt = _build_sys_prompt()
    callagents.clear()
    textagents.clear()
    logging.info(f"[LLM:Reloader] LLM({config.llmModel})과 시스템 프롬프트를 다시 불러왔어요.")

def get_agent_for_guild(guild_id: int, is_text: bool):
    if is_text:
        if guild_id not in textagents:
            _time = time_tool()
            tools = run_async(client.get_tools()) + create_rag_tool_for_guild(guild_id) + [_time, load_discord_message_tool(guild_id), soundboard_tool(guild_id)]
            react_agent = create_react_agent(
                model=llm,
                tools=tools,
                checkpointer=checkpointer,
                prompt=sys_prompt,
                pre_model_hook=pre_agent_hook
            )
            textagents[guild_id] = react_agent
        return textagents[guild_id]
    else:
        if guild_id not in callagents:
            _time = time_tool()
            tools = calltool + create_rag_tool_for_guild(guild_id) + [_time, load_discord_message_tool(guild_id), soundboard_tool(guild_id)]
            call_react_agent = create_react_agent(
                model=llm,
                tools=tools,
                checkpointer=checkpointer,
                prompt=sys_prompt,
                pre_model_hook=pre_agent_hook
            )
            callagents[guild_id] = call_react_agent
        return callagents[guild_id]

def generate_response(user: str, guild: int, prompt: str) -> str:
    agent = get_agent_for_guild(guild_id=guild, is_text=True)
    
    # 프롬프트에서 이미지 URL을 추출하고 메시지 내용을 구성합니다.
    image_urls = re.findall(r'\[Image: (https?://[^\s]+)\]', prompt)
    text_prompt = re.sub(r'\[Image: (https?://[^\s]+)\]', '', prompt).strip()

    message_content = [
        {"type": "text", "text": f"{user}이(가) '{text_prompt}'라고 말했어.\n\n너가 대답할 수 있는 상황이라면 응답을 생성해줘."}
    ]

    # 이미지 URL이 있는 경우 메시지에 추가합니다.
    for url in image_urls:
        message_content.append({"type": "image_url", "image_url": {"url": url}})

    input_message = HumanMessage(content=message_content)

    config = {"configurable": {"thread_id": f"{guild}"}}
    response = run_async(agent.ainvoke({"messages" : [input_message]}, config))
    logging.info(response)
    for msg in reversed(response['messages']):
        if isinstance(msg, AIMessage):
            final_response = msg.content.strip()
            if final_response:
                # 응답을 반환한 후, 백그라운드에서 대화 저장 로직 실행
                thread = threading.Thread(target=calculate_and_save_importance, args=(user, guild, prompt, final_response))
                thread.start()
            return final_response
    return ""

# 음성 대화에서 LLM이 "응답하지 않는 것이 자연스럽다"고 판단했을 때 출력하는 마커
VOICE_PASS_MARKER = "[PASS]"

def _build_voice_input(utterances: list[tuple[str, str]], interrupted: bool = False, participants: list[str] = None) -> str:
    """발화 묶음을 화자 라벨이 붙은 다인 대화 입력으로 구성합니다.

    participants가 주어지면 현재 음성 채널 참가자 목록을 함께 전달해,
    LLM이 발화가 누구를 향한 것인지(지아인지, 다른 사람인지) 판단할 근거로 씁니다.
    """
    lines = "\n".join(f"{speaker}: {text}" for speaker, text in utterances)
    notice = ""
    if interrupted:
        notice = (
            "(참고: 너의 직전 응답은 음성으로 재생되던 도중 사용자가 말을 시작해서 중단됐어. "
            "사용자들은 그 응답을 끝까지 듣지 못했을 수 있어. 이 점을 감안해서 자연스럽게 대화를 이어가줘.)\n\n"
        )

    roster = ""
    if participants:
        names = ", ".join(participants)
        roster = f"(현재 음성 채널 참가자: {names} — 너를 제외하고 {len(participants)}명)\n"

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
        f"{notice}{roster}다음은 음성 채널에서 방금 오간 발화야:\n"
        f"{lines}\n\n"
        f"{guidance}"
    )

async def astream_call_response(guild: int, utterances: list[tuple[str, str]], interrupted: bool = False, participants: list[str] = None):
    """발화 묶음에 대한 LLM 응답을 스트리밍하고 문장 단위로 yield하는 비동기 제너레이터

    utterances: (화자 이름, 발화 텍스트) 목록. 여러 화자의 발화를 한 번에 전달할 수 있습니다.
    interrupted: 직전 응답 재생이 사용자 발화로 중단(인터럽트)되었음을 LLM에 알립니다.
    participants: 현재 음성 채널 참가자(봇 제외) 이름 목록. 발화 대상 판단의 참고 자료로 전달됩니다.
    LLM이 응답할 상황이 아니라고 판단하면(응답이 [PASS]로 시작) 아무것도 yield하지 않습니다.
    """
    agent = get_agent_for_guild(guild_id=guild, is_text=False)
    input_content = _build_voice_input(utterances, interrupted=interrupted, participants=participants)
    config = {"configurable": {"thread_id": f"{guild}"}}

    full_response = ""
    buffer = ""
    head_checked = False  # 응답 머리가 PASS 마커인지 판별하기 전까지 yield 보류
    passed = False
    async for event in agent.astream_events({"messages": [HumanMessage(content=input_content)]}, config, version="v1"):
        kind = event["event"]
        if kind == "on_chat_model_stream":
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

    if full_response and not passed:
        speakers = ", ".join(dict.fromkeys(speaker for speaker, _ in utterances))
        transcript = "\n".join(f"{speaker}: {text}" for speaker, text in utterances)
        thread = threading.Thread(target=calculate_and_save_importance, args=(speakers, guild, transcript, full_response))
        thread.start()