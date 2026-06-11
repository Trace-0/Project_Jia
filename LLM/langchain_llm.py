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
import asyncio
import re
import threading
from dotenv import load_dotenv

load_dotenv()

llm = ChatOllama(model=config.llmModel, keep_alive=-1)
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
sys_prompt = config.llmSystemPrompt
sys_prompt += f"""\n\n너는 trace_0가 만든 대화 인공지능 '지아'야. 너는 지아라는 사람처럼 대화하고 행동해야 해.\n\n대화의 흐름에 맞춰서 자연스럽게 이어지는 응답을 생성해줘.\n다만, 프로그램의 한계로 너의 응답이 1000자를 넘으면 너의 응답을 사용자가 보거나 들을 수 없게 돼. 그러니 절대 너무 길게 응답을 생성하지마.\n\n만약 사용자가 이전에 있었던 일에 대해 떠올리길 원한다면 'Conversation_Memory_Search' 도구를 호출해줘. 여기에는 너가 모르는 대화 기록이 저장되어 있으니 과거의 일을 떠올려야 한다면 반드시 이 도구를 호출해.\n인터넷 검색이 필요하다면 'DuckDuckGoSearchResults' 도구를 호출해줘.\n응답에 시간 정보가 필요하다면 'Current_Time' 도구를 호출해줘.\n디스코드 메시지나 이미지를 불러오고 싶다면 'get_discord_message' 도구를 호출해줘."""

loop = asyncio.new_event_loop()
asyncio.set_event_loop(loop)

def pre_agent_hook(state):
    """
    에이전트가 LLM을 호출하기 전에 메시지를 3턴(6개 메시지)으로 자릅니다.
    """
    trimmed_messages = trim_messages(
        state["messages"],
        max_tokens=6, # 3턴 = 사용자 메시지 3개 + AI 응답 3개
        strategy="last",
        token_counter=len, # 메시지 개수로 계산
        include_system=True,
    )
    return {"messages": trimmed_messages}

def run_async(coro):
    result = loop.run_until_complete(coro)
    return result

callagents = {}
textagents = {}

def get_agent_for_guild(guild_id: int, is_text: bool):
    if is_text:
        if guild_id not in textagents:
            _time = time_tool()
            tools = run_async(client.get_tools()) + create_rag_tool_for_guild(guild_id) + [_time, load_discord_message_tool(guild_id)]
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
            tools = calltool + create_rag_tool_for_guild(guild_id) + [_time, load_discord_message_tool(guild_id)]
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

async def astream_call_response(user: str, guild: int, prompt: str):
    """LLM 응답을 스트리밍하고 문장 단위로 yield하는 비동기 제너레이터"""
    agent = get_agent_for_guild(guild_id=guild, is_text=False)
    input_content = f"{user}이(가) '{prompt}'라고 말했어.\n\n너가 대답할 수 있는 상황이라면 응답을 생성해줘."
    config = {"configurable": {"thread_id": f"{guild}"}}
    
    full_response = ""
    buffer = ""
    async for event in agent.astream_events({"messages": [HumanMessage(content=input_content)]}, config, version="v1"):
        kind = event["event"]
        if kind == "on_chat_model_stream":
            chunk = event["data"]["chunk"]
            if isinstance(chunk, AIMessageChunk) and chunk.content:
                content = chunk.content
                buffer += content
                
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
    if buffer:
        yield buffer.strip()
        full_response += buffer.strip()
    
    if full_response:
        thread = threading.Thread(target=calculate_and_save_importance, args=(user, guild, prompt, full_response))
        thread.start()