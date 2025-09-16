import time
from config.config_manager import config
from langchain_ollama.chat_models import ChatOllama
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import InMemorySaver
from langchain_community.tools import DuckDuckGoSearchResults
from langchain_core.messages import AIMessage
from langchain_core.tools import Tool, StructuredTool
from memory.RAG import RAG, save_conversation
import logging
from datetime import datetime
from LLM.langchain_tools.mcp_manager import client
import asyncio
from pydantic import BaseModel, Field
from dotenv import load_dotenv

load_dotenv()

llm = ChatOllama(model=config.llmModel)
checkpointer = InMemorySaver()
memory_managers = {}

def get_memory_manager(guild_id: int) -> RAG:
    if guild_id not in memory_managers:
        memory_managers[guild_id] = RAG(guild_id=guild_id)
    return memory_managers[guild_id]

def create_rag_tool_for_guild(guild_id: int):
    memory_manager = get_memory_manager(guild_id)
    
    def _retrieve_and_inform(query: str) -> str:
        """검색이 완료되었음을 에이전트에게 알리기 위해 대화 기록을 검색하고, 추가 검색을 방지하는 프롬프트를 삽입합니다."""
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
        description="현재 대화와 관련된 과거 대화 내용을 검색합니다. 사용자가 이전에 했던 말을 기억해야 할 때 사용하세요."
    )
    return [retrieve_tool]

def time_tool():
    def _get_current_time() -> str:
        """현재 시간을 ISO 8601 형식으로 반환합니다."""
        return datetime.now().isoformat()
    
    return Tool(
        name="Current_Time",
        func=_get_current_time,
        description="현재 시간을 ISO 8601 형식으로 반환합니다. 시간 관련 정보가 필요하거나 사용자가 현재 시간을 물어볼 때 사용하세요."
    )

class SaveConversationInput(BaseModel):
    user_name: str = Field(description="마지막 발화자의 이름")
    user_input: str = Field(description="마지막 발화자의 발화 내용")
    assistant_response: str = Field(description="사용자의 발화에 대한 어시스턴트(지아)의 응답 내용")
    summary: str = Field(description="대화 내용을 나중에 기억할 수 있도록 한 문장으로 요약한 내용.")
    importance: float = Field(description="이 대화의 중요도. 0.0에서 1.0 사이의 값이며, 1.0에 가까울수록 중요함.")

def create_save_tool_for_guild(guild_id: int):
    def _save(user_name: str, user_input: str, assistant_response: str, summary: str, importance: float) -> str:
        """Wrapper function to call the actual save_conversation with the guild_id."""
        if not assistant_response or assistant_response.strip() == "":
            return "응답 내용이 없어서 저장하지 않았어요."
        save_conversation(user=user_name, user_input=user_input, assistant_response=assistant_response, guild_id=guild_id, summary=summary, importance=importance)
        return f"사용자 '{user_name}'와의 대화를 저장했어요."

    return StructuredTool.from_function(
        func=_save,
        name="Conversation_Memory_Save",
        description="사용자와의 대화가 중요하여 나중에 기억해야 할 필요가 있을 때, 대화 내용을 요약하고 중요도를 평가하여 장기 기억에 저장합니다.",
        args_schema=SaveConversationInput
    )

calltool = [DuckDuckGoSearchResults()]
sys_prompt = config.llmSystemPrompt
sys_prompt += f"""\n\n대화의 흐름에 맞춰서 자연스럽게 이어지는 응답을 생성해줘.\n다만, 프로그램의 한계로 너의 응답이 1000자를 넘으면 너의 응답을 사용자가 보거나 들을 수 없게 돼. 그러니 절대 너무 길게 응답을 생성하지마.\n\n너의 응답을 생성한 후, 만약 대화 내용이 나중에 기억할 만한 가치가 있다고 판단되면 'Conversation_Memory_Save' 도구를 호출해서 대화 내용을 저장해줘. 이때, 대화의 핵심 내용을 요약하고 0.0에서 1.0 사이의 중요도 점수를 매겨야 해.\n만약 사용자가 이전에 있었던 일에 대해 떠올리길 원한다면 'Conversation_Memory_Search' 도구를 호출해줘. 여기에는 너가 모르는 대화 기록이 저장되어 있으니 과거의 일을 떠올려야 한다면 반드시 이 도구를 호출해.\n인터넷 검색이 필요하다면 'DuckDuckGoSearchResults' 도구를 호출해줘.\n응답에 시간 정보가 필요하다면 'Current_Time' 도구를 호출해줘.\n\n다시 한 번 말하지만, 너는 지아야. 너는 지아라는 사람처럼 대화하고 행동해야 해."""

loop = asyncio.new_event_loop()
asyncio.set_event_loop(loop)

def run_async(coro):
    result = loop.run_until_complete(coro)
    return result

callagents = {}
textagents = {}

def get_agent_for_guild(guild_id: int, is_text: bool):
    if is_text:
        if guild_id not in textagents:
            save_tool = create_save_tool_for_guild(guild_id)
            _time = time_tool()
            tools = run_async(client.get_tools()) + create_rag_tool_for_guild(guild_id) + [save_tool] + [_time]
            react_agent = create_react_agent(
                model=llm,
                tools=tools,
                checkpointer=checkpointer,
                prompt=sys_prompt
            )
            textagents[guild_id] = react_agent
        return textagents[guild_id]
    else:
        if guild_id not in callagents:
            save_tool = create_save_tool_for_guild(guild_id)
            _time = time_tool()
            tools = calltool + create_rag_tool_for_guild(guild_id) + [save_tool] + [_time]
            call_react_agent = create_react_agent(
                model=llm,
                tools=tools,
                checkpointer=checkpointer,
                prompt=sys_prompt
            )
            callagents[guild_id] = call_react_agent
        return callagents[guild_id]

def generate_response(user: str, guild: int, prompt: str) -> str:
    agent = get_agent_for_guild(guild_id=guild, is_text=True)
    input = f"지금 시간은 {datetime.now().isoformat()}이야.\n마지막 발화는 {user}이(가) 말한 '{prompt}'이야.\n\n응답을 생성해줘"
    config = {"configurable": {"thread_id": f"{guild}"}}
    response = run_async(agent.ainvoke({"messages" : [{"role" : "user", "content": input}]}, config))
    logging.info(response)
    for msg in reversed(response['messages']):
        if isinstance(msg, AIMessage):
            return msg.content
    return ""

def generate_call_response(user: str, guild: int, prompt: str) -> str:
    agent = get_agent_for_guild(guild_id=guild, is_text=False)
    input_content = f"지금 시간은 {datetime.now().isoformat()}이야.\n마지막 발화는 {user}이(가) 말한 '{prompt}'이야.\n\n너가 대답할 수 있는 상황이라면 응답을 생성해줘."
    config = {"configurable": {"thread_id": f"{guild}"}}
    response = run_async(agent.ainvoke({"messages": [{"role": "user", "content": input_content}]}, config))
    logging.info(response)
    for msg in reversed(response['messages']):
        if isinstance(msg, AIMessage) and msg.content:
            return msg.content.strip()
    return ""