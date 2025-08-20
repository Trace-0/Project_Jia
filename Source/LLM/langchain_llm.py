from config.config_manager import config
from langchain_ollama.chat_models import ChatOllama
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import InMemorySaver
from langchain.memory import ConversationBufferMemory
from langchain_community.tools import DuckDuckGoSearchResults
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import AIMessage
from memory.RAG import get_context
import logging
from datetime import datetime
from LLM.langchain_tools.mcp_manager import client
import asyncio

llm = ChatOllama(model=config.llmModel)
memory = ConversationBufferMemory(return_messages=True)
checkpointer = InMemorySaver()
calltools = [DuckDuckGoSearchResults()]
sys_prompt = config.llmSystemPrompt
sys_prompt += f"""\n\n현재 시간은 {datetime.now().isoformat()}이야.\n\n대화의 흐름에 맞춰서 자연스럽게 이어지는 응답을 생성해줘.\n다만, 프로그램의 한계로 너의 응답이 1000자를 넘으면 너의 응답을 사용자가 보거나 들을 수 없게 돼. 그러니 절대 너무 길게 응답을 생성하지마.\n\n다시 한 번 말하지만, 너는 지아야. 너는 지아라는 사람처럼 대화하고 행동해야 해."""
system_prompt = ChatPromptTemplate.from_messages([
    ("system", sys_prompt),
    ("human", "{input}"),
    ("placeholder", "{agent_scratchpad}")
])
callagent = create_tool_calling_agent(llm=llm, tools=calltools, prompt=system_prompt)

loop = asyncio.new_event_loop()
asyncio.set_event_loop(loop)

def run_async(coro):
    result = loop.run_until_complete(coro)
    return result

texttools = run_async(client.get_tools())
react_agent = create_react_agent(
    model=llm,
    tools=texttools,
    checkpointer=checkpointer,
    prompt="너는 \"지아\"라는 이름의 친구야. 너의 말투는 친구처럼 반말을 써. 대화의 흐름에 맞춰서 자연스럽게 이어지는 응답을 생성해줘.\n다만, 프로그램의 한계로 너의 응답이 1000자를 넘으면 너의 응답을 사용자가 보거나 들을 수 없게 돼. 그러니 절대 너무 길게 응답을 생성하지마.\n\n다시 한 번 말하지만, 너는 지아야. 너는 지아라는 사람처럼 대화하고 행동해야 해."
)

def generate_response(user: str, guild: int, prompt: str) -> str:
    global llm, memory
    context = get_context(guild_id=guild, user_input=prompt)
    logging.info(f"[LLM:faiss] faiss가 찾은 대화: {context}")
    input = ""
    if context:
        # 대화가 있는 경우, 프롬프트에 추가
        input = f"아래는 너가 참고할 수 있는 대화의 요약본 또는 단어 사전이야. 아래 내용은 과거에 있었던 대화일 가능성이 높으니 응답에 반영할 때 조심해서 사용해야해.\n{context}\n\n여기까지가 참고할 수 있는 내용이야.\n\n"
    input = f"지금 시간은 {datetime.now().isoformat()}이야.\n{input}마지막 발화는 {user}이(가) 말한 '{prompt}'이야.\n\n응답을 생성해줘"
    id = {"configurable": {"thread_id": guild}}
    response = run_async(react_agent.ainvoke({"messages" : [{"role" : "user", "content": input}]}, id))
    logging.info(response)
    for msg in reversed(response['messages']):
        if isinstance(msg, AIMessage):
            return msg.content
    return ""

def generate_call_response(user: str, guild: int, prompt: str) -> str:
    global llm, memory, callagent
    context = get_context(guild_id=guild, user_input=prompt)
    logging.info(f"[LLM:faiss] faiss가 찾은 대화: {context}")
    input = ""
    if context:
        # 대화가 있는 경우, 프롬프트에 추가
        input = f"아래는 너가 참고할 수 있는 대화의 요약본 또는 단어 사전이야. 아래 내용은 과거에 있었던 대화일 가능성이 높으니 응답에 반영할 때 조심해서 사용해야해.\n{context}\n\n여기까지가 참고할 수 있는 내용이야.\n\n"
    input = f"{input}마지막 발화는 {user}이(가) 말한 '{prompt}'이야.\n\n응답을 생성해줘."
    input = f"마지막 발화는 {user}이(가) 말한 '{prompt}'이야.\n\n너가 대답할 수 있는 상황이라면 응답을 생성해줘."
    agent_executor = AgentExecutor(agent=callagent, tools=calltools, memory=memory, verbose=True)
    response = agent_executor.invoke({"input" : input})
    reply = response['output']
    to_remove = ["\n</end_of_turn>", "지아: "]
    for phrase in to_remove:
        if phrase in reply:
            logging.info(f"[LLM:LangChain] LLM이 {phrase} 단어를 사용했어요")
            reply = reply.replace(phrase, "")
    if reply == "":
        return ""
    else:
        return reply.strip()