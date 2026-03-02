import ollama
import discord
from collections import deque
import logging as logger
from config.config_manager import config
from memory.RAG import get_context, save_conversation
from discord_interface.tts_output import generate_tts
from datetime import datetime

history = deque(maxlen=3)

def build_prompt(history, latest_turn):
    """
    Ollama에 넘길 프롬프트 문자열 생성해요
    """

    lines = []
    for turn in history:
        lines.append(f"{turn['speaker']}: {turn['text']}")
    # 마지막 발화까지 포함해요
    lines.append(f"{latest_turn['speaker']}: {latest_turn['text']}")
    return "\n".join(lines)

def generate_reply(history, latest_turn):
    system_prompt = f"""

현재 시간은 {datetime.now().isoformat()}이야.

너는 trace_0가 만든 대화 인공지능이야.
"야"라는 표현보다 그 사람의 이름 혹은 닉네임으로 불러줘.

입력된 대화 내용을 파악하고 응답할 수 있는 상황이라면 응답을 생성하고 아니라면 아무 응답도 하지 않아야 해.
"""

    system_prompt += config.llmSystemPrompt

    model = config.llmModel

    if history:
        history = f"아래는 너가 참고할 수 있는 대화 또는 단어 사전이야. 아래 내용은 과거에 있었던 대화일 가능성이 높으니 응답에 반영할 때 조심해서 사용해야해.\n{history}\n\n여기까지가 참고할 수 있는 대화 내용이야.\n\n"

    messages = [
        {"role" : "system", "content" : system_prompt},
        {"role" : "user", "content" : f"{history}마지막 발화는 {latest_turn['speaker']}이(가) 말한 '{latest_turn['text']}'이야.\n\n만약 여기에 너가 응답할 수 있는 상황이라고 판단된다면 응답을 생성해줘.\n지아: "}
    ]

    response = ollama.chat(
        model=model,
        messages=messages,
        options={
            "temperature" : 0.9,
            "top_p" : 0.9,
            "top_k" : 40,
            "repeat_penalty" : 1.1
        }
    )

    return response["message"]["content"]

async def process_message(user: discord.User, guild: discord.Guild, prompt: str, is_text_chat: bool = False):
    """
    LLM이 응답을 생성하는 과정이에요
    """
    latest = {"speaker": user.name, "text": prompt}

    # 텍스트 채팅인 경우 ESPnet2 TTS 요청을 처리하지 않아요
    if is_text_chat:
        # faiss 검색을 시작해요
        context = get_context(guild_id=guild.id, user_input=prompt)
        logger.info(f"[LLM:faiss] faiss가 찾은 대화: {context}")
        # 지아가 응답을 생성해요
        reply = generate_reply(context, latest)
        # LLM이 응답에 자주 포함시키는 오류 단어를 제거해요
        to_remove = ["\n</end_of_turn>", "\n지아:"]
        for phrase in to_remove:
            if phrase in reply:
                logger.info(f"[LLM:Engine] LLM이 {phrase} 단어를 사용했어요")
                reply = reply.replace(phrase, "")
        save_conversation(user_input=prompt, assistant_response=reply, guild_id=guild.id, user=user)  # 대화 저장
        return reply, None

    logger.info(f"[LLM:Preprocessor] 마지막 응답 : {latest}")
    # faiss 검색을 시작해요
    context = get_context(guild_id=guild.id, user_input=prompt)
    logger.info(f"[LLM:faiss] faiss가 찾은 대화: {context}")
    # 지아가 응답을 생성해요
    reply = generate_reply(context, latest)
    # LLM이 응답에 자주 포함시키는 오류 단어를 제거해요
    to_remove = ["\n</end_of_turn>", "지아: "]
    for phrase in to_remove:
        if phrase in reply:
            logger.info(f"[LLM:Engine] LLM이 {phrase} 단어를 사용했어요")
            reply = reply.replace(phrase, "")
    if reply == "":
        history.append(latest)
        logger.info("[LLM:Engine] LLM이 응답을 생성하지 않았어요")
        save_conversation(user_input=prompt, assistant_response="", guild_id=guild.id, user=user)  # 빈 응답 저장
        return None, None
    wav = generate_tts(reply)
    save_conversation(user_input=prompt, assistant_response=reply, guild_id=guild.id, user=user)  # 대화 저장
    # history 초기화후 새 발화와 응답을 추가해요
    history.clear()
    history.append(latest)
    history.append({"speaker": "지아", "text": reply})
    return reply, wav

async def generate_async(prompt: str, user: discord.User, guild: discord.Guild):
    return await process_message(user, guild, prompt)