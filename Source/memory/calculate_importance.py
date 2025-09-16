from config.config_manager import config
import ollama
import logging
from collections import deque

history = deque(maxlen=5)

def llm_base_importance(user, text: str, assistant_response: str) -> tuple[str, float]:
    """
    LLM 기반 중요도 계산
    """
    global history
    history.append(f"{user} : {text}")
    if assistant_response:
        history.append(f"지아 : {assistant_response}")

    prompt = (f"""다음 대화중에 가장 최신이고 기억해야할 내용이 있다면 최대한 짧게 요약해서 출력하고 얼마나 중요한지 0에서 1사이의 소수점 숫자로 표현해줘.\n대화 내용\n{history}\n\n 중요한 내용과 중요도:"""
    )
    res = ollama.chat(
        model=config.llmModel,
        messages=[{"role" : "system", "content" : "너는 기억 관리자야. 대화의 중요도를 판단하고 요약하는 역할을 해. 대화의 중요도는 0에서 1사이의 소수점 숫자로 표현하는데 0으로 갈수록 중요하지 않고 1에 가까울수록 중요해. 중요한 내용과 중요도는 \"@\"로 구분해서 출력해줘. 만약 중요한 내용이 없다고 판단되면 \"None\"을 출력해줘.\n예시: \"엄마가 아프다고 하셨다/0.9\", \"None\"\n\n 내용과 중요도를 구분하는 기호가 \"@\"라서 이 기호를 다른 곳에 사용하지 않게 주의해줘."},
                   {"role": "user", "content": prompt}]
    )
    answer = res["message"]["content"].strip()
    if answer == "None":
        logging.info("[LLM:Importance] 저장할 중요한 내용이 없어요")
        return "", 0.2
    memory = answer.split("@")
    if memory[0] == "None" or memory[1] == "None" or memory[1] == "0":
        logging.info("[LLM:Importance] 저장할 중요한 내용이 없어요")
        return "", 0.2
    else:
        logging.info(f"[LLM:Importance] 요약 : {memory[0]} / 중요도 : {memory[1]}")
        return memory[0], memory[1]
