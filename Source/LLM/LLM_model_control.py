import ollama
import logging

def unload_ollama_model(model_name: str):
    """지정된 Ollama 모델을 메모리에서 언로드합니다."""
    system_prompt = "If you receive 'ul', you just reply 'ul'."
    messages = [
        {"role" : "system", "content" : system_prompt},
        {"role" : "user", "content" : "ul"}
    ]
    logging.info(f"[LLM:Unload] \"{model_name}\" 모델 언로드를 위해 \"ul\" 메시지 전달!")
    response = ollama.chat(
        model=model_name,
        messages=messages,
        keep_alive=0
    )
    logging.info(f"[LLM:Unload] {response['message']['content']} 메시지 수신. 모델이 언로드됩니다.")

def load_ollama_model(model_name: str):
    """지정된 Ollama 모델을 메모리에 로드합니다."""
    system_prompt = "If you receive 'll', you just reply 'll'."
    messages = [
        {"role" : "system", "content" : system_prompt},
        {"role" : "user", "content" : "ll"}
    ]
    logging.info(f"[LLM:Load] \"{model_name}\" 모델 로드를 위해 \"ll\" 메시지 전달!")
    response = ollama.chat(
        model=model_name,
        messages=messages,
        keep_alive=-1
    )
    logging.info(f"[LLM:Load] {response['message']['content']} 메시지 수신. 모델이 메모리에 로드됩니다.")