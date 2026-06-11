import ollama
import logging

def unload_ollama_model(model_name: str):
    """지정된 Ollama 모델을 메모리에서 언로드합니다.

    프롬프트 없는 빈 요청에 keep_alive=0을 설정하면 토큰 생성 없이 즉시 언로드됩니다.
    """
    logging.info(f"[LLM:Unload] \"{model_name}\" 모델 언로드 요청!")
    response = ollama.generate(model=model_name, keep_alive=0)
    logging.info(f"[LLM:Unload] 응답 수신(done_reason={response.get('done_reason')}). 모델이 언로드됩니다.")

def load_ollama_model(model_name: str):
    """지정된 Ollama 모델을 메모리에 로드합니다.

    프롬프트 없는 빈 요청에 keep_alive=-1을 설정하면 토큰 생성 없이 로드만 수행됩니다.
    """
    logging.info(f"[LLM:Load] \"{model_name}\" 모델 로드 요청!")
    response = ollama.generate(model=model_name, keep_alive=-1)
    logging.info(f"[LLM:Load] 응답 수신(done_reason={response.get('done_reason')}). 모델이 메모리에 로드됩니다.")
