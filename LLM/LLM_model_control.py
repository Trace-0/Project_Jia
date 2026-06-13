import ollama
import logging
import os
from config.config_manager import config
from langchain_ollama.chat_models import ChatOllama

# 제공자 이름 -> 그 제공자의 SDK가 읽는 API 키 환경 변수.
# 외부 API 키를 환경 변수로 넣어주면 langchain 통합이 알아서 사용합니다. (제공자마다 키 인자명이 달라 환경 변수가 가장 안전)
_PROVIDER_API_KEY_ENV = {
    "openai": "OPENAI_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
    "google_genai": "GOOGLE_API_KEY",
    "groq": "GROQ_API_KEY",
    "mistralai": "MISTRAL_API_KEY",
    "together": "TOGETHER_API_KEY",
    "fireworks": "FIREWORKS_API_KEY",
    "cohere": "COHERE_API_KEY",
    "deepseek": "DEEPSEEK_API_KEY",
    "xai": "XAI_API_KEY",
}


def _provider() -> str:
    """현재 설정된 LLM 제공자 이름(소문자)을 반환합니다. 비어 있으면 ollama로 간주합니다."""
    return (config.llm_provider or "ollama").strip().lower()


def using_ollama() -> bool:
    """로컬 Ollama를 사용 중인지 여부. (외부 API면 모델 로드/언로드가 필요 없음)"""
    return _provider() in ("", "ollama")


def create_chat_model(*, for_agent: bool = True):
    """설정에 맞는 LangChain 채팅 모델을 생성합니다.

    provider가 ollama면 ChatOllama를, 외부 API면 langchain의 init_chat_model로 해당 제공자 모델을 만듭니다.
    for_agent: 대화 에이전트용이면 True (Ollama에서 keep_alive/num_ctx 적용). 단발성 요약 등은 False.
    """
    if using_ollama():
        if for_agent:
            return ChatOllama(model=config.llmModel, keep_alive=-1, num_ctx=config.llmNumCtx)
        return ChatOllama(model=config.llmModel)

    provider = _provider()
    api_key = (config.llm_api_key or "").strip()
    api_base = (config.llm_api_base or "").strip()

    # 키가 있으면 제공자 SDK가 읽는 환경 변수에 넣어줌 (인자명이 제공자마다 달라 환경 변수가 가장 호환성이 높음)
    if api_key:
        env_var = _PROVIDER_API_KEY_ENV.get(provider)
        if env_var:
            os.environ[env_var] = api_key

    kwargs = {"model_provider": provider}
    # 환경 변수 매핑이 없는 제공자에는 키를 직접 전달 (best effort)
    if api_key and provider not in _PROVIDER_API_KEY_ENV:
        kwargs["api_key"] = api_key
    if api_base:
        kwargs["base_url"] = api_base

    try:
        from langchain.chat_models import init_chat_model
        model = init_chat_model(config.llmModel, **kwargs)
        logging.info(f"[LLM:Model] 외부 LLM 제공자 '{provider}'의 모델 '{config.llmModel}'을(를) 사용해요.")
        return model
    except ImportError as e:
        raise RuntimeError(
            f"'{provider}' 제공자를 사용하려면 관련 패키지를 설치해야 해요. "
            f"예: pip install langchain-{provider.replace('_', '-')}  (원래 오류: {e})"
        )


def unload_ollama_model(model_name: str):
    """지정된 Ollama 모델을 메모리에서 언로드합니다. (외부 API 사용 중이면 아무 동작도 하지 않습니다.)

    프롬프트 없는 빈 요청에 keep_alive=0을 설정하면 토큰 생성 없이 즉시 언로드됩니다.
    """
    if not using_ollama():
        return
    logging.info(f"[LLM:Unload] \"{model_name}\" 모델 언로드 요청!")
    response = ollama.generate(model=model_name, keep_alive=0)
    logging.info(f"[LLM:Unload] 응답 수신(done_reason={response.get('done_reason')}). 모델이 언로드됩니다.")

def load_ollama_model(model_name: str):
    """지정된 Ollama 모델을 메모리에 로드합니다. (외부 API 사용 중이면 아무 동작도 하지 않습니다.)

    프롬프트 없는 빈 요청에 keep_alive=-1을 설정하면 토큰 생성 없이 로드만 수행됩니다.
    """
    if not using_ollama():
        return
    logging.info(f"[LLM:Load] \"{model_name}\" 모델 로드 요청!")
    response = ollama.generate(model=model_name, keep_alive=-1)
    logging.info(f"[LLM:Load] 응답 수신(done_reason={response.get('done_reason')}). 모델이 메모리에 로드됩니다.")
