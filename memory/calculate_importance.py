from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from langchain_core.output_parsers import PydanticOutputParser
from langchain_ollama.chat_models import ChatOllama
from config.config_manager import config
from memory.RAG import save_conversation
import logging

class ConversationSummary(BaseModel):
    """대화 요약 및 중요도 평가를 위한 데이터 구조"""
    summary: str = Field(description="대화 내용을 나중에 기억할 수 있도록 한 문장으로 요약한 내용. 요약할 내용이 없다면 빈 문자열로 둔다.")
    importance: float = Field(description="이 대화의 중요도. 0.0에서 1.0 사이의 값이며, 1.0에 가까울수록 중요함.")

def calculate_and_save_importance(user_name: str, guild_id: int, user_prompt: str, assistant_response: str):
    """대화 내용을 바탕으로 요약과 중요도를 생성하고, 일정 중요도 이상일 경우 RAG에 저장합니다."""
    if not assistant_response:
        logging.info("[RAG:importance]어시스턴트 응답이 없어 대화를 저장하지 않습니다.")
        return

    try:
        llm = ChatOllama(model=config.llmModel)
        parser = PydanticOutputParser(pydantic_object=ConversationSummary)

        prompt_template = ChatPromptTemplate.from_messages([
            ("system", """당신은 대화 내용을 분석하고 요약하는 AI입니다. 주어진 대화 내용을 바탕으로, 나중에 기억할 만한 가치가 있는지 판단하고, 한 문장으로 요약한 후 중요도 점수를 0.0에서 1.0 사이로 매겨주세요.
중요도는 이 대화가 나중에 사용자에게 다시 언급될 가치가 있는지, 사용자의 성향, 관계, 중요한 사건 등을 담고 있는지에 따라 결정됩니다. 중요한 정보나 사건은 0.7 이상으로 평가하세요.
사용자의 개인 정보(이름, 나이, 직업 등), 중요한 약속, 감정적인 교류, 새로운 사실 등이 포함된 경우 중요도가 높습니다.
{format_instructions}"""),
            ("human", """다음은 사용자와 AI 어시스턴트 '지아'의 대화입니다.
사용자 ({user_name}): {user_prompt}
지아: {assistant_response}
이 대화 내용을 한국어로 요약하고 중요도를 평가해주세요.""")
        ])

        prompt = prompt_template.format_prompt(
            format_instructions=parser.get_format_instructions(),
            user_name=user_name,
            user_prompt=user_prompt,
            assistant_response=assistant_response
        )

        response = llm.invoke(prompt.to_messages())
        parsed_response = parser.parse(response.content)

        if parsed_response.summary and parsed_response.importance > config.rag_save_importance_min:
            save_conversation(user=user_name, guild_id=guild_id, summary=parsed_response.summary, importance=parsed_response.importance)
            logging.info(f"[RAG:importance]대화 저장 완료: 서버 아이디({guild_id}), 사용자({user_name}), 요약({parsed_response.summary}), 중요도({parsed_response.importance})")
        else:
            logging.info(f"[RAG:importance]중요도가 낮아 대화를 저장하지 않음: 요약({parsed_response.summary}), 중요도({parsed_response.importance})")

    except Exception as e:
        logging.error(f"[RAG:importance]대화 요약 및 저장 중 오류 발생: {e}")
