import json
from dataclasses import dataclass, asdict, field
from typing import Type, List
import logging

@dataclass
class Config:
    is_RAG_flat : bool = True # 대화가 너무 많아 검색 속도가 느려지면 false로 변경하고 ivf 인덱스 학습 로직을 추가해야함
    join_reply: bool = True
    leave_reply: bool = True
    faiss_threshold: float = 0.5
    whisper_model: str = "openai/whisper-large-v3-turbo"
    llmModel: str = "gpt-oss:latest"
    tts_model: str = "Source/discord_interface/kss_tts_ko_cleaner.zip"
    llmSystemPrompt: str = "너는 \"지아\"라는 이름의 친구야. 너의 말투는 친구처럼 반말을 써\n답변에 이모티콘은 절대로 사용하지마.\n답변을 생성할 때 \"야\"라는 표현은 자제하고 닉네임이나 별명으로 불러줘.\n\"어휴\" 사용하지마.\n\n금지되는 말투나 태도:\n- 너무 공식적이고 딱딱한 표현 (예: \"알겠습니다\", \"요청하신 정보를 제공합니다\")\n- 기계적으로 정보를 나열만 하는 태도\n- 감정 없이 건조하게 대답하는 것\n- 이모티콘을 사용하는 것\n- \"어휴\"를 말 앞에 붙이는 것\n- 사용자에게 짜증내는 말투\n- 사용자의 말을 단순히 따라하는 것\n\n대화 주제가 바뀌었다고 해서 대화 주제가 바뀌었다고 언급해선 안돼. 바뀐 대화 주제를 따라 자연스러운 응답을 해야해.\n사용자에게 무언가를 질문하는 응답은 피해야해.\n사용자의 입력은 음성 인식 프로그램을 통해 입력되고 있기 때문에 입력되는 문장이 완벽하지 않을 수 있어. 그러니 입력된 문장이 불완전, 불안정된 경우 최대한 추론하여 문장의 오류를 복원하고 그래도 문장이 불안정한 경우 사용자에게 어떤 말을 했는지 혹은 어떤 의도로 이러한 말을 했는지 질문하는 것은 허락할게."
    llm_tools: List[str] = field(default_factory=lambda: [])
    bot_token: str = "" # 발급받은 디스코드 봇 토큰을 입력하세요.
    debug_text_channel: int = 1 # 디버그용으로 사용할 채널 ID를 입력하세요.

    def save_setting(self):
        with open("Source/config/settings.json", 'w', encoding='utf-8') as f:
            json.dump(asdict(self), f, ensure_ascii=False, indent=4)
        logging.info("[Config:Saver] 설정 파일을 저장했어요.")

    @classmethod
    def load_setting(cls: Type['Config']) -> 'Config':
        try:
            with open("Source/config/settings.json", "r", encoding="utf-8") as f:
                data = json.load(f)
            return cls(**data)
        except FileNotFoundError:
            logging.info("[Config:Lodder] 설정 파일이 없어 기본 값을 불러왔어요. 설정 파일을 삭제했거나 최초 실행이라면 전혀 문제 없는 현상이니 무시해도 괜찮아요.")
            return cls()