from langchain_core.tools import Tool
import requests
import logging

def load_discord_message_tool(guild_id: int):
    """지정된 길드의 디스코드 메시지를 불러오는 LangChain 도구"""
    def _get_message(query: str) -> str:
        """
        입력된 채널 이름에서 최근 메시지 10개를 불러옵니다.
        query: 메시지를 불러올 채널의 이름 (문자열)
        """
        try:
            response = requests.post("http://localhost:8001/get-channel-messages", json={
                "guild_id": guild_id,
                "channel_name": query,
                "limit": 10
            })
            response.raise_for_status()
            data = response.json()
            if data.get("status") == "success":
                formatted_messages = []
                for msg in data.get("messages", []):
                    content = msg.get('content', '')
                    for url in msg.get('attachments', []):
                        content += f" [Image: {url}]"
                    formatted_messages.append(f"{msg.get('author')}: {content.strip()}")
                return "\n".join(formatted_messages)
            else:
                return f"메시지를 불러오는 데 실패했습니다: {data.get('message')}"
        except requests.RequestException as e:
            logging.error(f"디스코드 메시지 로드 API 호출 중 오류 발생: {e}")
            return f"API 호출 중 오류가 발생했습니다: {e}"

    return Tool(
        name="get_discord_message",
        func=_get_message,
        description="디스코드의 특정 텍스트 채널에서 최근 대화 기록을 가져옵니다. 채널 이름을 정확하게 입력해야 합니다."
    )