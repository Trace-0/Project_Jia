from langchain_mcp_adapters.client import MultiServerMCPClient
from config.config_manager import config
import logging

MCP_SERVER_METADATA_KEYS = {"description", "use_when"}

# MCP 서버 구성은 settings.toml의 [llm] tools에서 읽습니다.
# 형식 예시 ([llm.tools.서버이름] 테이블, MultiServerMCPClient 형식 그대로):
#   [llm.tools.ddg-search]
#   command = "uvx"
#   args = ["duckduckgo-mcp-server"]
#   transport = "stdio"
#   description = "DuckDuckGo 검색으로 웹 검색 결과를 가져올 수 있습니다."
#   use_when = "최신 정보나 인터넷 확인이 필요한 사실을 물어볼 때 사용합니다."
#
#   [llm.tools.my-remote-server]
#   url = "http://localhost:9000/mcp"
#   transport = "streamable_http"


def _client_spec(spec: dict) -> dict:
    """LLM 안내용 메타데이터를 제외하고 MCP 클라이언트가 이해하는 연결 설정만 반환합니다."""
    return {key: value for key, value in spec.items() if key not in MCP_SERVER_METADATA_KEYS}


def _build_client() -> MultiServerMCPClient:
    servers = {}
    for name, spec in (config.llm_tools or {}).items():
        if not isinstance(spec, dict):
            logging.warning(f"[MCP] '{name}' 서버 설정이 테이블 형식이 아니라서 무시했어요: {spec!r}")
            continue
        servers[name] = _client_spec(spec)
    if servers:
        logging.info(f"[MCP] MCP 서버 {len(servers)}개 구성을 불러왔어요: {list(servers)}")
    else:
        logging.info("[MCP] 연결할 MCP 서버가 없어요. ([llm] tools가 비어 있음)")
    return MultiServerMCPClient(servers)


client = _build_client()


def get_client() -> MultiServerMCPClient:
    return client


def get_mcp_usage_guidance() -> str:
    """settings.toml의 MCP 서버별 설명을 LLM 시스템 프롬프트에 넣을 문자열로 만듭니다."""
    lines = []
    for name, spec in (config.llm_tools or {}).items():
        if not isinstance(spec, dict):
            continue
        description = str(spec.get("description") or "").strip()
        use_when = str(spec.get("use_when") or "").strip()
        if not description and not use_when:
            continue
        parts = []
        if description:
            parts.append(f"할 수 있는 일: {description}")
        if use_when:
            parts.append(f"사용할 상황: {use_when}")
        lines.append(f"- {name}: {' / '.join(parts)}")
    return "\n".join(lines)


def rebuild_client():
    """settings.toml의 [llm] tools 변경을 반영해 MCP 클라이언트를 다시 만듭니다."""
    global client
    client = _build_client()
