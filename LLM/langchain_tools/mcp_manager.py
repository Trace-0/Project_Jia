from langchain_mcp_adapters.client import MultiServerMCPClient
from config.config_manager import config
import logging

# MCP 서버 구성은 settings.toml의 [llm] tools에서 읽습니다.
# 형식 예시 ([llm.tools.서버이름] 테이블, MultiServerMCPClient 형식 그대로):
#   [llm.tools.ddg-search]
#   command = "uvx"
#   args = ["duckduckgo-mcp-server"]
#   transport = "stdio"
#
#   [llm.tools.my-remote-server]
#   url = "http://localhost:9000/mcp"
#   transport = "streamable_http"


def _build_client() -> MultiServerMCPClient:
    servers = {}
    for name, spec in (config.llm_tools or {}).items():
        if not isinstance(spec, dict):
            logging.warning(f"[MCP] '{name}' 서버 설정이 테이블 형식이 아니라서 무시했어요: {spec!r}")
            continue
        servers[name] = dict(spec)
    if servers:
        logging.info(f"[MCP] MCP 서버 {len(servers)}개 구성을 불러왔어요: {list(servers)}")
    else:
        logging.info("[MCP] 연결할 MCP 서버가 없어요. ([llm] tools가 비어 있음)")
    return MultiServerMCPClient(servers)


client = _build_client()


def get_client() -> MultiServerMCPClient:
    return client


def rebuild_client():
    """settings.toml의 [llm] tools 변경을 반영해 MCP 클라이언트를 다시 만듭니다."""
    global client
    client = _build_client()
