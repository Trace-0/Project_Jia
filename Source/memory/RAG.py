import sqlite3
import uuid
from pathlib import Path
from datetime import datetime
import logging
from config.config_manager import config
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import normalize

# 임베딩 모델을 전역 변수로 한 번만 로드합니다.
logging.info("[RAG:Embedding] SentenceTransformer 모델을 로드하고 있어요...")
embedding_model = SentenceTransformer("dragonkue/BGE-m3-ko")
logging.info("[RAG:Embedding] SentenceTransformer 모델을 로드했어요.")

rag_instances = {}

def get_rag_instance(guild_id: int) -> 'RAG':
    """길드별 RAG 인스턴스를 가져오거나 생성합니다."""
    if guild_id not in rag_instances:
        rag_instances[guild_id] = RAG(guild_id)
    return rag_instances[guild_id]

class RAG:
    def __init__(self, guild_id):
        self.guild_id = guild_id
        self.embedding_model = embedding_model
        self.embedding_dim = self.embedding_model.get_sentence_embedding_dimension()
        self.DATA_DIR, self.INDEX_FILE, self.METADATA_FILE = self._get_guild_paths()
        self._init_metadata_db()
        self.index = self._load_index()

    def _get_guild_paths(self):
        DATA_DIR = Path(f"Source/memory/data/{self.guild_id}")
        DATA_DIR.mkdir(exist_ok=True, parents=True)
        INDEX_FILE = DATA_DIR / "chat_faiss_index.faiss"
        METADATA_FILE = DATA_DIR / "chat_metadata.db"
        return DATA_DIR, INDEX_FILE, METADATA_FILE

    def _init_metadata_db(self):
        conn = sqlite3.connect(self.METADATA_FILE)
        cur = conn.cursor()
        cur.execute("""
        CREATE TABLE IF NOT EXISTS chat_metadata (
            id TEXT PRIMARY KEY,
            username TEXT,
            summary TEXT,
            assistant TEXT,
            text TEXT,
            importance REAL,
            forgettable INTEGER,
            timestamp TEXT
        )
        """)
        cur.execute("""
    CREATE TABLE IF NOT EXISTS index_mapping (
                    faiss_index INTEGER PRIMARY KEY,
                    source TEXT NOT NULL,
                    row_id TEXT NOT NULL
                    )
                    """)
        conn.commit()
        conn.close()

    def _load_index(self):
        if self.INDEX_FILE.exists():
            return faiss.read_index(str(self.INDEX_FILE))
        else:
            quantizer = faiss.IndexFlatIP(self.embedding_dim)
            return quantizer


    def _add_vector_with_mapping(self, vector, source, row_id):
        vector = normalize(vector.reshape(1, -1), axis=1)
        self.index.add(vector)
        faiss_index = self.index.ntotal - 1
        conn = sqlite3.connect(self.METADATA_FILE)
        cur = conn.cursor()
        cur.execute("INSERT INTO index_mapping (faiss_index, source, row_id) VALUES (?, ?, ?)", (faiss_index, source, row_id))
        conn.commit()
        conn.close()


    def _save_conversation_internal(self, user: str, user_input: str, assistant_response: str, summary: str, importance: float):
        date = datetime.now().isoformat()
        forgettable = importance < 0.8 # 중요도 0.8 미만은 잊어버릴 수 있는 기억으로 처리
        if not summary == "":
            vector = self.embedding_model.encode([summary], convert_to_numpy=True)[0].astype('float32')

            item = {
                "id": str(uuid.uuid4()),
                "username": user,
                "summary": summary,
                "importance": importance,
                "forgettable": forgettable,
                "timestamp": date
            }

            # DB에 저장
            conn = sqlite3.connect(self.METADATA_FILE)
            cur = conn.cursor()
            cur.execute("""
            INSERT INTO chat_metadata (id, username, summary, importance, forgettable, timestamp)
            VALUES (?, ?, ?, ?, ?, ?)
            """, (
                item["id"], item["username"], item["summary"],
                item["importance"], int(item["forgettable"]), item["timestamp"]
            ))
            conn.commit()
            conn.close()

            self._add_vector_with_mapping(vector, "chat", item["id"])
            # 벡터 추가 후 인덱스를 즉시 저장하여 데이터 유실을 방지합니다.
            self.save_index()

    def save_index(self):
        faiss.write_index(self.index, str(self.INDEX_FILE))
        logging.info(f"[RAG:faiss] faiss 인덱스 파일을 저장했어요. / 저장된 인덱스 파일 경로: {self.INDEX_FILE}")

    def _get_slang_db_path(self):
        DATA_DIR = Path("Source/memory/data")
        DATA_DIR.mkdir(exist_ok=True, parents=True)
        SLANG_DB_FILE = DATA_DIR / "slang_metadata.db"
        self._init_slang_metadata_db()
        return SLANG_DB_FILE

    def _init_slang_metadata_db(self):
        DB_FILE = Path("Source/memory/data/slang_metadata.db")
        conn = sqlite3.connect(DB_FILE)
        cur = conn.cursor()
        cur.execute("""
        CREATE TABLE IF NOT EXISTS slang_metadata (
            id TEXT PRIMARY KEY,
            word TEXT,
            meaning TEXT,
            examples TEXT,
            added_by TEXT,
            timestamp TEXT
        )
        """)
        conn.commit()
        conn.close()

    def save_slang_metadata(self, word, meaning, examples, added_by, timestamp):
        DB_FILE = self._get_slang_db_path()
        conn = sqlite3.connect(DB_FILE)
        cur = conn.cursor()
        slang_id = str(uuid.uuid4())
        cur.execute("""
        INSERT INTO slang_metadata (id, word, meaning, examples, added_by, timestamp)
        VALUES (?, ?, ?, ?, ?, ?)
        """, (
            slang_id, word, meaning, examples, added_by, timestamp
        ))
        conn.commit()
        conn.close()
        text = f"{word}: {meaning}\n 예시: {examples}"
        vector = self.embedding_model.encode([text], convert_to_numpy=True)[0].astype('float32')
        self._add_vector_with_mapping(vector, "slang", slang_id)
        faiss.write_index(self.index, str(self.INDEX_FILE))

    def _get_metadata_from_index(self, idx):
        conn = sqlite3.connect(self.METADATA_FILE)
        cur = conn.cursor()
        cur.execute("SELECT source, row_id FROM index_mapping WHERE faiss_index = ?", (int(idx),))
        result = cur.fetchone()
        conn.close()
        if not result:
            logging.error(f"[RAG:DB] 매핑 메타데이터 DB에서 결과를 찾을 수 없어요.")
            return None
        source, row_id = result
        if source == "chat":
            conn = sqlite3.connect(self.METADATA_FILE)
            cur = conn.cursor()
            cur.execute("SELECT summary, importance, timestamp FROM chat_metadata WHERE id = ?", (row_id,))
        else:
            conn = sqlite3.connect(self._get_slang_db_path())
            cur = conn.cursor()
            cur.execute("SELECT word, meaning, examples FROM slang_metadata WHERE id = ?", (row_id,))
        row = cur.fetchone()
        conn.close()
        return (source, row)

    def retrieve_similar_conversations(self, query: str, top_k: int = 3):
        q_vec = self.embedding_model.encode([query], convert_to_numpy=True).astype('float32')
        q_vec = normalize(q_vec, axis=1)
        D, I = self.index.search(q_vec, top_k)

        results = []
        for score, idx in zip(D[0], I[0]):
            if score < config.faiss_threshold:
                logging.info(f"[RAG:faiss] 검색한 결과가 faiss_threshold({config.faiss_threshold}) 값보다 작아 무시되었어요. / 검색된 결과의 score : {score}")
                continue
            if idx == -1:
                continue
            logging.info(f"[RAG:faiss] {int(idx) + 1} 번째 기억을 불러오고 있어요. / 검색된 결과의 score : {score}")
            meta = self._get_metadata_from_index(idx)
            if not meta:
                logging.error(f"[RAG:faiss] faiss와 연결된 DB에서 값을 찾을 수 없어요. DB에서 임의로 데이터를 삭제했나요?")
                continue
            source, row = meta
            if source == "chat":
                logging.info("[RAG:faiss] 서버 대화 기록 DB에서 결과 값을 찾았어요.")
                summary, importance, timestamp = row
                warn = ""
                if importance < 0.5:
                    warn = "\n정확하지 않거나 중요하지 않은 내용이니 이 내용을 참고할 땐 조심스럽게 사용해줘."
                results.append(f"[{timestamp}] {summary}{warn}")
            elif source == "slang":
                logging.info("[RAG:faiss] 단어 사전 DB에서 결과 값을 찾았어요.")
                word, meaning, examples = row
                results.append(f"{word}: {meaning} (예시: {examples})")
        return results

    def get_context(self, user_input: str) -> str:
        context_snippets = self.retrieve_similar_conversations(user_input, top_k=3)
        context = "\n---\n".join(context_snippets)
        return context.strip()

    def sync_all_metadata_to_faiss(self):
        """
        chat/slang DB의 모든 데이터를 timestamp 기준으로 정렬하여
        FAISS 인덱스를 새로 생성하고 전체 데이터를 추가합니다.
        """
        _, INDEX_FILE, DB_FILE = self._get_guild_paths()

        self._init_metadata_db()
        slang_DB_FILE = self._get_slang_db_path()

        # 1. 기존 인덱스 파일 삭제
        if INDEX_FILE.exists():
            INDEX_FILE.unlink()

        # 2. 새로운 인덱스 객체 생성
        quantizer = faiss.IndexFlatIP(self.embedding_dim)
        index = quantizer

        # 3. 기존 매핑 테이블 삭제
        conn = sqlite3.connect(DB_FILE)
        cur = conn.cursor()
        cur.execute("DELETE FROM index_mapping")
        conn.commit()

        # 4. 대화 기억 메타데이터 불러오기
        cur.execute("SELECT id, summary FROM chat_metadata ORDER BY timestamp ASC")
        chat_rows = cur.fetchall()
        conn.close()

        # 5. 단어 사전 메타데이터 불러오기 
        conn = sqlite3.connect(slang_DB_FILE)
        cur = conn.cursor()
        cur.execute("SELECT id, word, meaning, examples FROM slang_metadata ORDER BY timestamp ASC")
        slang_rows = cur.fetchall()
        conn.close()

        vectors = []
        metadata = []

        # 6. 불러온 기억 메타데이터를 벡터화 및 매핑
        for item_id, summary in chat_rows:
            if not summary:
                logging.warning(f"[RAG:faiss] 대화 기억 메타데이터에 빈 요약이 있어 무시합니다. ID: {item_id}")
                continue
            vector = self.embedding_model.encode([summary], convert_to_numpy=True)[0].astype('float32')
            vectors.append(vector)
            metadata.append(("chat", item_id))
        
        # 7. 불러온 단어 사전 메타데이터를 벡터화 및 매핑
        for item_id, word, meaning, examples in slang_rows:
            text = f"{word}: {meaning}\n 예시: {examples}"
            vector = self.embedding_model.encode([text], convert_to_numpy=True)[0].astype('float32')
            vectors.append(vector)
            metadata.append(("slang", item_id))

        # 8. 벡터화 및 매핑한 결과값을 faiss 및 매핑 테이블에 저장
        if vectors:
            vectors = normalize(np.vstack(vectors), axis=1)
            index.add(vectors)

            conn = sqlite3.connect(DB_FILE)
            cur = conn.cursor()
            indexed_metadata = [(i, source, row_id) for i, (source, row_id) in enumerate(metadata)]
            cur.executemany("INSERT INTO index_mapping (faiss_index, source, row_id) VALUES (?, ?, ?)", indexed_metadata)
            conn.commit()
            conn.close()

            faiss.write_index(index, str(INDEX_FILE))
            logging.info(f"[RAG:faiss] 전체 {len(vectors)}개 벡터로 인덱스롤 새로 생성했어요.")
        else:
            logging.warning("[RAG:faiss] 생성할 벡터가 없어요.")

def save_conversation(user: str, user_input: str, assistant_response: str, guild_id: int, summary: str, importance: float):
    """
    대화 내용을 RAG 시스템에 저장하기 위한 외부 호출용 함수입니다.
    내부적으로 길드별 RAG 인스턴스를 관리합니다.
    """
    rag_instance = get_rag_instance(guild_id)
    rag_instance._save_conversation_internal(user=user, user_input=user_input, assistant_response=assistant_response, summary=summary, importance=importance)