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
        # check_same_thread=False를 사용하여 여러 스레드에서 DB에 접근할 수 있도록 합니다.
        self.conn = sqlite3.connect(":memory:", check_same_thread=False) 
        self._load_db_from_disk()
        self._init_metadata_db()
        self.index = self._load_index()
    def _get_guild_paths(self):
        DATA_DIR = Path(f"Source/memory/data/{self.guild_id}")
        DATA_DIR.mkdir(exist_ok=True, parents=True)
        INDEX_FILE = DATA_DIR / "chat_faiss_index.faiss"
        METADATA_FILE = DATA_DIR / "chat_metadata.db"
        return DATA_DIR, INDEX_FILE, METADATA_FILE

    def _load_db_from_disk(self):
        """디스크의 DB 파일을 인메모리 DB로 로드합니다."""
        if self.METADATA_FILE.exists() and self.METADATA_FILE.stat().st_size > 0:
            disk_conn = sqlite3.connect(self.METADATA_FILE)
            with disk_conn:
                disk_conn.backup(self.conn)
            logging.info(f"[RAG:DB] 디스크에서 메모리로 DB를 로드했어요: {self.METADATA_FILE}")

    def _save_db_to_disk(self):
        """인메모리 DB를 디스크 파일로 저장합니다."""
        disk_conn = sqlite3.connect(self.METADATA_FILE)
        with disk_conn:
            self.conn.backup(disk_conn)
        logging.info(f"[RAG:DB] 메모리 DB를 디스크에 저장했어요: {self.METADATA_FILE}")

    def _init_metadata_db(self):
        """인메모리 DB에 테이블이 없으면 생성합니다."""
        cur = self.conn.cursor()
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
        )"""
        )

    def _load_index(self):
        if self.INDEX_FILE.exists():
            return faiss.read_index(str(self.INDEX_FILE))
        else:
            quantizer = faiss.IndexFlatIP(self.embedding_dim)
            return faiss.IndexIDMap(quantizer)

    def _save_conversation_internal(self, user: str, summary: str, importance: float):
        if not summary:
            return

        date = datetime.now().isoformat()
        forgettable = importance < 0.8 # 중요도 0.8 미만은 잊어버릴 수 있는 기억으로 처리
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
        cur = self.conn.cursor()
        cur.execute("""
        INSERT INTO chat_metadata (id, username, summary, importance, forgettable, timestamp)
        VALUES (?, ?, ?, ?, ?, ?)
        """, (
            item["id"], item["username"], item["summary"],
            item["importance"], int(item["forgettable"]), item["timestamp"]
        ))
        row_id = cur.lastrowid
        self.conn.commit()

        # 벡터와 DB의 rowid를 FAISS 인덱스에 추가
        vector = normalize(vector.reshape(1, -1), axis=1)
        self.index.add_with_ids(vector, np.array([row_id], dtype=np.int64))

        self.save_all()

    def save_index(self):
        faiss.write_index(self.index, str(self.INDEX_FILE))
        logging.info(f"[RAG:faiss] faiss 인덱스 파일을 저장했어요. / 저장된 인덱스 파일 경로: {self.INDEX_FILE}")

    def save_all(self):
        """FAISS 인덱스와 DB를 모두 디스크에 저장합니다."""
        self.save_index()
        self._save_db_to_disk()

    def _get_metadata_from_index(self, idx):
        cur = self.conn.cursor()
        # idx는 numpy.int64일 수 있으므로 int로 변환합니다.
        cur.execute("SELECT summary, importance, timestamp FROM chat_metadata WHERE rowid = ?", (int(idx),))
        row = cur.fetchone()
        return row

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
            logging.info("[RAG:faiss] 서버 대화 기록 DB에서 결과 값을 찾았어요.")
            summary, importance, timestamp = meta
            warn = ""
            if importance < 0.5:
                warn = "\n정확하지 않거나 중요하지 않은 내용이니 이 내용을 참고할 땐 조심스럽게 사용해줘."
            results.append(f"[{timestamp}] {summary}{warn}")
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
        # 1. 기존 인덱스 파일 삭제
        if INDEX_FILE.exists():
            INDEX_FILE.unlink()

        # 2. 새로운 인덱스 객체 생성
        quantizer = faiss.IndexFlatIP(self.embedding_dim)
        new_index = faiss.IndexIDMap(quantizer)

        # 3. 대화 기억 메타데이터 불러오기 (rowid 포함)
        cur = self.conn.cursor()
        cur.execute("SELECT rowid, summary FROM chat_metadata ORDER BY timestamp ASC")
        chat_rows = cur.fetchall()

        vectors = []
        ids = []

        # 4. 불러온 기억 메타데이터를 벡터화 및 ID 수집
        for row_id, summary in chat_rows:
            if not summary:
                logging.warning(f"[RAG:faiss] 대화 기억 메타데이터에 빈 요약이 있어 무시합니다. rowid: {row_id}")
                continue
            vector = self.embedding_model.encode([summary], convert_to_numpy=True)[0].astype('float32')
            vectors.append(vector)
            ids.append(row_id)

        # 5. 벡터화 및 매핑한 결과값을 faiss에 저장
        if vectors:
            vectors = normalize(np.vstack(vectors), axis=1)
            new_index.add_with_ids(vectors, np.array(ids, dtype=np.int64))

            faiss.write_index(new_index, str(INDEX_FILE))
            logging.info(f"[RAG:faiss] 전체 {len(vectors)}개 벡터로 인덱스롤 새로 생성했어요.")
        else:
            logging.warning("[RAG:faiss] 생성할 벡터가 없어요.")

        self.index = new_index

def save_conversation(user: str, guild_id: int, summary: str, importance: float):
    """
    대화 내용을 RAG 시스템에 저장하기 위한 외부 호출용 함수입니다.
    내부적으로 길드별 RAG 인스턴스를 관리합니다.
    """
    rag_instance = get_rag_instance(guild_id)
    rag_instance._save_conversation_internal(user=user, summary=summary, importance=importance)