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

# 임베딩 모델을 전역 변수로 한 번만 로드합니다. (변경 시 재시작 필요 - 기존 인덱스와 차원이 달라질 수 있음)
logging.info("[RAG:Embedding] SentenceTransformer 모델을 로드하고 있어요...")
embedding_model = SentenceTransformer(config.embedding_model)
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
        DATA_DIR = Path(f"memory/data/{self.guild_id}")
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
        # 망각 시스템용 컬럼 마이그레이션: 마지막으로 검색에 사용된 시각 (없던 DB에는 추가)
        existing_columns = {row[1] for row in cur.execute("PRAGMA table_info(chat_metadata)")}
        if "last_accessed" not in existing_columns:
            cur.execute("ALTER TABLE chat_metadata ADD COLUMN last_accessed TEXT")
        # 사용자별 프로필 (대화에서 알게 된 사용자에 대한 사실)
        cur.execute("""
        CREATE TABLE IF NOT EXISTS user_profiles (
            id TEXT PRIMARY KEY,
            username TEXT,
            fact TEXT,
            timestamp TEXT
        )"""
        )
        # 기억 시스템 사용을 거부한 사용자 목록
        cur.execute("""
        CREATE TABLE IF NOT EXISTS memory_optout (
            username TEXT PRIMARY KEY,
            timestamp TEXT
        )"""
        )
        self.conn.commit()

    def _load_index(self):
        if self.INDEX_FILE.exists():
            return faiss.read_index(str(self.INDEX_FILE))
        else:
            quantizer = faiss.IndexFlatIP(self.embedding_dim)
            return faiss.IndexIDMap(quantizer)

    def _save_conversation_internal(self, user: str, summary: str, importance: float):
        if not summary:
            return
        if self.is_opted_out(user):
            logging.info(f"[RAG:DB] {user}은(는) 기억 저장을 거부한 사용자라 대화를 저장하지 않아요.")
            return

        date = datetime.now().isoformat()
        forgettable = importance < config.rag_forgettable_importance # 기준 중요도 미만은 잊어버릴 수 있는 기억으로 처리
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
        INSERT INTO chat_metadata (id, username, summary, importance, forgettable, timestamp, last_accessed)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (
            item["id"], item["username"], item["summary"],
            item["importance"], int(item["forgettable"]), item["timestamp"], item["timestamp"]
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

    def retrieve_similar_conversations(self, query: str, top_k: int | None = None):
        if top_k is None:
            top_k = config.rag_top_k
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
            if importance < config.rag_warn_importance:
                warn = "\n정확하지 않거나 중요하지 않은 내용이니 이 내용을 참고할 땐 조심스럽게 사용해줘."
            results.append(f"[{timestamp}] {summary}{warn}")
            self._boost_memory(idx)  # 실제로 사용된 기억은 더 오래 유지되도록 보정
        return results

    def _boost_memory(self, rowid):
        """검색에 사용된 기억의 중요도를 올리고 마지막 사용 시각을 갱신합니다. (자주 쓰는 기억은 잊히지 않게)"""
        boost = config.rag_retrieval_boost
        if boost <= 0:
            return
        cur = self.conn.cursor()
        cur.execute("""
        UPDATE chat_metadata
        SET importance = MIN(1.0, importance + ?),
            forgettable = CASE WHEN MIN(1.0, importance + ?) < ? THEN 1 ELSE 0 END,
            last_accessed = ?
        WHERE rowid = ?
        """, (boost, boost, config.rag_forgettable_importance, datetime.now().isoformat(), int(rowid)))
        self.conn.commit()

    def apply_forgetting(self) -> int:
        """마지막 사용 이후 경과 시간만큼 중요도를 감쇠시켜, 기준 미만으로 떨어진
        '잊어버릴 수 있는 기억(forgettable)'을 삭제합니다. 삭제한 개수를 반환합니다.

        감쇠는 저장된 중요도를 바꾸지 않고 계산 시점에만 적용되므로,
        몇 번을 호출해도 결과가 같습니다. (호출 주기에 따라 더 빨리 잊히지 않음)
        """
        decay = config.rag_forget_decay_per_day
        if decay <= 0:
            return 0
        cur = self.conn.cursor()
        cur.execute("""
        DELETE FROM chat_metadata
        WHERE forgettable = 1
          AND importance - ? * (julianday('now', 'localtime') - julianday(COALESCE(last_accessed, timestamp))) < ?
        """, (decay, config.rag_forget_threshold))
        deleted = cur.rowcount
        self.conn.commit()
        if deleted > 0:
            logging.info(f"[RAG:Forget] 오랫동안 사용되지 않은 기억 {deleted}개를 잊었어요.")
        return deleted

    def get_context(self, user_input: str) -> str:
        context_snippets = self.retrieve_similar_conversations(user_input)
        context = "\n---\n".join(context_snippets)
        return context.strip()

    def sync_all_metadata_to_faiss(self):
        """
        chat/slang DB의 모든 데이터를 timestamp 기준으로 정렬하여
        FAISS 인덱스를 새로 생성하고 전체 데이터를 추가합니다.
        """
        _, INDEX_FILE, DB_FILE = self._get_guild_paths()
        # 0. 인덱스를 새로 만들기 전에 오래되어 잊힐 기억을 먼저 정리
        self.apply_forgetting()
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

    # === 기억 관리 (/jiamemory) ===
    def count_memories(self) -> int:
        cur = self.conn.cursor()
        return cur.execute("SELECT COUNT(*) FROM chat_metadata").fetchone()[0]

    def list_memories(self, page: int = 1, page_size: int = 10) -> list[tuple]:
        """최신순으로 기억 목록을 (id, username, summary, importance, timestamp) 튜플로 반환합니다."""
        offset = max(page - 1, 0) * page_size
        cur = self.conn.cursor()
        cur.execute("""
        SELECT id, username, summary, importance, timestamp FROM chat_metadata
        ORDER BY timestamp DESC LIMIT ? OFFSET ?
        """, (page_size, offset))
        return cur.fetchall()

    def search_memories(self, query: str, top_k: int = 5) -> list[tuple]:
        """유사도 검색으로 기억을 (id, username, summary, importance, timestamp) 튜플로 반환합니다. (중요도 보정 없음)"""
        q_vec = self.embedding_model.encode([query], convert_to_numpy=True).astype('float32')
        q_vec = normalize(q_vec, axis=1)
        D, I = self.index.search(q_vec, top_k)
        cur = self.conn.cursor()
        results = []
        for score, idx in zip(D[0], I[0]):
            if idx == -1:
                continue
            cur.execute("SELECT id, username, summary, importance, timestamp FROM chat_metadata WHERE rowid = ?", (int(idx),))
            row = cur.fetchone()
            if row:
                results.append(row)
        return results

    def delete_memory(self, id_prefix: str) -> int:
        """id가 주어진 접두사로 시작하는 기억을 삭제하고, 삭제한 개수를 반환합니다.
        삭제 후에는 sync_all_metadata_to_faiss()로 인덱스를 다시 맞춰야 합니다."""
        if not id_prefix:
            return 0
        cur = self.conn.cursor()
        cur.execute("DELETE FROM chat_metadata WHERE id LIKE ?", (id_prefix + "%",))
        deleted = cur.rowcount
        self.conn.commit()
        return deleted

    # === 사용자별 프로필 ===
    def add_profile_fact(self, username: str, fact: str):
        """사용자에 대해 새로 알게 된 사실을 프로필에 추가합니다. (중복은 무시, 최대 개수 초과 시 오래된 것부터 삭제)"""
        fact = (fact or "").strip()
        if not fact or self.is_opted_out(username):
            return
        cur = self.conn.cursor()
        exists = cur.execute(
            "SELECT 1 FROM user_profiles WHERE username = ? AND fact = ?", (username, fact)
        ).fetchone()
        if exists:
            return
        cur.execute(
            "INSERT INTO user_profiles (id, username, fact, timestamp) VALUES (?, ?, ?, ?)",
            (str(uuid.uuid4()), username, fact, datetime.now().isoformat())
        )
        # 최대 개수 초과 시 오래된 사실부터 삭제
        cur.execute("""
        DELETE FROM user_profiles WHERE username = ? AND id NOT IN (
            SELECT id FROM user_profiles WHERE username = ? ORDER BY timestamp DESC LIMIT ?
        )""", (username, username, config.rag_profile_max_facts))
        self.conn.commit()
        logging.info(f"[RAG:Profile] {username}의 프로필에 새 사실을 기억했어요: {fact}")

    def get_profile_facts(self, username: str) -> list[str]:
        """사용자에 대해 기억하고 있는 사실 목록을 오래된 순으로 반환합니다."""
        cur = self.conn.cursor()
        cur.execute("SELECT fact FROM user_profiles WHERE username = ? ORDER BY timestamp ASC", (username,))
        return [row[0] for row in cur.fetchall()]

    def delete_profile(self, username: str) -> int:
        """사용자의 프로필 사실을 모두 삭제하고, 삭제한 개수를 반환합니다."""
        cur = self.conn.cursor()
        cur.execute("DELETE FROM user_profiles WHERE username = ?", (username,))
        deleted = cur.rowcount
        self.conn.commit()
        return deleted

    # === 기억 사용 거부 (opt-out) ===
    def is_opted_out(self, username: str) -> bool:
        cur = self.conn.cursor()
        return cur.execute("SELECT 1 FROM memory_optout WHERE username = ?", (username,)).fetchone() is not None

    def set_optout(self, username: str, opted_out: bool):
        """사용자의 기억 시스템 사용 거부 여부를 설정합니다.
        거부 시 기존 프로필과 해당 사용자 단독 명의의 대화 기록도 함께 삭제합니다."""
        cur = self.conn.cursor()
        if opted_out:
            cur.execute(
                "INSERT OR IGNORE INTO memory_optout (username, timestamp) VALUES (?, ?)",
                (username, datetime.now().isoformat())
            )
            cur.execute("DELETE FROM user_profiles WHERE username = ?", (username,))
            # 여러 화자가 함께 저장된 기록(쉼표로 묶인 이름)은 다른 사람의 기억이기도 하므로 남겨둠
            cur.execute("DELETE FROM chat_metadata WHERE username = ?", (username,))
        else:
            cur.execute("DELETE FROM memory_optout WHERE username = ?", (username,))
        self.conn.commit()
        self.save_all()

def save_conversation(user: str, guild_id: int, summary: str, importance: float):
    """
    대화 내용을 RAG 시스템에 저장하기 위한 외부 호출용 함수입니다.
    내부적으로 길드별 RAG 인스턴스를 관리합니다.
    """
    rag_instance = get_rag_instance(guild_id)
    rag_instance._save_conversation_internal(user=user, summary=summary, importance=importance)