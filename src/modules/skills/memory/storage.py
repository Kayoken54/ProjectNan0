from typing import Optional, Dict

from src.utils.logger import get_logger

logger = get_logger("bea.skills.memory.storage")


class MemoryStorage:

    def __init__(
        self,
        db_path: str,
        embedding_mode: str = "local",
        embedding_model: str = "all-MiniLM-L6-v2",
        openai_key: Optional[str] = None
    ):
        self.db_path = db_path
        self.embedding_mode = embedding_mode
        self.embedding_model = embedding_model
        self.openai_key = openai_key

        self.chroma_client = None
        self.collection = None
        self.embedding_function = None
        self.initialized = False


    @staticmethod
    def _load_chromadb():
        import chromadb
        from chromadb.utils import embedding_functions

        return chromadb, embedding_functions

    def initialize(self) -> bool:

        if self.initialized and self.collection is not None:
            return True

        try:
            logger.info(
                f"MemoryStorage: Initializing ChromaDB at {self.db_path}..."
            )

            chromadb, embedding_functions = self._load_chromadb()

            self.chroma_client = chromadb.PersistentClient(
                path=self.db_path
            )

            if self.embedding_mode == "local":

                self.embedding_function = (
                    embedding_functions.SentenceTransformerEmbeddingFunction(
                        model_name=self.embedding_model
                    )
                )

                logger.info(
                    f"MemoryStorage: Using local embeddings: {self.embedding_model}"
                )

            else:

                if not self.openai_key:

                    logger.error(
                        "MemoryStorage: OpenAI embedding mode selected, but no API key found."
                    )

                    return False

                self.embedding_function = (
                    embedding_functions.OpenAIEmbeddingFunction(
                        api_key=self.openai_key,
                        model_name=self.embedding_model
                    )
                )

                logger.info(
                    f"MemoryStorage: Using OpenAI embeddings: {self.embedding_model}"
                )

            self.collection = self.chroma_client.get_or_create_collection(
                name="bea_diary",
                embedding_function=self.embedding_function,
                metadata={"hnsw:space": "cosine"}
            )

            logger.info(
                f"MemoryStorage: ChromaDB initialized. Count: {self.collection.count()}"
            )

            self.initialized = True
            return True

        except Exception as e:

            logger.error(
                f"MemoryStorage: Error initializing: {e}"
            )

            return False

    def add_entry(
        self,
        content: str,
        metadata: Dict,
        entry_id: str
    ):

        if not self.collection:
            return

        self.collection.add(
            documents=[content],
            metadatas=[metadata],
            ids=[entry_id]
        )

    def query_similar(
        self,
        query: str,
        limit: int = 3
    ):

        if not self.collection:
            return None

        count = self.collection.count()
        if count <= 0:
            return None

        return self.collection.query(
            query_texts=[query],
            n_results=min(max(1, int(limit)), count),
            include=[
                "documents",
                "metadatas",
                "distances"
            ]
        )

    def entry_exists(
        self,
        entry_id: str
    ) -> bool:

        if not self.collection:
            return False

        try:

            result = self.collection.get(
                ids=[entry_id]
            )

            return len(result["ids"]) > 0

        except Exception:

            return False
