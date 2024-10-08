# biabot/config.py

from dataclasses import dataclass, field
import os
from typing import List, Optional

from dotenv import load_dotenv

load_dotenv()

ENV_CONFIG = {
    "api_key": os.getenv("API_KEY", None),
    "conversation_id": os.getenv("CONVERSATION_ID", None),
    "device": os.getenv("DEVICE", "cpu"),
    "documents_dir": os.getenv("DOCUMENTS_DIR", "documents"),
    "guidance": os.getenv("GUIDANCE", None),
    "lancedb_uri": os.getenv("LANCEDB_URI", "lancedb"),
    "llm_provider": os.getenv("LLM_PROVIDER", "openai"),
    "max_tokens": int(os.getenv("MAX_TOKENS", 256)),
    "memory_window": int(os.getenv("MEMORY_WINDOW", 8)),
    "model_url": os.getenv("MODEL_URL", None),
    "model_name": os.getenv("MODEL_NAME", None),
    "persona_id": os.getenv("PERSONA_ID", "assistant"),
    "persona_location": os.getenv("PERSONA_LOCATION", None),
    "persona_mood": os.getenv("PERSONA_MOOD", "Inquisitive"),
    "persona_path": os.getenv("PERSONA_PATH", "profiles"),
    "recall_size": int(os.getenv("RECALL_SIZE", 3)),
    "temperature": float(os.getenv("TEMPERATURE", 0.7)),
    "top_n": int(os.getenv("TOP_N", 3)),
    "user_id": os.getenv("USER_ID", "user"),
    "workdir_folder": os.getenv("OUTPUT_FOLDER", "export"),
}


@dataclass
class ChatConfig:
    device: str = "cpu"
    lancedb_uri: str = "lancedb"
    persona_path: str = "profiles"
    workdir_folder: str = "export"
    documents_dir: Optional[str] = None
    user_id: str = "user"
    persona_id: str = "assistant"
    conversation_id: Optional[str] = None
    llm_provider: str = "openai"
    model_url: Optional[str] = None
    model_name: Optional[str] = None
    api_key: str = ""
    max_tokens: int = 512
    temperature: float = 0.7
    top_n: int = 3
    recall_size: int = 3
    memory_window: int = 8
    generations: int = 1
    presence: Optional[float] = None
    repetition: Optional[float] = None
    system_message: Optional[str] = None
    persona_location: Optional[str] = None
    persona_mood: str = "Inquisitive"
    debug: bool = False
    no_retry: bool = False
    guidance: Optional[str] = None
    stop_sequences: List[str] = field(
        default_factory=lambda: ["You:", "<|im_end|>", "</s>"]
    )

    @classmethod
    def from_env(cls):
        return cls(**ENV_CONFIG)
