from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_name: str = "bmo-rag"
    log_level: str = "INFO"
    llm_model: str = "gpt-4.1-mini"
    embedding_model: str = "BAAI/bge-m3"
    vector_store: str = "qdrant"
    top_k: int = 5

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )
