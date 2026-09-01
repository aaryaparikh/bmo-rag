from bmo_rag.config import Settings


def test_default_settings() -> None:
    settings = Settings()

    assert settings.app_name == "bmo-rag"
    assert settings.top_k == 5
