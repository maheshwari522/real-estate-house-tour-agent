rom pydantic import BaseSettings

class Settings(BaseSettings):
    OPENAI_API_KEY: str
    DATABASE_URL: str = "sqlite:///./test.db"  # Use Postgres later

    class Config:
        env_file = ".env"

settings = Settings()