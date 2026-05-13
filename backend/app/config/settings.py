import os
from dotenv import load_dotenv
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent.parent  # → backend/
load_dotenv(BASE_DIR / ".env")


class Settings:
    DATABASE_URL: str       = os.getenv("POSTGRE_URL", "")
    JWT_SECRET: str         = os.getenv("JWT_SECRET", "changethis")
    JWT_ALGORITHM: str      = os.getenv("JWT_ALGORITHM", "HS256")
    JWT_EXPIRE_MINUTES: int = int(os.getenv("JWT_EXPIRE_MINUTES", "60"))
    STORAGE_PATH: str       = os.getenv("STORAGE_PATH", "./storage/documents")
    CHROMA_PERSIST_DIR: str = os.getenv("CHROMA_PERSIST_DIR", "./chroma_storage")
    RAG_PROJECT_ROOT: str   = os.getenv("RAG_PROJECT_ROOT", "")
    LLM_BACKEND: str        = os.getenv("LLM_BACKEND", "groq")
    GROQ_API_KEY: str       = os.getenv("GROQ_API_KEY", "")

    def validate(self):
        errors = []
        if not self.DATABASE_URL:
            errors.append("DATABASE_URL is not set — check your .env file")
        if self.JWT_SECRET == "changethis":
            errors.append(
                "JWT_SECRET is still the default — "
                "run: openssl rand -hex 32  and paste the result in .env"
            )
        if errors:
            raise RuntimeError(
                "\n\n❌ Missing or invalid environment variables:\n  - "
                + "\n  - ".join(errors)
                + "\n\nCheck backend/.env"
            )
        print("✅ Settings loaded successfully")
        print(f"DATABASE_URL      : {self.DATABASE_URL[:40]}...")
        print(f"JWT_ALGORITHM     : {self.JWT_ALGORITHM}")
        print(f"JWT_EXPIRE_MINUTES: {self.JWT_EXPIRE_MINUTES}")
        print(f"STORAGE_PATH      : {self.STORAGE_PATH}")
        print(f"CHROMA_PERSIST_DIR: {self.CHROMA_PERSIST_DIR}")
        print(f"LLM_BACKEND       : {self.LLM_BACKEND}")


settings = Settings()
