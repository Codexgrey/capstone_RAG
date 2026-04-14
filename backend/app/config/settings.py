# settings.py keeps all env vars.
# Other files import from here,
# so config stays in one place.

import os
from dotenv import load_dotenv
from pathlib import Path

# ── Load .env from backend/ root (one level above app/) ───────────────────────
# This works regardless of where you run the server from
BASE_DIR = Path(__file__).resolve().parent.parent.parent  # → backend/
load_dotenv(BASE_DIR / ".env")


class Settings:
    DATABASE_URL: str = os.getenv("POSTGRE_URL", "")

    JWT_SECRET: str = os.getenv("JWT_SECRET", "changethis")
    JWT_ALGORITHM: str = os.getenv("JWT_ALGORITHM", "HS256")
    JWT_EXPIRE_MINUTES: int = int(os.getenv("JWT_EXPIRE_MINUTES", "60"))

    # Validation on startup
    def validate(self):
        # Called once from main.py on startup.
        # Raises clear errors if critical variables are missing.
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

        # if it reaches here just to cinform everything is running
        print("✅ Settings loaded successfully")
        print(f"DATABASE_URL : {self.DATABASE_URL[:40]}...")
        print(f"JWT_ALGORITHM : {self.JWT_ALGORITHM}")
        print(f"JWT_EXPIRE_MINUTES: {self.JWT_EXPIRE_MINUTES}")



# Single instance imported everywhere ───────────────────────────────────────
settings = Settings()