import os
from pathlib import Path
BASE_DIR = Path(__file__).resolve().parent
class Config:
    SECRET_KEY = os.environ.get("SECRET_KEY", "change-this-local-development-key")
    MAX_CONTENT_LENGTH = 120 * 1024 * 1024
    UPLOAD_FOLDER = str(BASE_DIR / "uploads")
    ARTIFACT_FOLDER = str(BASE_DIR / "artifacts")
    ALLOWED_EXTENSIONS = {"csv"}
