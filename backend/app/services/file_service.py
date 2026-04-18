# file_service.py — handles saving uploads to disk
# storage path: /storage/documents/{user_id}/{filename}
# keeps each user’s files in their own folder
# backend stores full filepath in Postgres for lookup
# files are private — only served via authenticated endpoints
import uuid
import os
from pathlib import Path
from fastapi import UploadFile
from app.config.settings import settings


def get_user_storage_path(user_id: str) -> Path:
    # Returns the storage directory for a specific user.
    # Creates the directory if it doesn't exist yet.
    user_dir = Path(settings.STORAGE_PATH) / str(user_id)
    user_dir.mkdir(parents=True, exist_ok=True)
    return user_dir


async def save_upload_file(file: UploadFile, user_id: str) -> dict:
    # Save an uploaded file to the user's storage directory.
    # Adds a UUID prefix to the filename to prevent collisions
    # if the same filename is uploaded twice.
    from app.ingestion.parser import get_file_type

    # Validate file type before saving
    file_type = get_file_type(file.filename)

    # Add UUID prefix to avoid filename collisions
    # e.g. "report.pdf" → "a1b2c3d4_report.pdf"
    unique_prefix = uuid.uuid4().hex[:8]
    safe_name = f"{unique_prefix}_{file.filename}"

    # Build full save path
    user_dir = get_user_storage_path(user_id)
    save_path = user_dir / safe_name

    # Write file to disk
    contents = await file.read()
    with open(save_path, "wb") as f:
        f.write(contents)

    print(f"✅ File saved: {save_path}")

    return {
        "filepath":  str(save_path),
        "filename":  file.filename,
        "file_type": file_type,
        "safe_name": safe_name,
    }


def delete_file(filepath: str) -> bool:
    # Delete a file from disk.
    # Called if document ingestion fails after saving,
    # so we don't leave orphaned files on the server
    path = Path(filepath)
    if path.exists():
        path.unlink()
        print(f"🗑️  File deleted: {filepath}")
        return True
    return False


def get_file_size_kb(filepath: str) -> float:
    # Return the file size in kilobytes.
    return round(os.path.getsize(filepath) / 1024, 2)