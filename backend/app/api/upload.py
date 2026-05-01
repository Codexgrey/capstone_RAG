import os
from email.mime import text
import uuid
from datetime import datetime
from fastapi import APIRouter, Depends, UploadFile, File, HTTPException, status, BackgroundTasks
from sqlalchemy.orm import Session

from app.config.database import get_db
from app.config.dependencies import get_current_user
from app.models.db_models import User, Document, DocumentChunk, Log
from app.services.file_service import save_upload_file, delete_file
from app.ingestion.parser import extract_text
from app.ingestion.chunker import chunk_text
from app.ingestion.indexer import index_chunks

router = APIRouter(tags=["Documents"])


@router.post("/upload", status_code=status.HTTP_202_ACCEPTED)
async def upload_document(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    # 1: Generate shared document_id ─
    document_id = f"doc_{uuid.uuid4().hex[:12]}"

    # 2: Save file to disk ─
    try:
        file_info = await save_upload_file(file, user_id=str(current_user.id))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    # 3: Save metadata to PostgreSQL (status=processing) ─
    doc = Document(
        document_id = document_id,
        filename = file_info["filename"],
        filepath = file_info["filepath"],
        file_type = file_info["file_type"],
        uploaded_by = current_user.id,
        upload_date = datetime.utcnow(),
        status = "processing"
    )
    db.add(doc)
    db.add(Log(
        user_id = current_user.id,
        action = "document_upload_started",
        detail = f"file={file_info['filename']} doc_id={document_id}",
        timestamp = datetime.utcnow()
    ))
    db.commit()

    # 4: Queue background processing ─
    background_tasks.add_task(
        process_document_background,
        document_id = document_id,
        filepath = file_info["filepath"],
        file_type = file_info["file_type"],
        filename = file_info["filename"],
        user_id = current_user.id,
        username = current_user.username
    )

    #  Return immediately — don't wait for processing 
    return {
        "message" : "Document uploaded. Processing in background.",
        "document_id" : document_id,
        "filename" : file_info["filename"],
        "status" : "processing"
    }


async def process_document_background(
    document_id: str,
    filepath: str,
    file_type: str,
    filename: str,
    user_id: int,
    username: str
):
    from app.config.database import SessionLocal
    db = SessionLocal()

    try:
        #  Extract text 
        text = extract_text(filepath, file_type)
        print(f"  ✅ Text extracted from {filename}")

        #  Chunk the text 
        chunks = chunk_text(
            text = text,
            document_id = document_id,
            source_name = filename
        )
        print(f"  ✅ {len(chunks)} chunks created")

        # Index chunks in ChromaDB 
        index_chunks(                                  
            chunks      = chunks,
            document_id = document_id,
            uploaded_by = username,
            file_type   = file_type
        )
        print(f"  ✅ Chunks indexed in ChromaDB")

        #  Save chunk metadata to PostgreSQL 
        for chunk in chunks:
            db.add(DocumentChunk(
                chunk_id = chunk["chunk_id"],
                document_id = document_id,
                source_name = filename,
                text = chunk["text"],
                page = chunk.get("page", 1),
                start_char = chunk.get("start_char", 0),
                end_char = chunk.get("end_char", 0),
            ))

        #  Mark document as completed 
        doc = db.query(Document).filter(
            Document.document_id == document_id
        ).first()
        if doc:
            doc.status = "completed"

        db.add(Log(
            user_id = user_id,
            action = "document_uploaded",
            detail = f"file={filename} chunks={len(chunks)} doc_id={document_id}",
            timestamp = datetime.utcnow()
        ))
        db.commit()
        print(f"  ✅ Document {document_id} processing complete")

        # ── Ingest into all three retrieval indexes ───────────────────────────
        #
        # Resolve the file path to pass to each module:
        #   - PDFs are saved as OCR .txt for Collins (he reads .txt natively)
        #   - Olivier and Nathan read the original file directly (support .pdf)
        #
        abs_filepath = os.path.abspath(filepath)

        # For Collins: save OCR text as .txt so his loader picks it up cleanly
        if text and file_type == "pdf":
            txt_path = abs_filepath.replace(".pdf", "_ocr.txt")
            try:
                with open(txt_path, "w", encoding="utf-8") as f:
                    f.write(text)
                collins_file_path = txt_path
            except Exception as e:
                print(f"  ⚠️  Could not save OCR txt: {e}")
                collins_file_path = abs_filepath
        else:
            collins_file_path = abs_filepath

        # 1. Collins — FAISS vector index
        try:
            from app.retrieval.vector_adapter import ingest as collins_ingest
            result = collins_ingest(file_paths=[collins_file_path])
            if result.get("status") == "ok":
                print(f"  ✅ Collins FAISS: {result.get('total_chunks')} total chunks")
            else:
                print(f"  ⚠️  Collins ingest: {result.get('error')}")
        except Exception as e:
            print(f"  ⚠️  Collins ingest failed: {e}")

        # 2. Olivier — BM25 keyword index
        try:
            from app.retrieval.keyword_adapter import ingest as olivier_ingest
            result = olivier_ingest(file_paths=[abs_filepath])
            if result.get("status") == "ok":
                print(f"  ✅ Olivier BM25: {result.get('total_chunks')} total chunks")
            else:
                print(f"  ⚠️  Olivier ingest: {result.get('error')}")
        except Exception as e:
            print(f"  ⚠️  Olivier ingest failed: {e}")

        # 3. Nathan — Hybrid FAISS+BM25+RRF index
        try:
            from app.retrieval.hybrid_adapter import ingest as nathan_ingest
            result = nathan_ingest(file_paths=[abs_filepath])
            if result.get("status") == "ok":
                print(f"  ✅ Nathan hybrid: {result.get('total_chunks')} total chunks")
            else:
                print(f"  ⚠️  Nathan ingest: {result.get('error')}")
        except Exception as e:
            print(f"  ⚠️  Nathan ingest failed: {e}")
    except Exception as e:
        print(f"  ❌ Failed: {e}")
        try:
            doc = db.query(Document).filter(
                Document.document_id == document_id
            ).first()
            if doc:
                doc.status = "failed"
            db.add(Log(
                user_id = user_id,
                action = "document_upload_failed",
                detail = f"file={filename} error={str(e)}",
                timestamp = datetime.utcnow()
            ))
            db.commit()
        except:
            pass

    finally:
        db.close()


@router.get("/documents", status_code=status.HTTP_200_OK)
def list_documents(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """List all documents uploaded by the current user."""
    docs = db.query(Document)\
             .filter(Document.uploaded_by == current_user.id)\
             .order_by(Document.upload_date.desc())\
             .all()

    return {
        "documents": [
            {
                "document_id" : d.document_id,
                "filename" : d.filename,
                "file_type" : d.file_type,
                "status" : d.status,
                "upload_date" : d.upload_date,
            }
            for d in docs
        ],
        "total": len(docs)
    }


@router.get("/document/{document_id}/status")
def get_document_status(
    document_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Check the processing status of a specific document."""
    doc = db.query(Document).filter(
        Document.document_id == document_id,
        Document.uploaded_by == current_user.id
    ).first()

    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")

    return {
        "document_id" : doc.document_id,
        "filename" : doc.filename,
        "status" : doc.status,
        "upload_date" : doc.upload_date
    }