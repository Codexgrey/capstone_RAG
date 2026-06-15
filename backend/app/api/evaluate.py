"""
backend/app/api/evaluate.py
===========================
TriviaQA evaluation endpoints.

Registered in main.py as:
    from app.api import evaluate
    app.include_router(evaluate.router, prefix="/api")

Endpoints
---------
GET  /api/evaluate/triviaqa/stats
    Summary statistics about the 5,000-question bank.

GET  /api/evaluate/triviaqa/questions
    Paginated question list (no answers) for the frontend browser.
    Query params: n, domain, qtype, offset

GET  /api/evaluate/triviaqa/questions/{question_id}
    Single question lookup by ID.

POST /api/evaluate/triviaqa/score
    Score one predicted answer against TriviaQA ground truth.

POST /api/evaluate/triviaqa/run
    Run the full RAG pipeline on a subset of questions and score results.
    Supports filtering by domain, answer type, explicit IDs, or count.
"""

from typing import Any, Dict, List, Optional
from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from app.config.database import get_db
from app.config.dependencies import get_current_user
from app.models.db_models import User
from app.evaluation.triviaqa_evaluator import (
    TRIVIAQA_BANK,
    get_bank_stats,
    get_question_by_id,
    get_test_questions,
    score_single,
    run_triviaqa_batch,
)
from app.evaluation.triviaqa_precision_evaluator import (
    run_precision_batch,
    get_checkpoint_status,
    reset_checkpoint,
)

router = APIRouter(prefix="/evaluate", tags=["Evaluation"])


# ---------------------------------------------------------------------------
# Request schemas
# ---------------------------------------------------------------------------

class ScoreRequest(BaseModel):
    question_id:      str
    predicted_answer: str


class RunRequest(BaseModel):
    retrieval_method: str = "vector"
    top_k:            int = Field(default=5, ge=1, le=20)
    # Selection filters — use one of:
    question_ids:     Optional[List[str]] = None   # explicit list of IDs
    n:                Optional[int]       = Field(default=None, ge=1, le=5000)
    domain:           Optional[str]       = None   # "Wikipedia" | "Web"
    qtype:            Optional[str]       = None   # "WikipediaEntity"|"Numerical"|"FreeForm"


class PrecisionRunRequest(BaseModel):
    retrieval_method: str = "vector"
    top_k:            int = Field(default=5,   ge=1,  le=20)
    batch_size:       int = Field(default=200, ge=1,  le=500)
    chunk_size:       int = Field(default=200, ge=50, le=600)
    chunk_overlap:    int = Field(default=40,  ge=0,  le=200)


# ---------------------------------------------------------------------------
# GET /evaluate/triviaqa/stats
# ---------------------------------------------------------------------------

@router.get("/triviaqa/stats")
def triviaqa_stats(
    current_user: User = Depends(get_current_user),
):
    """Bank statistics — total questions, domain split, answer type distribution."""
    return get_bank_stats()


# ---------------------------------------------------------------------------
# GET /evaluate/triviaqa/questions
# ---------------------------------------------------------------------------

@router.get("/triviaqa/questions")
def list_triviaqa_questions(
    n:      Optional[int] = Query(default=50,  ge=1,  le=5000),
    offset: Optional[int] = Query(default=0,   ge=0),
    domain: Optional[str] = Query(default=None),
    qtype:  Optional[str] = Query(default=None),
    current_user: User = Depends(get_current_user),
):
    """
    Paginated list of TriviaQA questions (no answers).

    Use `n` + `offset` for pagination, `domain` and `qtype` for filtering.
    Safe to expose to the frontend — answers are not included.
    """
    all_qs = get_test_questions(domain=domain, qtype=qtype)
    page   = all_qs[offset: offset + n]

    return {
        "total":     len(all_qs),
        "returned":  len(page),
        "offset":    offset,
        "questions": page,
        "source":    "TriviaQA (Joshi et al., ACL 2017)",
    }


# ---------------------------------------------------------------------------
# GET /evaluate/triviaqa/questions/{question_id}
# ---------------------------------------------------------------------------

@router.get("/triviaqa/questions/{question_id}")
def get_triviaqa_question(
    question_id: str,
    current_user: User = Depends(get_current_user),
):
    """Look up a single question by ID. Returns the question without the answer."""
    item = get_question_by_id(question_id)
    if not item:
        raise HTTPException(
            status_code=404,
            detail=f"Question ID '{question_id}' not found in the TriviaQA bank.",
        )
    return {
        "question_id":  item["QuestionId"],
        "question":     item["Question"],
        "domain":       item["Domain"],
        "answer_type":  item["Answer"].get("Type", ""),
        "alias_count":  len(item["Answer"]["NormalizedAliases"]),
    }


# ---------------------------------------------------------------------------
# POST /evaluate/triviaqa/score
# ---------------------------------------------------------------------------

@router.post("/triviaqa/score")
def score_single_answer(
    payload: ScoreRequest,
    current_user: User = Depends(get_current_user),
):
    """
    Score one predicted answer against TriviaQA ground truth.

    Returns exact_match (0.0 or 1.0) and F1 (0.0–1.0) for the question,
    plus the canonical answer value and all valid aliases.
    """
    item = get_question_by_id(payload.question_id)
    if not item:
        raise HTTPException(
            status_code=404,
            detail=f"Question ID '{payload.question_id}' not found in test bank.",
        )

    scores = score_single(payload.predicted_answer, item["Answer"])
    return {
        "question_id":      item["QuestionId"],
        "question":         item["Question"],
        "predicted_answer": payload.predicted_answer,
        "ground_truth":     item["Answer"]["Value"],
        "aliases":          item["Answer"]["NormalizedAliases"],
        "exact_match":      scores["exact_match"],
        "f1":               scores["f1"],
        "note": (
            "Scoring is max-over-aliases after normalisation "
            "(lowercase, remove articles and punctuation)."
        ),
    }


# ---------------------------------------------------------------------------
# POST /evaluate/triviaqa/run
# ---------------------------------------------------------------------------

@router.post("/triviaqa/run")
def run_triviaqa_evaluation(
    payload: RunRequest,
    db:           Session = Depends(get_db),
    current_user: User    = Depends(get_current_user),
):
    """
    Run the full RAG pipeline on a subset of TriviaQA questions and score.

    For each question the backend:
      1. Sends it through the live retrieval pipeline.
      2. Generates an answer via the LLM.
      3. Scores with official TriviaQA EM and F1 (max over all valid aliases).

    Selection options (mutually exclusive; `question_ids` takes priority):
      - question_ids : explicit list of QuestionIds
      - n + domain + qtype : filter and cap

    Returns per-question breakdown and aggregate averages.

    ⚠️  Running large batches (n > 100) is slow due to LLM API rate limits.
        For a full 5,000-question run, use an async job rather than this endpoint.
        Recommended batch size for interactive use: n ≤ 50.
    """
    from app.services.rag_service import handle_query

    def _rag_query(question: str, retrieval_method: str, top_k: int) -> str:
        result = handle_query(
            db               = db,
            user_id          = current_user.id,
            question         = question,
            session_id       = None,
            retrieval_method = retrieval_method,
            top_k            = top_k,
            persist          = False,
        )
        return result.get("answer", "")

    batch = run_triviaqa_batch(
        retrieval_method = payload.retrieval_method,
        top_k            = payload.top_k,
        question_ids     = payload.question_ids,
        n                = payload.n,
        domain           = payload.domain,
        qtype            = payload.qtype,
        rag_query_fn     = _rag_query,
    )

    return {
        **batch,
        "scoring_protocol": "TriviaQA official EM + F1 (Joshi et al., ACL 2017)",
        "note": (
            "EM = 1.0 when the prediction exactly matches any normalised alias. "
            "F1 measures token overlap with the best-matching alias."
        ),
    }


# ---------------------------------------------------------------------------
# GET /evaluate/triviaqa-precision/status
# ---------------------------------------------------------------------------

@router.get("/triviaqa-precision/status")
def triviaqa_precision_status(
    retrieval_method: str = Query(default="vector"),
    current_user: User = Depends(get_current_user),
):
    """
    Current resume position and cumulative metrics for a method's
    Precision/Recall benchmark, without running any questions.

    Use this to populate the panel on load and after each run, so the
    frontend can show "N questions done so far — click Run to continue".
    """
    if retrieval_method not in ("vector", "keyword", "hybrid"):
        raise HTTPException(
            status_code=400,
            detail="retrieval_method must be one of: vector, keyword, hybrid",
        )
    return get_checkpoint_status(retrieval_method)


# ---------------------------------------------------------------------------
# POST /evaluate/triviaqa-precision/reset
# ---------------------------------------------------------------------------

@router.post("/triviaqa-precision/reset")
def triviaqa_precision_reset(
    retrieval_method: str = Query(default="vector"),
    current_user: User = Depends(get_current_user),
):
    """Discard the saved checkpoint for a method, starting that benchmark
    over from question 0."""
    if retrieval_method not in ("vector", "keyword", "hybrid"):
        raise HTTPException(
            status_code=400,
            detail="retrieval_method must be one of: vector, keyword, hybrid",
        )
    reset_checkpoint(retrieval_method)
    return get_checkpoint_status(retrieval_method)


# ---------------------------------------------------------------------------
# POST /evaluate/triviaqa-precision/run
# ---------------------------------------------------------------------------

@router.post("/triviaqa-precision/run")
def run_triviaqa_precision_evaluation(
    payload: PrecisionRunRequest,
    current_user: User = Depends(get_current_user),
):
    """
    Run the next batch of the Precision/Recall benchmark, resuming from
    the saved checkpoint for retrieval_method.

    For each question in the batch:
      1. Build chunks from that question's own TriviaQA evidence
         (search_results, falling back to entity_pages).
      2. Ingest into the chosen retrieval method's index in isolation.
      3. Retrieve top_k chunks and score both:
           - Answer-in-Context@k / Answer-in-Top-1 / MRR
           - Precision@k / Recall@k against the question's own evidence
      4. Remove the question's evidence from the index before the next
         question starts.

    ⚠️  This is isolated per-question evaluation — each question re-builds
        a small index from scratch. A batch of 200 questions takes roughly
        the same order of magnitude as the equivalent standalone benchmark
        run (minutes, not seconds). Streaming starts from question 0 on
        first use and resumes from the checkpoint on subsequent calls.
    """
    if payload.retrieval_method not in ("vector", "keyword", "hybrid"):
        raise HTTPException(
            status_code=400,
            detail="retrieval_method must be one of: vector, keyword, hybrid",
        )

    try:
        result = run_precision_batch(
            retrieval_method = payload.retrieval_method,
            top_k            = payload.top_k,
            batch_size       = payload.batch_size,
            chunk_size       = payload.chunk_size,
            chunk_overlap    = payload.chunk_overlap,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))

    return {
        **result,
        "scoring_protocol": (
            "Isolated per-question evaluation using TriviaQA's own evidence "
            "as the gold document (search_results, with entity_pages as "
            "fallback). Precision@k/Recall@k measure retrieved-chunk "
            "membership in that gold set; Answer-in-Context@k/Top-1/MRR "
            "measure whether a correct answer alias appears in the "
            "retrieved text."
        ),
    }
