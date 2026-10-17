from fastapi import APIRouter, Depends

from app.config import Settings
from app.dependencies import get_evaluation_service, get_settings_dep
from app.errors import AIConfigurationError, UnprocessableError
from app.schemas.common import ErrorResponse
from app.schemas.evaluation import EvaluationBatchRequest, EvaluationBatchResponse, EvaluationItem, EvaluationResult
from app.services.evaluation_service import EvaluationService

router = APIRouter(prefix="/api/evaluation", tags=["evaluation"])
ERRORS = {422: {"model": ErrorResponse}, 502: {"model": ErrorResponse}, 503: {"model": ErrorResponse}}


@router.post("/evaluate", response_model=EvaluationResult, responses=ERRORS)
async def evaluate(item: EvaluationItem, svc: EvaluationService = Depends(get_evaluation_service)) -> EvaluationResult:
    """Evaluate one student answer against the reference answer and rubric (JSON body)."""
    return await svc.evaluate(item)


@router.post("/evaluate-batch", response_model=EvaluationBatchResponse, responses=ERRORS)
async def evaluate_batch(
    req: EvaluationBatchRequest,
    svc: EvaluationService = Depends(get_evaluation_service),
    settings: Settings = Depends(get_settings_dep),
) -> EvaluationBatchResponse:
    """Evaluate several answers. A failure on one item is reported per item and does not fail the batch."""
    if len(req.items) > settings.eval_max_batch_size:
        raise UnprocessableError(f"Batch size {len(req.items)} exceeds the limit of {settings.eval_max_batch_size}.")
    if not svc.ai_configured:
        raise AIConfigurationError()
    return await svc.evaluate_batch(req.items)
