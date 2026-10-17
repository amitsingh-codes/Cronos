from fastapi import APIRouter, Depends

from app.dependencies import get_question_service
from app.schemas.common import ErrorResponse
from app.schemas.questions import QuestionPaperResponse, QuestionRequest
from app.services.question_service import QuestionService

router = APIRouter(prefix="/api/questions", tags=["questions"])


@router.post(
    "/generate", response_model=QuestionPaperResponse,
    responses={422: {"model": ErrorResponse}, 502: {"model": ErrorResponse}, 503: {"model": ErrorResponse}},
)
async def generate(req: QuestionRequest, svc: QuestionService = Depends(get_question_service)) -> QuestionPaperResponse:
    """Generate a validated question paper (JSON body) from the supplied source text."""
    return await svc.generate(req)
