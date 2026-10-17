from fastapi import APIRouter, Depends

from app.config import Settings
from app.dependencies import get_ocr, get_settings_dep
from app.schemas.common import HealthResponse
from app.services.ocr_service import OCRService

router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthResponse)
def health(settings: Settings = Depends(get_settings_dep), ocr: OCRService = Depends(get_ocr)) -> HealthResponse:
    """Cheap liveness check. Does not load ML models or call Gemini."""
    return HealthResponse(
        status="ok", version=settings.app_version,
        gemini_configured=settings.gemini_configured, ocr_available=ocr.available(),
    )
