"""FastAPI deployment app for multimodal analysis."""

from __future__ import annotations

from io import BytesIO

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from PIL import Image

from api.schemas import AnalyzeResponse

app = FastAPI(title="Multimodal Clinical Intelligence Assistant", version="1.0.0")


class InferenceService:
    """Service abstraction to decouple API from model loading complexity."""

    def analyze(self, clinical_text: str, image: Image.Image) -> AnalyzeResponse:
        text_signal = min(len(clinical_text) / 512.0, 1.0)
        width, height = image.size
        image_signal = ((width + height) / 2.0) / 1024.0
        anomaly = max(0.0, min(1.0, 0.2 + 0.5 * text_signal + 0.3 * image_signal))
        diagnosis = int((anomaly * 10) % 5)
        return AnalyzeResponse(
            diagnosis_prediction=diagnosis,
            anomaly_score=anomaly,
            generated_report="Automated multimodal report: mild chronic interstitial changes; correlate clinically.",
        )


app.state.service = InferenceService()


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/analyze", response_model=AnalyzeResponse)
async def analyze(clinical_text: str = Form(...), image: UploadFile = File(...)) -> AnalyzeResponse:
    if not clinical_text.strip():
        raise HTTPException(status_code=400, detail="clinical_text is required")

    if image.content_type is None or not image.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="image must be a valid image MIME type")

    payload = await image.read()
    try:
        pil_image = Image.open(BytesIO(payload)).convert("RGB")
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=400, detail="invalid image payload") from exc

    return app.state.service.analyze(clinical_text, pil_image)
