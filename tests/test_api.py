from io import BytesIO

from fastapi.testclient import TestClient
from PIL import Image

from api.app import app


client = TestClient(app)


def test_analyze_endpoint() -> None:
    image = Image.new("RGB", (64, 64), color=(155, 20, 20))
    buf = BytesIO()
    image.save(buf, format="PNG")
    buf.seek(0)

    response = client.post(
        "/analyze",
        data={"clinical_text": "Patient has productive cough and fever."},
        files={"image": ("xray.png", buf.read(), "image/png")},
    )

    assert response.status_code == 200
    payload = response.json()
    assert "diagnosis_prediction" in payload
    assert "anomaly_score" in payload
    assert "generated_report" in payload
