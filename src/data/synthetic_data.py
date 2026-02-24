"""Synthetic dataset generator for local development and CI smoke tests."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image


def generate_synthetic_dataset(output_dir: str = "data/processed", num_samples: int = 32) -> str:
    output = Path(output_dir)
    images_dir = output / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    records = []
    for index in range(num_samples):
        image_path = images_dir / f"sample_{index}.png"
        pixels = np.random.randint(0, 255, size=(224, 224, 3), dtype=np.uint8)
        Image.fromarray(pixels).save(image_path)

        records.append(
            {
                "patient_id": f"P-{index:04d}",
                "clinical_text": "Patient with chest discomfort and mild dyspnea.",
                "image_path": str(image_path),
                "label": int(index % 5),
                "anomaly_score": float(np.random.rand()),
                "report": "No acute cardiopulmonary findings."
            }
        )

    manifest = pd.DataFrame.from_records(records)
    csv_path = output / "train.csv"
    manifest.to_csv(csv_path, index=False)
    manifest.sample(frac=0.2, random_state=42).to_csv(output / "val.csv", index=False)
    manifest.sample(frac=0.2, random_state=7).to_csv(output / "test.csv", index=False)
    return str(csv_path)
