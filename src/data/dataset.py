"""PyTorch dataset for multimodal clinical samples."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from transformers import AutoTokenizer


@dataclass
class DatasetConfig:
    text_model_name: str
    text_max_length: int
    image_size: int


class MultimodalClinicalDataset(Dataset):
    """Dataset returning tokenized text, image tensor, and task labels."""

    def __init__(self, csv_path: str, config: DatasetConfig) -> None:
        self.frame = pd.read_csv(csv_path)
        self.tokenizer = AutoTokenizer.from_pretrained(config.text_model_name)
        self.max_length = config.text_max_length
        self.image_transform = transforms.Compose(
            [
                transforms.Resize((config.image_size, config.image_size)),
                transforms.ToTensor(),
            ]
        )

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        row = self.frame.iloc[index]
        encoded = self.tokenizer(
            row["clinical_text"],
            max_length=self.max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        image = self.image_transform(Image.open(Path(row["image_path"])).convert("RGB"))
        return {
            "input_ids": encoded["input_ids"].squeeze(0),
            "attention_mask": encoded["attention_mask"].squeeze(0),
            "image": image,
            "label": torch.tensor(int(row["label"]), dtype=torch.long),
            "anomaly_score": torch.tensor(float(row["anomaly_score"]), dtype=torch.float32),
            "report": row["report"],
        }
