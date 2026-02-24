"""Entrypoint for model training."""

from __future__ import annotations

from torch.utils.data import DataLoader

from src.config.settings import load_settings
from src.data.dataset import DatasetConfig, MultimodalClinicalDataset
from src.data.synthetic_data import generate_synthetic_dataset
from src.models.multitask_model import MultimodalClinicalModel
from src.training.trainer import Trainer
from src.utils.reproducibility import set_global_seed


def run_training(config_path: str = "src/config/default.yaml") -> None:
    settings = load_settings(config_path)
    set_global_seed(settings.experiment["seed"])
    generate_synthetic_dataset(num_samples=40)

    dataset_cfg = DatasetConfig(
        text_model_name=settings.model["text_model_name"],
        text_max_length=settings.data["text_max_length"],
        image_size=settings.data["image_size"],
    )
    train_ds = MultimodalClinicalDataset(settings.data["train_csv"], dataset_cfg)
    val_ds = MultimodalClinicalDataset(settings.data["val_csv"], dataset_cfg)
    train_loader = DataLoader(train_ds, batch_size=settings.training["batch_size"], shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=settings.training["batch_size"])

    model = MultimodalClinicalModel(
        text_model_name=settings.model["text_model_name"],
        hidden_dim=settings.model["hidden_dim"],
        num_classes=settings.model["num_classes"],
    )
    device = "cuda" if settings.inference["device"] == "cuda" else "cpu"
    trainer = Trainer(model, settings.raw, device)
    trainer.train(train_loader, val_loader, run_name="multimodal-baseline")


if __name__ == "__main__":
    run_training()
