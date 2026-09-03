"""Training utilities for the image watermarking pipeline."""

import csv
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn

from src.data import get_dataloaders
from src.decoder import WatermarkDecoder
from src.encoder import WatermarkEncoder
from src.image_metrics import mean_squared_error, peak_signal_noise_ratio
from src.message_metrics import bit_error_rate, exact_message_accuracy
from src.noise import apply_random_attack
from src.utils import SEED, set_seed

@dataclass
class TrainConfig:
    """Configuration for one encoder-decoder training experiment."""

    experiment_name: str = "baseline"
    epochs: int = 20
    batch_size: int = 32
    message_length: int = 32
    encoder_channels: tuple[int, int, int] = (64, 64, 32)
    decoder_channels: tuple[int, int, int] = (32, 64, 128)
    learning_rate: float = 1e-3
    image_loss_weight: float = 1.0
    device: str = "auto"
    checkpoint_dir: str = "results/checkpoints"
    log_path: str = "results/experiments.csv"
    attack_configs: list[dict[str, Any]] | None = None

def _select_device(device: str) -> torch.device:
    """Select the configured training device."""

    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available")
        return torch.device("cuda")

    if device == "cpu":
        return torch.device("cpu")

    raise ValueError("device must be 'auto', 'cpu' or 'cuda'")

def _random_messages(batch_size: int, message_length: int, device: torch.device) -> torch.Tensor:
    """Generate a batch of random binary messages on the selected device."""

    return torch.randint(
        0,
        2,
        (batch_size, message_length),
        device=device,
        dtype=torch.float32,
    )

def _batch_messages(fixed_messages: torch.Tensor | None, batch_start: int, batch_size: int, message_length: int, device: torch.device) -> torch.Tensor:
    """Return fixed messages for a batch or generate random messages when none are provided."""

    if fixed_messages is None:
        return _random_messages(batch_size, message_length, device)

    batch_end = batch_start + batch_size
    if batch_end > len(fixed_messages):
        raise ValueError("fixed_messages does not contain enough messages")

    return fixed_messages[batch_start:batch_end].to(device)

def _run_batch(encoder: nn.Module, decoder: nn.Module, images: torch.Tensor, masks: torch.Tensor, messages: torch.Tensor, criterion: nn.Module,
               image_loss_weight: float, attack_configs: list[dict[str, Any]] | None) -> dict[str, torch.Tensor]:
    """Run encoding, optional image attack, decoding, loss calculation and metric calculation."""

    watermarked = encoder(images, messages, masks)

    if attack_configs:
        decoder_input = apply_random_attack(watermarked, attack_configs)
    else:
        decoder_input = watermarked

    logits = decoder(decoder_input)
    message_loss = criterion(logits, messages)

    content = masks.expand_as(images).bool()
    content_images = images[content]
    content_watermarked = watermarked[content]
    image_loss = mean_squared_error(content_images, content_watermarked)
    total_loss = message_loss + image_loss_weight * image_loss

    return {
        "loss": total_loss,
        "message_loss": message_loss,
        "image_loss": image_loss,
        "ber": bit_error_rate(logits.detach(), messages),
        "exact_accuracy": exact_message_accuracy(logits.detach(), messages),
        "psnr": peak_signal_noise_ratio(
            content_images.detach(),
            content_watermarked.detach(),
        ),
    }

def train_one_epoch(encoder: nn.Module, decoder: nn.Module, loader: Any, optimizer: torch.optim.Optimizer, criterion: nn.Module, device: torch.device,
                    message_length: int = 32, image_loss_weight: float = 1.0, attack_configs: list[dict[str, Any]] | None = None,
                    fixed_messages: torch.Tensor | None = None) -> dict[str, float]:
    """Run one optimization epoch and return averaged batch statistics."""

    encoder.train()
    decoder.train()
    totals: dict[str, float] = {}
    samples = 0

    batch_start = 0
    for images, masks in loader:
        images = images.to(device)
        masks = masks.to(device)
        batch_size = images.size(0)
        messages = _batch_messages(
            fixed_messages,
            batch_start,
            batch_size,
            message_length,
            device,
        )

        optimizer.zero_grad(set_to_none=True)
        values = _run_batch(encoder, decoder, images, masks, messages, criterion, image_loss_weight, attack_configs)
        values["loss"].backward()
        optimizer.step()
        batch_start += batch_size

        samples += batch_size
        for name, value in values.items():
            totals[name] = (
                totals.get(name, 0.0)
                + value.detach().item() * batch_size
            )

    if samples == 0:
        raise ValueError("The training loader is empty")

    return {name: value / samples for name, value in totals.items()}

@torch.no_grad()
def validate(encoder: nn.Module, decoder: nn.Module, loader: Any, criterion: nn.Module, device: torch.device, message_length: int = 32,
             image_loss_weight: float = 1.0, attack_configs: list[dict[str, Any]] | None = None,
             fixed_messages: torch.Tensor | None = None) -> dict[str, float]:
    """Evaluate the pipeline without updating model parameters."""

    encoder.eval()
    decoder.eval()
    totals: dict[str, float] = {}
    samples = 0

    batch_start = 0
    for images, masks in loader:
        images = images.to(device)
        masks = masks.to(device)
        batch_size = images.size(0)
        messages = _batch_messages(
            fixed_messages,
            batch_start,
            batch_size,
            message_length,
            device,
        )
        batch_start += batch_size

        values = _run_batch(encoder, decoder, images, masks, messages, criterion, image_loss_weight, attack_configs)

        samples += batch_size
        for name, value in values.items():
            totals[name] = totals.get(name, 0.0) + value.item() * batch_size

    if samples == 0:
        raise ValueError("The validation loader is empty")

    return {name: value / samples for name, value in totals.items()}

def save_checkpoint(path: str | Path, encoder: nn.Module, decoder: nn.Module, optimizer: torch.optim.Optimizer, epoch: int,
                    best_val_loss: float, config: TrainConfig) -> None:
    """Save all state needed to resume an experiment."""

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "encoder_state_dict": encoder.state_dict(),
            "decoder_state_dict": decoder.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "best_val_loss": best_val_loss,
            "config": asdict(config),
            "seed": SEED,
            "python_random_state": random.getstate(),
            "torch_random_state": torch.get_rng_state(),
            "torch_cuda_random_state": (
                torch.cuda.get_rng_state_all()
                if torch.cuda.is_available()
                else None
            ),
        },
        path,
    )

def load_checkpoint(path: str | Path, encoder: nn.Module, decoder: nn.Module, optimizer: torch.optim.Optimizer | None = None,
                    device: torch.device | str = "cpu") -> tuple[int, float]:
    """Load model and optionally optimizer state; return next epoch and best loss."""

    checkpoint = torch.load(path, map_location=device, weights_only=False)

    encoder.load_state_dict(checkpoint["encoder_state_dict"])
    decoder.load_state_dict(checkpoint["decoder_state_dict"])

    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    if "python_random_state" in checkpoint:
        random.setstate(checkpoint["python_random_state"])

    if "torch_random_state" in checkpoint:
        torch.set_rng_state(checkpoint["torch_random_state"].cpu())

    cuda_state = checkpoint.get("torch_cuda_random_state")
    if cuda_state is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([state.cpu() for state in cuda_state])

    return checkpoint["epoch"] + 1, checkpoint.get("best_val_loss", float("inf"))

def _append_log(path: str | Path, row: dict[str, Any]) -> None:
    """Append one epoch of training statistics to a CSV log file."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(row)
    exists = path.exists()

    with path.open("a", newline="", encoding="utf-8") as log_file:
        writer = csv.DictWriter(log_file, fieldnames=fields)
        if not exists:
            writer.writeheader()
        writer.writerow(row)

def fit(
    train_loader: Any,
    val_loader: Any,
    config: TrainConfig | None = None,
    resume_from: str | Path | None = None,
    fixed_train_messages: torch.Tensor | None = None,
    fixed_val_messages: torch.Tensor | None = None,
) -> tuple[nn.Module, nn.Module, list[dict[str, Any]]]:
    """Train encoder and decoder, checkpointing every epoch and the best model."""

    config = config or TrainConfig()
    set_seed()
    device = _select_device(config.device)
    encoder = WatermarkEncoder(
        message_length=config.message_length,
        feature_channels=config.encoder_channels,
    ).to(device)
    decoder = WatermarkDecoder(
        message_length=config.message_length,
        feature_channels=config.decoder_channels,
    ).to(device)
    optimizer = torch.optim.Adam(list(encoder.parameters()) + list(decoder.parameters()), lr=config.learning_rate)
    criterion = nn.BCEWithLogitsLoss()
    start_epoch = 0
    best_val_loss = float("inf")

    if fixed_val_messages is None:
        generator = torch.Generator().manual_seed(SEED)
        fixed_val_messages = torch.randint(
            0,
            2,
            (len(val_loader.dataset), config.message_length),
            generator=generator,
            dtype=torch.float32,
        )

    if resume_from is not None:
        start_epoch, best_val_loss = load_checkpoint(resume_from, encoder, decoder, optimizer, device)

    history: list[dict[str, Any]] = []
    checkpoint_dir = Path(config.checkpoint_dir) / config.experiment_name

    for epoch in range(start_epoch, config.epochs):
        train_stats = train_one_epoch(
            encoder=encoder,
            decoder=decoder,
            loader=train_loader,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
            message_length=config.message_length,
            image_loss_weight=config.image_loss_weight,
            attack_configs=config.attack_configs,
            fixed_messages=fixed_train_messages,
        )
        val_stats = validate(
            encoder=encoder,
            decoder=decoder,
            loader=val_loader,
            criterion=criterion,
            device=device,
            message_length=config.message_length,
            image_loss_weight=config.image_loss_weight,
            attack_configs=None,
            fixed_messages=fixed_val_messages,
        )

        row: dict[str, Any] = {
            "experiment": config.experiment_name,
            "epoch": epoch + 1,
            "seed": SEED,
            "learning_rate": config.learning_rate,
            "batch_size": config.batch_size,
            "encoder_channels": json.dumps(config.encoder_channels),
            "decoder_channels": json.dumps(config.decoder_channels),
            "image_loss_weight": config.image_loss_weight,
            "attack_configs": json.dumps(config.attack_configs),
        }
        row.update({f"train_{name}": value for name, value in train_stats.items()})
        row.update({f"val_{name}": value for name, value in val_stats.items()})
        history.append(row)
        _append_log(config.log_path, row)

        is_best = val_stats["loss"] < best_val_loss
        if is_best:
            best_val_loss = val_stats["loss"]

        save_checkpoint(checkpoint_dir / "last.pt", encoder, decoder, optimizer, epoch, best_val_loss, config)

        if is_best:
            save_checkpoint(checkpoint_dir / "best_model.pt", encoder, decoder, optimizer, epoch, best_val_loss, config)

    return encoder, decoder, history

def run_training(
    data_dir: str | Path,
    splits_file: str | Path,
    config: TrainConfig | None = None,
    num_workers: int = 0,
    resume_from: str | Path | None = None,
    fixed_train_messages: torch.Tensor | None = None,
    fixed_val_messages: torch.Tensor | None = None,
    shuffle_train: bool = True,
) -> tuple[nn.Module, nn.Module, list[dict[str, Any]]]:
    """Build project dataloaders and start training."""

    config = config or TrainConfig()
    train_loader, val_loader, _ = get_dataloaders(data_dir, splits_file, batch_size=config.batch_size, num_workers=num_workers, shuffle_train=shuffle_train)

    if train_loader is None or val_loader is None:
        raise ValueError("Both train and val splits are required for training")

    return fit(
        train_loader,
        val_loader,
        config=config,
        resume_from=resume_from,
        fixed_train_messages=fixed_train_messages,
        fixed_val_messages=fixed_val_messages,
    )
