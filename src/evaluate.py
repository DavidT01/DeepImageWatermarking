"""Evaluation utilities for trained watermarking models."""

import math
from pathlib import Path

import torch
from torch import nn

from src.image_metrics import mean_squared_error, structural_similarity_index
from src.message_metrics import logits_to_bits
from src.noise import apply_attack
from src.train import TrainConfig, build_models
from src.utils import SEED


def load_models(
    checkpoint_path: str | Path,
    device: torch.device | str,
) -> tuple[nn.Module, nn.Module, dict]:
    """Load an encoder-decoder pair from a training checkpoint."""
    device = torch.device(device)
    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False,
    )
    config = checkpoint["config"].copy()
    config["seed"] = checkpoint.get("seed", SEED)
    config.setdefault(
        "architecture",
        "advanced" if "conv4.weight" in checkpoint["encoder_state_dict"] else "simple",
    )
    for model_name in ("encoder", "decoder"):
        channel_key = f"{model_name}_channels"
        if config.get(channel_key) is None:
            weights = checkpoint[f"{model_name}_state_dict"]
            config[channel_key] = (
                tuple(weights[f"conv{index}.weight"].shape[0] for index in (1, 2, 3))
                if config["architecture"] == "simple"
                else weights["conv1.weight"].shape[0]
            )
    model_config = TrainConfig(
        architecture=config["architecture"],
        message_length=config["message_length"],
        encoder_channels=config["encoder_channels"],
        decoder_channels=config["decoder_channels"],
        encoder_max_delta=config.get("encoder_max_delta"),
        decoder_normalization=config.get("decoder_normalization", "batch"),
        decoder_pooling=config.get("decoder_pooling", "max"),
    )
    config["decoder_normalization"] = model_config.decoder_normalization
    encoder, decoder = build_models(model_config)
    encoder.to(device)
    decoder.to(device)

    encoder.load_state_dict(checkpoint["encoder_state_dict"])
    decoder.load_state_dict(checkpoint["decoder_state_dict"])
    encoder.eval()
    decoder.eval()

    return encoder, decoder, config


def _content_ssim(
    images: torch.Tensor,
    watermarked: torch.Tensor,
    masks: torch.Tensor,
) -> float:
    """Calculate summed SSIM over the unpadded regions."""
    total = 0.0

    for image, marked_image, mask in zip(images, watermarked, masks):
        content = mask[0].bool()
        rows = torch.where(content.any(dim=1))[0]
        columns = torch.where(content.any(dim=0))[0]

        top, bottom = rows[0].item(), rows[-1].item() + 1
        left, right = columns[0].item(), columns[-1].item() + 1
        image = image[:, top:bottom, left:right]
        marked_image = marked_image[:, top:bottom, left:right]

        total += structural_similarity_index(image, marked_image).item()

    return total


@torch.no_grad()
def evaluate_model(
    encoder: nn.Module,
    decoder: nn.Module,
    loader,
    device: torch.device | str,
    attacks: list[dict] | None = None,
    message_length: int = 16,
    seed: int = SEED,
    fixed_messages: torch.Tensor | None = None,
) -> dict[str, float | int]:
    """Evaluate message recovery and watermarked image quality."""
    device = torch.device(device)
    encoder.to(device).eval()
    decoder.to(device).eval()

    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    message_generator = torch.Generator().manual_seed(seed)
    criterion = nn.BCEWithLogitsLoss(reduction="sum")

    message_loss = 0.0
    bit_errors = 0
    exact_messages = 0
    squared_error = 0.0
    content_values = 0
    ssim_total = 0.0
    num_images = 0
    message_start = 0

    for images, masks in loader:
        images = images.to(device)
        masks = masks.to(device)
        batch_size = images.size(0)

        if fixed_messages is None:
            messages = torch.randint(
                0,
                2,
                (batch_size, message_length),
                generator=message_generator,
                dtype=torch.float32,
            ).to(device)
        else:
            message_end = message_start + batch_size
            if message_end > len(fixed_messages):
                raise ValueError("fixed_messages does not contain enough messages")
            messages = fixed_messages[message_start:message_end].to(device)
            message_start = message_end

        watermarked = encoder(images, messages, masks)
        decoder_input = watermarked
        decoder_masks = masks
        for attack in attacks or []:
            decoder_input, decoder_masks = apply_attack(
                decoder_input,
                decoder_masks,
                attack,
            )

        logits = decoder(decoder_input, decoder_masks)
        predicted_bits = logits_to_bits(logits)
        target_bits = messages.bool()
        content = masks.expand_as(images).bool()
        batch_content_values = content.sum().item()
        batch_mse = mean_squared_error(
            images[content],
            watermarked[content],
        ).item()

        message_loss += criterion(logits, messages).item()
        bit_errors += (predicted_bits != target_bits).sum().item()
        exact_messages += (
            (predicted_bits == target_bits).all(dim=1).sum().item()
        )
        squared_error += batch_mse * batch_content_values
        content_values += batch_content_values
        ssim_total += _content_ssim(images, watermarked, masks)
        num_images += batch_size

    if num_images == 0:
        raise ValueError("The evaluation loader is empty")

    total_bits = num_images * message_length
    mse = squared_error / content_values
    psnr = float("inf") if mse == 0 else 10 * math.log10(1 / mse)

    return {
        "message_loss": message_loss / total_bits,
        "ber": bit_errors / total_bits,
        "exact_accuracy": exact_messages / num_images,
        "mse": mse,
        "psnr": psnr,
        "ssim": ssim_total / num_images,
        "num_images": num_images,
    }


def evaluate_scenarios(
    encoder: nn.Module,
    decoder: nn.Module,
    loader,
    device: torch.device | str,
    scenarios: dict[str, list[dict]],
    message_length: int = 16,
    seed: int = SEED,
    fixed_messages: torch.Tensor | None = None,
) -> list[dict]:
    """Evaluate named clean and attacked configurations."""
    results = []

    for name, attacks in scenarios.items():
        metrics = evaluate_model(
            encoder,
            decoder,
            loader,
            device,
            attacks=attacks,
            message_length=message_length,
            seed=seed,
            fixed_messages=fixed_messages,
        )
        results.append({"scenario": name, "attacks": attacks, **metrics})

    return results
