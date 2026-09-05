import csv
from dataclasses import asdict
from pathlib import Path
import sys

import torch
from torch import nn


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from src.data import get_dataloaders
from src.evaluate import load_models
from src.train import (
    TrainConfig,
    _validate_reproducibly,
    run_training,
)
from src.utils import SEED


MILD_ATTACKS = [
    {"name": "none"},
    {"name": "gaussian_noise", "std": 0.01},
    {"name": "gaussian_noise", "std": 0.03},
    {"name": "gaussian_blur", "kernel_size": 3, "sigma": 0.5},
    {"name": "downscale", "scale_factor": 0.75},
    {"name": "rotation", "angle": -2.0},
    {"name": "rotation", "angle": 2.0},
]

HARD_ATTACKS = MILD_ATTACKS + [
    {"name": "gaussian_noise", "std": 0.05},
    {"name": "gaussian_blur", "kernel_size": 5, "sigma": 1.0},
    {"name": "downscale", "scale_factor": 0.5},
    {"name": "rotation", "angle": -5.0},
    {"name": "rotation", "angle": 5.0},
]

EVALUATION_SCENARIOS = {
    "clean": None,
    "noise-0.01": [{"name": "gaussian_noise", "std": 0.01}],
    "noise-0.03": [{"name": "gaussian_noise", "std": 0.03}],
    "noise-0.05": [{"name": "gaussian_noise", "std": 0.05}],
    "blur-3-0.5": [
        {"name": "gaussian_blur", "kernel_size": 3, "sigma": 0.5}
    ],
    "blur-5-1.0": [
        {"name": "gaussian_blur", "kernel_size": 5, "sigma": 1.0}
    ],
    "downscale-0.75": [{"name": "downscale", "scale_factor": 0.75}],
    "downscale-0.50": [{"name": "downscale", "scale_factor": 0.5}],
    "rotation-minus-2": [{"name": "rotation", "angle": -2.0}],
    "rotation-plus-2": [{"name": "rotation", "angle": 2.0}],
    "rotation-minus-5": [{"name": "rotation", "angle": -5.0}],
    "rotation-plus-5": [{"name": "rotation", "angle": 5.0}],
}


def make_finetune_checkpoint(
    source_path: Path,
    destination_path: Path,
    config: TrainConfig,
) -> None:
    checkpoint = torch.load(
        source_path,
        map_location="cpu",
        weights_only=False,
    )
    checkpoint["epoch"] = -1
    checkpoint["best_val_loss"] = float("inf")
    checkpoint["best_val_metric"] = float("inf")
    checkpoint["config"] = asdict(config)
    checkpoint.pop("python_random_state", None)
    checkpoint.pop("torch_random_state", None)
    checkpoint.pop("torch_cuda_random_state", None)

    destination_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, destination_path)


def train_phase(
    config: TrainConfig,
    source_checkpoint: Path,
) -> Path:
    checkpoint_dir = Path(config.checkpoint_dir) / config.experiment_name
    last_checkpoint = checkpoint_dir / "last.pt"
    initial_checkpoint = checkpoint_dir / "initial.pt"

    if last_checkpoint.exists():
        resume_from = last_checkpoint
    else:
        make_finetune_checkpoint(
            source_checkpoint,
            initial_checkpoint,
            config,
        )
        resume_from = initial_checkpoint

    print(f"Training {config.experiment_name}")
    print(f"Resuming from: {resume_from}")
    run_training(
        data_dir=PROJECT_ROOT / "data",
        splits_file=PROJECT_ROOT / "data" / "splits.json",
        config=config,
        num_workers=4,
        resume_from=resume_from,
    )
    return checkpoint_dir / "best_model.pt"


def evaluate_checkpoint(
    checkpoint_path: Path,
    output_path: Path,
) -> list[dict[str, float | str]]:
    _, val_loader, _ = get_dataloaders(
        PROJECT_ROOT / "data",
        PROJECT_ROOT / "data" / "splits.json",
        batch_size=16,
        num_workers=4,
        shuffle_train=False,
    )
    if val_loader is None:
        raise ValueError("Validation split is required")

    device = torch.device("cuda")
    encoder, decoder, config = load_models(checkpoint_path, device)
    criterion = nn.BCEWithLogitsLoss()
    generator = torch.Generator().manual_seed(SEED)
    fixed_messages = torch.randint(
        0,
        2,
        (len(val_loader.dataset), config["message_length"]),
        generator=generator,
        dtype=torch.float32,
    )

    records: list[dict[str, float | str]] = []
    for name, attacks in EVALUATION_SCENARIOS.items():
        metrics = _validate_reproducibly(
            encoder=encoder,
            decoder=decoder,
            loader=val_loader,
            criterion=criterion,
            device=device,
            message_length=config["message_length"],
            image_loss_weight=0.0,
            attack_configs=attacks,
            fixed_messages=fixed_messages,
        )
        record: dict[str, float | str] = {"scenario": name}
        record.update(metrics)
        records.append(record)
        print(
            f"{name}: BER={metrics['ber']:.4f}, "
            f"exact={metrics['exact_accuracy']:.4f}, "
            f"PSNR={metrics['psnr']:.2f}"
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)

    return records


def load_evaluation_records(
    path: Path,
) -> list[dict[str, float | str]]:
    with path.open(newline="", encoding="utf-8") as input_file:
        return list(csv.DictReader(input_file))


def robust_target_reached(records: list[dict[str, float | str]]) -> bool:
    clean = next(record for record in records if record["scenario"] == "clean")
    attacked = [
        record for record in records if record["scenario"] != "clean"
    ]
    average_attacked_ber = sum(
        float(record["ber"]) for record in attacked
    ) / len(attacked)
    worst_attacked_ber = max(float(record["ber"]) for record in attacked)

    print(f"Average attacked BER: {average_attacked_ber:.4f}")
    print(f"Worst attacked BER: {worst_attacked_ber:.4f}")
    return (
        float(clean["ber"]) < 0.01
        and average_attacked_ber < 0.05
        and worst_attacked_ber < 0.15
    )


def main() -> None:
    clean_last_checkpoint = (
        PROJECT_ROOT
        / "results"
        / "checkpoints"
        / "different-architecture-full-clean"
        / "last.pt"
    )
    clean_checkpoint = (
        PROJECT_ROOT
        / "results"
        / "checkpoints"
        / "different-architecture-full-clean"
        / "best_model.pt"
    )

    clean_last = torch.load(
        clean_last_checkpoint,
        map_location="cpu",
        weights_only=False,
    )
    if clean_last["epoch"] + 1 < 200:
        raise RuntimeError("Clean training did not reach epoch 200")

    clean_evaluation_path = (
        PROJECT_ROOT / "results" / "logs" / "clean-attack-baseline.csv"
    )
    if clean_evaluation_path.exists():
        print("Loading the existing clean attack profile")
        clean_records = load_evaluation_records(clean_evaluation_path)
    else:
        print("Evaluating the final clean checkpoint")
        clean_records = evaluate_checkpoint(
            clean_checkpoint,
            clean_evaluation_path,
        )

    mild_config = TrainConfig(
        experiment_name="different-architecture-robust-mild-balanced",
        epochs=40,
        batch_size=16,
        message_length=16,
        encoder_channels=40,
        decoder_channels=40,
        encoder_max_delta=0.03,
        learning_rate=3e-4,
        image_loss_weight=0.0,
        device="cuda",
        checkpoint_dir=str(PROJECT_ROOT / "results" / "checkpoints"),
        log_path=str(
            PROJECT_ROOT / "results" / "logs" / "robust-mild-balanced.csv"
        ),
        attack_configs=MILD_ATTACKS,
        validation_attack_configs=MILD_ATTACKS,
        print_every=1,
        checkpoint_metric="ber",
    )
    mild_checkpoint = train_phase(mild_config, clean_checkpoint)
    mild_records = evaluate_checkpoint(
        mild_checkpoint,
        PROJECT_ROOT
        / "results"
        / "logs"
        / "robust-mild-balanced-scenarios.csv",
    )
    if robust_target_reached(mild_records):
        print("Robustness target reached after the mild phase")
        return

    hard_config = TrainConfig(
        experiment_name="different-architecture-robust-hard-balanced",
        epochs=60,
        batch_size=16,
        message_length=16,
        encoder_channels=40,
        decoder_channels=40,
        encoder_max_delta=0.03,
        learning_rate=1e-4,
        image_loss_weight=0.0,
        device="cuda",
        checkpoint_dir=str(PROJECT_ROOT / "results" / "checkpoints"),
        log_path=str(
            PROJECT_ROOT / "results" / "logs" / "robust-hard-balanced.csv"
        ),
        attack_configs=HARD_ATTACKS,
        validation_attack_configs=HARD_ATTACKS,
        print_every=1,
        checkpoint_metric="ber",
    )
    hard_checkpoint = train_phase(hard_config, mild_checkpoint)
    hard_records = evaluate_checkpoint(
        hard_checkpoint,
        PROJECT_ROOT
        / "results"
        / "logs"
        / "robust-hard-balanced-scenarios.csv",
    )
    if robust_target_reached(hard_records):
        print("Robustness target reached after the hard phase")
    else:
        print("Robustness target not reached; inspect scenario metrics")


if __name__ == "__main__":
    main()