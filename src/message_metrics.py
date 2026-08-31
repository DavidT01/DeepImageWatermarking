"""Metrics for decoded binary messages."""

import torch


def logits_to_bits(logits: torch.Tensor) -> torch.Tensor:
    """Convert logits to bits using zero as the decision threshold."""
    return logits >= 0


def bit_error_rate(logits: torch.Tensor, messages: torch.Tensor) -> torch.Tensor:
    """Calculate the fraction of incorrectly decoded bits."""
    predicted_bits = logits_to_bits(logits)
    return (predicted_bits != messages.bool()).float().mean()


def exact_message_accuracy(
    logits: torch.Tensor,
    messages: torch.Tensor,
) -> torch.Tensor:
    """Calculate the fraction of messages with all bits correct."""
    predicted_bits = logits_to_bits(logits)
    correct_messages = (predicted_bits == messages.bool()).all(dim=1)
    return correct_messages.float().mean()