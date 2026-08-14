"""Exactly matched GRU history encoders and method-specific training heads."""

from __future__ import annotations

from typing import Any

import torch
from torch import nn


class HistoryBackbone(nn.Module):
    """Shared normalized-transition GRU and deterministic latent projection."""

    def __init__(self, transition_dim: int = 7, hidden_size: int = 64, latent_dim: int = 8) -> None:
        super().__init__()
        self.transition_dim = int(transition_dim)
        self.hidden_size = int(hidden_size)
        self.latent_dim = int(latent_dim)
        self.gru = nn.GRU(self.transition_dim, self.hidden_size, num_layers=1, batch_first=True)
        self.latent_head = nn.Linear(self.hidden_size, self.latent_dim)

    def forward(self, history: torch.Tensor, lengths: torch.Tensor,
                mask: torch.Tensor | None = None) -> torch.Tensor:
        if history.ndim != 3 or history.shape[-1] != self.transition_dim:
            raise ValueError("history must have shape [batch,time,transition_dim]")
        lengths = lengths.to(device=history.device, dtype=torch.long)
        if mask is not None:
            expected = torch.arange(history.shape[1], device=history.device)[None, :] < lengths[:, None]
            if mask.shape != expected.shape or not torch.equal(mask.bool(), expected):
                raise ValueError("mask and lengths disagree")
        safe = lengths.clamp(min=1).cpu()
        packed = nn.utils.rnn.pack_padded_sequence(history, safe, batch_first=True, enforce_sorted=False)
        _, hidden = self.gru(packed)
        latent = self.latent_head(hidden[-1])
        # The prespecified empty-history representation is exactly zero for both methods.
        return latent * (lengths > 0).to(latent.dtype)[:, None]


class FutureDecoder(nn.Module):
    def __init__(self, latent_dim: int, horizon: int, hidden_size: int = 64) -> None:
        super().__init__()
        self.horizon = int(horizon)
        input_dim = latent_dim + 2 + 2 * horizon
        self.network = nn.Sequential(nn.Linear(input_dim, hidden_size), nn.Tanh(),
                                     nn.Linear(hidden_size, horizon * 3))

    def forward(self, latent: torch.Tensor, current_state: torch.Tensor,
                future_actions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        flat = torch.cat((latent, current_state, future_actions.flatten(1)), dim=-1)
        output = self.network(flat).reshape(-1, self.horizon, 3)
        return output[..., :2], output[..., 2]


class VAEHistoryEncoder(nn.Module):
    def __init__(self, transition_dim: int = 7, hidden_size: int = 64, latent_dim: int = 8,
                 future_horizon: int = 5) -> None:
        super().__init__()
        self.backbone = HistoryBackbone(transition_dim, hidden_size, latent_dim)
        self.logvar_head = nn.Linear(hidden_size, latent_dim)
        self.decoder = FutureDecoder(latent_dim, future_horizon, hidden_size)

    def distribution(self, history: torch.Tensor, lengths: torch.Tensor,
                     mask: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        mean = self.backbone(history, lengths, mask)
        safe = lengths.to(history.device).clamp(min=1).cpu()
        packed = nn.utils.rnn.pack_padded_sequence(history, safe, batch_first=True, enforce_sorted=False)
        _, hidden = self.backbone.gru(packed)
        logvar = self.logvar_head(hidden[-1]).clamp(-10.0, 10.0)
        active = (lengths.to(history.device) > 0).to(logvar.dtype)[:, None]
        return mean, logvar * active

    def encode(self, history: torch.Tensor, lengths: torch.Tensor,
               mask: torch.Tensor | None = None, deterministic: bool = True) -> torch.Tensor:
        mean, logvar = self.distribution(history, lengths, mask)
        if deterministic:
            return mean
        sample = mean + torch.exp(0.5 * logvar) * torch.randn_like(mean)
        return torch.where((lengths.to(history.device) > 0)[:, None], sample, torch.zeros_like(sample))

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        mean, logvar = self.distribution(batch["history"], batch["length"], batch.get("mask"))
        latent = mean + torch.exp(0.5 * logvar) * torch.randn_like(mean)
        latent = torch.where((batch["length"] > 0)[:, None], latent, torch.zeros_like(latent))
        state, reward = self.decoder(latent, batch["current_state"], batch["future_actions"])
        return {"mean": mean, "logvar": logvar, "latent": latent,
                "predicted_state_deltas": state, "predicted_rewards": reward}


class ContrastiveHistoryEncoder(nn.Module):
    def __init__(self, transition_dim: int = 7, hidden_size: int = 64, latent_dim: int = 8,
                 future_horizon: int = 5) -> None:
        super().__init__()
        self.backbone = HistoryBackbone(transition_dim, hidden_size, latent_dim)
        block_dim = 2 + future_horizon * (2 + 1 + 2)
        self.future_head = nn.Sequential(nn.Linear(block_dim, hidden_size), nn.Tanh(),
                                         nn.Linear(hidden_size, latent_dim))

    def encode(self, history: torch.Tensor, lengths: torch.Tensor,
               mask: torch.Tensor | None = None, deterministic: bool = True) -> torch.Tensor:
        del deterministic
        return self.backbone(history, lengths, mask)

    def future_embedding(self, current_state: torch.Tensor, future_states: torch.Tensor,
                         future_actions: torch.Tensor, future_rewards: torch.Tensor) -> torch.Tensor:
        block = torch.cat((current_state, future_states.flatten(1), future_actions.flatten(1),
                           future_rewards.flatten(1)), dim=-1)
        return nn.functional.normalize(self.future_head(block), dim=-1)


def vae_objective(outputs: dict[str, torch.Tensor], batch: dict[str, torch.Tensor],
                  state_weight: float, reward_weight: float, kl_weight: float) -> dict[str, torch.Tensor]:
    state_loss = nn.functional.mse_loss(outputs["predicted_state_deltas"], batch["future_state_deltas"])
    reward_loss = nn.functional.mse_loss(outputs["predicted_rewards"], batch["future_rewards"])
    kl = -0.5 * torch.mean(1.0 + outputs["logvar"] - outputs["mean"].square() - outputs["logvar"].exp())
    total = state_weight * state_loss + reward_weight * reward_loss + kl_weight * kl
    return {"total": total, "state_reconstruction": state_loss, "reward_reconstruction": reward_loss, "kl": kl}


def contrastive_objective(model: ContrastiveHistoryEncoder, batch: dict[str, torch.Tensor],
                          negative_rewards: torch.Tensor, temperature: float,
                          mode: str = "reward_relabel") -> dict[str, torch.Tensor]:
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    query = nn.functional.normalize(model.encode(batch["history"], batch["length"], batch.get("mask")), dim=-1)
    positive = model.future_embedding(batch["current_state"], batch["future_states"],
                                      batch["future_actions"], batch["future_rewards"])
    if mode == "reward_relabel" or mode == "reward_relabel_alternative":
        if negative_rewards.shape != batch["future_rewards"].shape:
            raise ValueError("one valid hard-negative reward block is required per sample")
        negative = model.future_embedding(batch["current_state"], batch["future_states"],
                                          batch["future_actions"], negative_rewards)
        logits = torch.stack(((query * positive).sum(-1), (query * negative).sum(-1)), dim=1) / temperature
        targets = torch.zeros(len(query), dtype=torch.long, device=query.device)
    elif mode == "in_batch":
        if len(query) < 2:
            raise ValueError("in-batch InfoNCE requires at least two samples")
        logits = query @ positive.T / temperature
        targets = torch.arange(len(query), device=query.device)
    else:
        raise ValueError(f"unknown negative mode {mode!r}")
    loss = nn.functional.cross_entropy(logits, targets)
    accuracy = (logits.argmax(dim=1) == targets).float().mean()
    return {"total": loss, "infonce": loss, "contrastive_accuracy": accuracy, "logits": logits}


def parameter_counts(model: nn.Module) -> dict[str, int]:
    backbone = sum(p.numel() for p in model.backbone.parameters())
    total = sum(p.numel() for p in model.parameters())
    return {"backbone": backbone, "method_specific": total - backbone, "total_training": total,
            "downstream_retained": backbone}


def checkpoint_payload(model: nn.Module, method: str, config: dict[str, Any], normalization: dict[str, Any],
                       dataset_checksum: str, seed: int, update: int, validation_loss: float) -> dict[str, Any]:
    return {"format_version": 1, "method": method, "seed": int(seed), "update": int(update),
            "validation_loss": float(validation_loss), "config": config,
            "normalization": normalization, "dataset_checksum": dataset_checksum,
            "parameter_counts": parameter_counts(model), "state_dict": model.state_dict()}


def build_model(method: str, encoder: dict[str, Any]) -> nn.Module:
    kwargs = {"transition_dim": int(encoder["transition_dim"]), "hidden_size": int(encoder["hidden_size"]),
              "latent_dim": int(encoder["latent_dim"]), "future_horizon": int(encoder["future_horizon"])}
    if method == "vae":
        return VAEHistoryEncoder(**kwargs)
    if method == "contrastive" or method == "contrastive_alternative":
        return ContrastiveHistoryEncoder(**kwargs)
    raise ValueError(f"unknown encoder method {method!r}")
