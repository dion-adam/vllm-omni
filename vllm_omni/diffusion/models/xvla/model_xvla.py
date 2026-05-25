# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from PIL import Image

logger = logging.getLogger(__name__)


def _first_sample_tensor(tensor: torch.Tensor) -> torch.Tensor:
    """Select one RGB frame from common vLLM-Omni robot image layouts."""
    if tensor.ndim == 5:
        # [B, T/V, C, H, W]: first batch item, most recent frame/view.
        tensor = tensor[0, -1]
    elif tensor.ndim == 4:
        # [B, C, H, W] or [T, C, H, W].
        tensor = tensor[-1]
    if tensor.ndim != 3:
        raise ValueError(f"Expected image tensor with 3-5 dims, got {tuple(tensor.shape)}")
    return tensor


def _tensor_to_pil(tensor: torch.Tensor) -> Image.Image:
    tensor = _first_sample_tensor(tensor)
    if tensor.shape[0] in (1, 3, 4):
        tensor = tensor.permute(1, 2, 0)
    if tensor.dtype.is_floating_point:
        tensor = tensor.detach().cpu().clamp(0.0, 1.0)
        tensor = (tensor * 255.0).round().to(torch.uint8)
    else:
        tensor = tensor.detach().cpu().to(torch.uint8)
    return Image.fromarray(tensor.numpy()).convert("RGB")


def _extract_images(batch_inputs: dict[str, Any]) -> list[Image.Image]:
    images: list[Image.Image] = []
    for key in sorted(batch_inputs):
        if not key.startswith("observation.images.") or key.endswith("_mask"):
            continue
        value = batch_inputs[key]
        if isinstance(value, torch.Tensor):
            images.append(_tensor_to_pil(value))
        elif isinstance(value, Image.Image):
            images.append(value.convert("RGB"))
        elif isinstance(value, np.ndarray):
            images.append(Image.fromarray(value).convert("RGB"))
    return images


def _extract_instruction(batch_inputs: dict[str, Any]) -> str:
    task = batch_inputs.get("observation.task", "")
    if isinstance(task, list):
        task = task[0] if task else ""
    return str(task)


def _extract_proprio(batch_inputs: dict[str, Any], dim: int) -> np.ndarray:
    state = batch_inputs.get("observation.state")
    if state is None:
        logger.debug("observation.state not found in batch_inputs; using zeros.")
        return np.zeros(dim, dtype=np.float32)
    if isinstance(state, torch.Tensor):
        while state.ndim > 1:
            state = state[0]
        return state.detach().cpu().float().numpy()
    return np.asarray(state, dtype=np.float32)


def _torch_dtype_from_config(dtype: Any) -> torch.dtype | None:
    if isinstance(dtype, torch.dtype):
        return dtype
    if isinstance(dtype, str):
        return getattr(torch, dtype.removeprefix("torch."), None)
    return None


class XVLAModel(nn.Module):
    """Thin adapter around upstream HuggingFace X-VLA checkpoints."""

    def __init__(self, model_path: str, od_config: Any) -> None:
        super().__init__()
        self.od_config = od_config

        from transformers import AutoModel, AutoProcessor

        torch_dtype = _torch_dtype_from_config(getattr(od_config, "dtype", None))
        logger.info("Loading XVLA model weights from %s", model_path)
        self._xvla = AutoModel.from_pretrained(
            model_path,
            trust_remote_code=True,
            torch_dtype=torch_dtype,
        )
        self._xvla.eval()

        logger.info("Loading XVLA processor from %s", model_path)
        self._processor = AutoProcessor.from_pretrained(
            model_path,
            trust_remote_code=True,
        )

        self.dim_action: int = int(self._xvla.action_space.dim_action)
        self.num_actions: int = int(self._xvla.num_actions)

    @property
    def device(self) -> torch.device:
        return next(self._xvla.parameters()).device

    @property
    def dtype(self) -> torch.dtype:
        return next(self._xvla.parameters()).dtype

    def load_weights(self, weights: Any) -> set[str]:
        del weights
        return set(self.state_dict().keys())

    @torch.inference_mode()
    def generate_actions(self, batch_inputs: dict[str, Any]) -> torch.Tensor:
        images = _extract_images(batch_inputs)
        if not images:
            raise ValueError(
                "XVLAModel.generate_actions requires at least one image under "
                "keys matching 'observation.images.*'."
            )

        instruction = _extract_instruction(batch_inputs)
        proprio_np = _extract_proprio(batch_inputs, dim=self.dim_action)
        domain_id_val = int(batch_inputs.get("domain_id", 0))
        steps = int(batch_inputs.get("steps", 10))

        proc_out = self._processor(
            images=images,
            language_instruction=instruction,
        )

        def to_model(tensor: torch.Tensor) -> torch.Tensor:
            if not isinstance(tensor, torch.Tensor):
                tensor = torch.as_tensor(tensor)
            if tensor.is_floating_point():
                return tensor.to(device=self.device, dtype=self.dtype)
            return tensor.to(device=self.device)

        inputs = {key: to_model(value) for key, value in proc_out.items()}
        inputs["proprio"] = torch.as_tensor(
            proprio_np,
            dtype=self.dtype,
            device=self.device,
        ).unsqueeze(0)
        inputs["domain_id"] = torch.tensor(
            [domain_id_val],
            dtype=torch.long,
            device=self.device,
        )

        actions = self._xvla.generate_actions(**inputs, steps=steps)
        return actions.detach().float().cpu()
