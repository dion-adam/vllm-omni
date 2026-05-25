# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import os
from collections.abc import Iterable
from typing import Any

from PIL import Image
import torch
from torch import nn
from vllm.logger import init_logger

from vllm_omni.diffusion.data import DiffusionOutput, OmniDiffusionConfig
from vllm_omni.diffusion.profiler.diffusion_pipeline_profiler import (
    DiffusionPipelineProfilerMixin,
    wrap_methods_by_paths,
)
from vllm_omni.diffusion.request import OmniDiffusionRequest
from vllm_omni.model_executor.model_loader.weight_utils import download_weights_from_hf_specific

logger = init_logger(__name__)


def get_xvla_post_process_func(od_config: OmniDiffusionConfig):
    del od_config
    return lambda x: x


def _resolve_model_path(model_name: str | None, revision: str | None) -> str:
    if not model_name:
        raise ValueError("XVLA model path or Hugging Face repo ID is required.")
    if os.path.isdir(model_name) or os.path.isfile(model_name):
        return model_name
    return download_weights_from_hf_specific(model_name, None, ["*"], revision=revision)


def _tensor_to_pil(image_tensor: torch.Tensor) -> Image.Image:
    if image_tensor.ndim != 3:
        raise ValueError(f"Expected image tensor with shape [C,H,W], got {tuple(image_tensor.shape)}")
    if image_tensor.dtype.is_floating_point:
        image_tensor = image_tensor.detach().cpu().clamp(0.0, 1.0)
        image_tensor = (image_tensor * 255.0).round().to(torch.uint8)
    else:
        image_tensor = image_tensor.detach().cpu().to(torch.uint8)
    array = image_tensor.permute(1, 2, 0).numpy()
    return Image.fromarray(array)


class XVLAPipeline(nn.Module, DiffusionPipelineProfilerMixin):
    """XVLA wrapper pipeline for vLLM-Omni."""

    supports_step_execution: bool = False

    def __init__(self, *, od_config: OmniDiffusionConfig, prefix: str = ""):
        super().__init__()
        self.od_config = od_config
        self.prefix = prefix
        self.model_path = _resolve_model_path(self.od_config.model, self.od_config.revision)
        self.processor = None
        self.model = self._initialize_model()
        self.weights_sources: list[Any] = []
        self._setup_profiler_targets()

    def _initialize_model(self) -> nn.Module:
        try:
            from transformers import XVLAForConditionalGeneration, XVLAProcessor
        except Exception as exc:
            raise RuntimeError(
                "XVLA support requires a transformers build with XVLA classes. "
                "Install the XVLA-enabled transformer package and retry."
            ) from exc

        logger.info("Loading XVLA model from %s", self.model_path)
        model = XVLAForConditionalGeneration.from_pretrained(
            self.model_path,
            trust_remote_code=self.od_config.trust_remote_code,
        )
        self.processor = XVLAProcessor.from_pretrained(
            self.model_path,
            trust_remote_code=self.od_config.trust_remote_code,
        )

        model.eval()
        if self.od_config.dtype is not None:
            model.to(self.od_config.dtype)
        return model

    def _setup_profiler_targets(self) -> None:
        if not self.od_config.enable_diffusion_pipeline_profiler:
            return
        wrap_methods_by_paths(self, ["model.forward", "model.generate", "model.generate_actions"])

    def has_real_checkpoint(self) -> bool:
        return bool(self.model_path and os.path.exists(self.model_path))

    def runtime_mode(self) -> str:
        return "real_checkpoint_loaded" if self.has_real_checkpoint() else "no_checkpoint_policy"

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        del weights
        # The XVLA model is loaded during from_pretrained and does not require
        # an external weight-loading pass through AutoWeightsLoader.
        return set(self.state_dict().keys())

    def _build_images(self, batch_inputs: dict[str, Any]) -> list[Image.Image]:
        images: list[Image.Image] = []
        for key, value in sorted(batch_inputs.items()):
            if not key.startswith("observation.images.") or key.endswith("_mask"):
                continue
            if not isinstance(value, torch.Tensor):
                continue
            if value.ndim == 4:
                value = value[0]
            images.append(_tensor_to_pil(value))
        return images

    def _build_prompt(self, batch_inputs: dict[str, Any]) -> str:
        task = batch_inputs.get("observation.task")
        if isinstance(task, list) and len(task) == 1:
            task = task[0]
        if task is None:
            return ""
        if not isinstance(task, str):
            return str(task)
        return task

    def _decode_actions(self, output: Any) -> list[Any]:
        if isinstance(output, torch.Tensor):
            return output.detach().cpu().tolist()
        if isinstance(output, list):
            return output
        if hasattr(output, "tolist"):
            return output.tolist()
        return [output]

    def _predict_actions(self, batch_inputs: dict[str, Any]) -> list[Any]:
        if hasattr(self.model, "generate_actions"):
            result = self.model.generate_actions(batch_inputs)
            return self._decode_actions(result)

        if self.processor is None:
            raise RuntimeError("XVLA processor is not initialized.")

        images = self._build_images(batch_inputs)
        prompt = self._build_prompt(batch_inputs)
        inputs = self.processor(
            images=images or None,
            text=prompt,
            return_tensors="pt",
            padding=True,
        )

        for key, value in list(inputs.items()):
            if isinstance(value, torch.Tensor):
                inputs[key] = value.to(self.model.device)

        custom_args = self.od_config.custom_pipeline_args or {}
        max_new_tokens = int(custom_args.get("max_new_tokens", 64))
        output_ids = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )

        if hasattr(self.processor, "batch_decode"):
            return self.processor.batch_decode(output_ids, skip_special_tokens=True)
        if hasattr(self.processor, "decode"):
            return [self.processor.decode(output_ids[0], skip_special_tokens=True)]
        return self._decode_actions(output_ids)

    @torch.inference_mode()
    def forward(self, req: OmniDiffusionRequest) -> DiffusionOutput:
        extra_args = getattr(req.sampling_params, "extra_args", {}) or {}
        batch_inputs = extra_args.get("batch_inputs")
        if batch_inputs is None:
            logger.warning("XVLAPipeline.forward called without batch_inputs; returning dummy output.")
            return DiffusionOutput(
                output=torch.empty(0, device="cpu"),
                custom_output={"warning": "missing batch_inputs; XVLA requires observation inputs"},
                post_process_func=get_xvla_post_process_func(self.od_config),
            )

        actions = self._predict_actions(batch_inputs)
        return DiffusionOutput(
            output=torch.empty(0, device="cpu"),
            custom_output={"actions": actions},
            post_process_func=get_xvla_post_process_func(self.od_config),
        )
