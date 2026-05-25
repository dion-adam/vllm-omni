# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import os
from collections.abc import Iterable
from typing import Any

import torch
from torch import nn
from vllm.logger import init_logger

from vllm_omni.diffusion.data import DiffusionOutput, OmniDiffusionConfig
from vllm_omni.diffusion.profiler.diffusion_pipeline_profiler import (
    DiffusionPipelineProfilerMixin,
    wrap_methods_by_paths,
)
from vllm_omni.diffusion.request import OmniDiffusionRequest
from vllm_omni.model_executor.model_loader.weight_utils import (
    download_weights_from_hf_specific,
)

from .model_xvla import XVLAModel

logger = init_logger(__name__)


def xvla_post_process_func(x: Any) -> Any:
    return x


def get_xvla_post_process_func(od_config: OmniDiffusionConfig):
    del od_config
    return xvla_post_process_func


def _resolve_model_path(model_name: str | None, revision: str | None) -> str:
    if not model_name:
        raise ValueError("XVLA model path or Hugging Face repo ID is required.")
    if os.path.isdir(model_name) or os.path.isfile(model_name):
        return model_name
    return download_weights_from_hf_specific(
        model_name, None, ["*"], revision=revision
    )


class XVLAPipeline(nn.Module, DiffusionPipelineProfilerMixin):
    """Single-stage vLLM-Omni pipeline for HuggingFace X-VLA checkpoints."""

    supports_step_execution: bool = False

    def __init__(self, *, od_config: OmniDiffusionConfig, prefix: str = ""):
        super().__init__()
        self.od_config = od_config
        self.prefix = prefix
        self.model_path = _resolve_model_path(
            self.od_config.model, self.od_config.revision
        )
        self.model = self._initialize_model()
        self.weights_sources: list[Any] = []
        self._setup_profiler_targets()

    def _initialize_model(self) -> XVLAModel:
        return XVLAModel(
            model_path=self.model_path,
            od_config=self.od_config,
        )

    def _setup_profiler_targets(self) -> None:
        if not self.od_config.enable_diffusion_pipeline_profiler:
            return
        wrap_methods_by_paths(
            self,
            [
                "model.generate_actions",
                "model._xvla.generate_actions",
                "model._xvla.forward_vlm",
            ],
        )

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        del weights
        return self.model.load_weights(())

    def has_real_checkpoint(self) -> bool:
        return bool(self.model_path and os.path.exists(self.model_path))

    def runtime_mode(self) -> str:
        return (
            "real_checkpoint_loaded"
            if self.has_real_checkpoint()
            else "no_checkpoint_policy"
        )

    def _predict_actions(self, batch_inputs: dict[str, Any]) -> list[Any]:
        actions_tensor = self.model.generate_actions(batch_inputs)
        return [actions_tensor[i].tolist() for i in range(actions_tensor.shape[0])]

    @torch.inference_mode()
    def forward(self, req: OmniDiffusionRequest) -> DiffusionOutput:
        if len(req.prompts) > 1:
            logger.warning("XVLAPipeline only supports a single prompt/request; taking the first sample.")

        extra_args = getattr(req.sampling_params, "extra_args", {}) or {}
        batch_inputs = extra_args.get("batch_inputs")

        if batch_inputs is None:
            logger.warning(
                "XVLAPipeline.forward called without batch_inputs; "
                "returning empty output. Set extra_args['batch_inputs'] "
                "with observation tensors before calling the pipeline."
            )
            return DiffusionOutput(
                output=torch.empty(0, device="cpu"),
                custom_output={
                    "warning": (
                        "missing batch_inputs; XVLA requires "
                        "observation.images.*, observation.task, "
                        "and observation.state inputs"
                    )
                },
                post_process_func=get_xvla_post_process_func(self.od_config),
            )

        actions = self._predict_actions(batch_inputs)
        return DiffusionOutput(
            output=torch.empty(0, device="cpu"),
            custom_output={"actions": actions},
            post_process_func=get_xvla_post_process_func(self.od_config),
        )
