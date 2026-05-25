from vllm_omni.config.stage_config import (
    PipelineConfig,
    StageExecutionType,
    StagePipelineConfig,
)

XVLA_SINGLE_STAGE_PIPELINE = PipelineConfig(
    model_type="xvla_single_stage",
    model_arch="XVLAPipeline",
    hf_architectures=(),
    stages=(
        StagePipelineConfig(
            stage_id=0,
            model_stage="dit",
            execution_type=StageExecutionType.DIFFUSION,
            input_sources=(),
            final_output=True,
            final_output_type="action",
        ),
    ),
)