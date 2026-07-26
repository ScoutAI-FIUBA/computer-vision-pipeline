"""Deployment-agnostic basketball video processing pipeline.

This package contains plain Python only. It must never import Modal, FastAPI, or
any other deployment framework, so it stays runnable from a laptop, a Colab
notebook, or a GPU container without changes.

Entry point: `pipeline.runner.run_pipeline`.
"""

from pipeline.config import PipelineConfig
from pipeline.results import PipelineResults

__all__ = ["PipelineConfig", "PipelineResults"]
