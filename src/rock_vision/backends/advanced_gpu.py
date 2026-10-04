
from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np


@dataclass
class GPUBackendInfo:
    available: bool
    device: str
    device_name: str | None
    torch_version: str | None
    cuda_version: str | None
    total_vram_gb: float | None
    reason: str


def advanced_gpu_available() -> GPUBackendInfo:
    """
    Detect whether the CUDA execution backend is genuinely
    executable through the installed PyTorch environment.

    This is a runtime capability check, not merely an NVIDIA
    hardware check.
    """

    try:
        import torch

        if not torch.cuda.is_available():

            return GPUBackendInfo(
                available=False,
                device="cpu",
                device_name=None,
                torch_version=torch.__version__,
                cuda_version=torch.version.cuda,
                total_vram_gb=None,
                reason="PyTorch CUDA runtime is unavailable.",
            )

        props = torch.cuda.get_device_properties(0)

        total_vram_gb = (
            props.total_memory
            / (1024 ** 3)
        )

        return GPUBackendInfo(
            available=True,
            device="cuda:0",
            device_name=str(props.name),
            torch_version=torch.__version__,
            cuda_version=torch.version.cuda,
            total_vram_gb=round(
                total_vram_gb,
                2
            ),
            reason="CUDA execution backend ready.",
        )

    except Exception as exc:

        return GPUBackendInfo(
            available=False,
            device="cpu",
            device_name=None,
            torch_version=None,
            cuda_version=None,
            total_vram_gb=None,
            reason=f"GPU backend error: {exc}",
        )


class AdvancedGPUBackend:
    """
    Real CUDA execution backend for the GEO AI prototype.

    Current role
    ------------
    GPU-accelerated image normalization and multiscale feature
    preparation.

    Geological semantic interpretation remains in the existing
    validated prototype pipeline until trained geological GPU
    model weights are introduced.

    Therefore this backend must NOT be described as a trained
    geological segmentation model.
    """

    name = "geo-ai-advanced-gpu-v1"

    def __init__(self):

        import torch

        info = advanced_gpu_available()

        if not info.available:
            raise RuntimeError(
                info.reason
            )

        self.torch = torch
        self.device = torch.device(
            "cuda:0"
        )
        self.info = info


    def synchronize(self):

        self.torch.cuda.synchronize(
            self.device
        )


    def preprocess(
        self,
        rgb: np.ndarray,
        detail: str = "Standard"
    ):
        """
        Execute real CUDA preprocessing and feature preparation.

        Returns
        -------
        enhanced_rgb : np.ndarray
            CPU RGB uint8 image compatible with the existing
            geological analysis pipeline.

        metadata : dict
            Runtime provenance describing the actual CUDA run.
        """

        torch = self.torch

        if detail not in (
            "Standard",
            "Detailed",
            "Deep"
        ):
            detail = "Standard"


        passes = {
            "Standard": 1,
            "Detailed": 2,
            "Deep": 3,
        }[detail]


        # ----------------------------------------------------
        # RESET RUNTIME METRICS
        # ----------------------------------------------------

        torch.cuda.reset_peak_memory_stats(
            self.device
        )


        # ----------------------------------------------------
        # RGB -> CUDA
        # ----------------------------------------------------

        x = torch.from_numpy(
            np.ascontiguousarray(rgb).copy()
        )

        x = (
            x
            .to(
                self.device,
                non_blocking=True
            )
            .float()
            .permute(2, 0, 1)
            .unsqueeze(0)
            / 255.0
        )

        self.synchronize()

        t0 = time.perf_counter()


        # ----------------------------------------------------
        # CUDA NORMALIZATION
        # ----------------------------------------------------

        mean = x.mean(
            dim=(2, 3),
            keepdim=True
        )

        std = x.std(
            dim=(2, 3),
            keepdim=True
        ).clamp_min(1e-5)

        feature = (
            x - mean
        ) / std


        # ----------------------------------------------------
        # MULTISCALE FEATURE PREPARATION
        # ----------------------------------------------------

        import torch.nn.functional as F

        for _ in range(passes):

            local_3 = F.avg_pool2d(
                feature,
                kernel_size=3,
                stride=1,
                padding=1
            )

            local_7 = F.avg_pool2d(
                feature,
                kernel_size=7,
                stride=1,
                padding=3
            )

            feature = (
                0.60 * feature
                + 0.25 * local_3
                + 0.15 * local_7
            )


        # ----------------------------------------------------
        # NORMALIZE FEATURE IMAGE
        # ----------------------------------------------------

        lo = feature.amin(
            dim=(2, 3),
            keepdim=True
        )

        hi = feature.amax(
            dim=(2, 3),
            keepdim=True
        )

        enhanced = (
            feature - lo
        ) / (
            hi - lo
        ).clamp_min(1e-5)

        enhanced = (
            enhanced
            .clamp(0, 1)
            .mul(255)
            .byte()
            .squeeze(0)
            .permute(1, 2, 0)
            .cpu()
            .numpy()
        )

        self.synchronize()


        runtime_ms = (
            time.perf_counter()
            - t0
        ) * 1000.0

        peak_vram_mb = (
            torch.cuda.max_memory_allocated(
                self.device
            )
            / (1024 ** 2)
        )


        metadata = {
            "backend":
                self.name,

            "device":
                str(self.device),

            "device_name":
                self.info.device_name,

            "total_vram_gb":
                self.info.total_vram_gb,

            "detail":
                detail,

            "feature_passes":
                passes,

            "cuda_runtime_ms":
                round(
                    runtime_ms,
                    3
                ),

            "peak_vram_mb":
                round(
                    peak_vram_mb,
                    1
                ),

            "torch_version":
                self.info.torch_version,

            "cuda_version":
                self.info.cuda_version,

            "trained_geological_model":
                False,

            "role":
                (
                    "CUDA-accelerated normalization and "
                    "multiscale feature preparation"
                ),
        }

        return (
            enhanced,
            metadata
        )


    def smoke_test(
        self,
        height: int = 1024,
        width: int = 1024
    ):
        """
        Execute an independent CUDA tensor test.
        """

        torch = self.torch

        import torch.nn.functional as F

        torch.cuda.reset_peak_memory_stats(
            self.device
        )

        x = torch.rand(
            (
                1,
                3,
                height,
                width
            ),
            device=self.device
        )

        self.synchronize()

        t0 = time.perf_counter()

        y = F.avg_pool2d(
            x,
            kernel_size=5,
            stride=1,
            padding=2
        )

        y = torch.relu(
            y
        )

        self.synchronize()

        runtime_ms = (
            time.perf_counter()
            - t0
        ) * 1000.0

        peak_vram_mb = (
            torch.cuda.max_memory_allocated(
                self.device
            )
            / (1024 ** 2)
        )

        checksum = float(
            y.mean().item()
        )

        return {
            "backend":
                self.name,

            "device":
                str(self.device),

            "device_name":
                self.info.device_name,

            "total_vram_gb":
                self.info.total_vram_gb,

            "runtime_ms":
                round(
                    runtime_ms,
                    3
                ),

            "peak_vram_mb":
                round(
                    peak_vram_mb,
                    1
                ),

            "checksum":
                round(
                    checksum,
                    6
                ),
        }
