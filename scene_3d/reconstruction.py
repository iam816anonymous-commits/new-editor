"""
Scene Complexity Router and Adaptive Execution Backend Abstraction for scene_3d/.
"""

from enum import Enum
from dataclasses import dataclass
from typing import Dict, Any, Tuple, Optional, List
import numpy as np
import cv2
import torch


class SceneComplexityTier(str, Enum):
    TIER1_SIMPLE = "TIER 1 - SIMPLE"
    TIER2_MODERATE = "TIER 2 - MODERATE"
    TIER3_COMPLEX = "TIER 3 - COMPLEX"
    TIER4_EXTREME = "TIER 4 - EXTREME"
    # Legacy backward compatibility aliases
    SIMPLE = "TIER 1 - SIMPLE"
    MODERATE = "TIER 2 - MODERATE"
    COMPLEX = "TIER 3 - COMPLEX"


class ExecutionBackendType(str, Enum):
    CPU = "CPU"
    CUDA = "CUDA"
    HYBRID = "HYBRID"


@dataclass
class HardwareProfile:
    has_cuda: bool
    device_name: str
    vram_gb: float
    ram_gb: float
    cpu_cores: int

    @classmethod
    def detect(cls) -> 'HardwareProfile':
        import os
        has_cuda = torch.cuda.is_available()
        vram_gb = 0.0
        dev_name = "CPU"
        if has_cuda:
            try:
                vram_bytes = torch.cuda.get_device_properties(0).total_memory
                vram_gb = vram_bytes / (1024**3)
                dev_name = torch.cuda.get_device_name(0)
            except Exception:
                vram_gb = 4.0
                dev_name = "CUDA Device"

        ram_gb = 16.0
        try:
            import psutil
            ram_gb = psutil.virtual_memory().total / (1024**3)
        except Exception:
            ram_gb = 16.0

        cpu_cores = os.cpu_count() or 4
        return cls(has_cuda=has_cuda, device_name=dev_name, vram_gb=round(vram_gb, 2), ram_gb=round(ram_gb, 2), cpu_cores=cpu_cores)


@dataclass
class QualityProfile:
    profile_name: str  # CPU_FAST, CPU_QUALITY, GPU_BALANCED, GPU_HIGH, GPU_ULTRA
    output_resolution: Tuple[int, int]  # (W, H)
    reconstruction_resolution: Tuple[int, int]  # (W, H)
    depth_model: str
    segmentation_model: str
    depth_refinement_strength: float
    splat_resolution: Tuple[int, int]
    inpainting_quality: str
    temporal_quality: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "profile_name": self.profile_name,
            "output_resolution": f"{self.output_resolution[0]}x{self.output_resolution[1]}",
            "reconstruction_resolution": f"{self.reconstruction_resolution[0]}x{self.reconstruction_resolution[1]}",
            "depth_model": self.depth_model,
            "segmentation_model": self.segmentation_model,
            "depth_refinement_strength": self.depth_refinement_strength,
            "inpainting_quality": self.inpainting_quality,
            "temporal_quality": self.temporal_quality
        }


def estimate_render_memory_bytes(
    resolution: Tuple[int, int],
    frame_count: int = 48,
    is_cuda: bool = False
) -> Dict[str, float]:
    """Estimates peak RAM and VRAM footprint in Gigabytes for a render job."""
    w, h = resolution
    pixels = w * h
    frame_buf_mb = (pixels * 3 * 4 * frame_count) / (1024**2)
    model_ram_mb = 1200.0
    model_vram_mb = 1800.0 if is_cuda else 0.0

    est_ram_gb = round((model_ram_mb + frame_buf_mb + 500.0) / 1024.0, 2)
    est_vram_gb = round((model_vram_mb + (frame_buf_mb * 0.5 if is_cuda else 0.0)) / 1024.0, 2)

    return {
        "estimated_ram_gb": est_ram_gb,
        "estimated_vram_gb": est_vram_gb,
        "frame_buffer_mb": round(frame_buf_mb, 2)
    }


@dataclass
class QualityDecision:
    hardware: HardwareProfile
    backend: ExecutionBackendType
    scene_complexity_tier: SceneComplexityTier
    source_resolution: Tuple[int, int]
    reconstruction_resolution: Tuple[int, int]
    output_resolution: Tuple[int, int]
    selected_quality_profile: QualityProfile
    estimated_ram_gb: float
    estimated_vram_gb: float
    downgrade_reason: Optional[str] = None
    is_native_reconstruction: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return {
            "hardware": {
                "has_cuda": self.hardware.has_cuda,
                "device_name": self.hardware.device_name,
                "vram_gb": self.hardware.vram_gb,
                "ram_gb": self.hardware.ram_gb,
                "cpu_cores": self.hardware.cpu_cores
            },
            "backend": self.backend.value,
            "scene_complexity_tier": self.scene_complexity_tier.value,
            "source_resolution": f"{self.source_resolution[0]}x{self.source_resolution[1]}",
            "reconstruction_resolution": f"{self.reconstruction_resolution[0]}x{self.reconstruction_resolution[1]}",
            "output_resolution": f"{self.output_resolution[0]}x{self.output_resolution[1]}",
            "selected_quality_profile": self.selected_quality_profile.to_dict(),
            "estimated_ram_gb": self.estimated_ram_gb,
            "estimated_vram_gb": self.estimated_vram_gb,
            "downgrade_reason": self.downgrade_reason,
            "is_native_reconstruction": self.is_native_reconstruction
        }


class QualityPlanner:
    """
    Evidence-based quality planner taking HardwareProfile, SceneComplexityTier,
    InputResolution, and UserQualityPreference to generate a QualityDecision.
    """

    @staticmethod
    def plan(
        hardware: HardwareProfile,
        complexity_tier: SceneComplexityTier,
        input_resolution: Tuple[int, int],
        requested_quality: str = "auto",
        requested_resolution: str = "auto",
        frame_count: int = 48
    ) -> QualityDecision:
        w_in, h_in = input_resolution
        req_q = requested_quality.lower()
        req_res = requested_resolution.lower()

        downgrade_reasons = []

        res_targets = {
            "480p": (854, 480),
            "720p": (1280, 720),
            "1080p": (1920, 1080),
            "1440p": (2560, 1440),
            "4k": (3840, 2160)
        }

        # 1. Determine Target Output Resolution & Reconstruction Resolution
        if not hardware.has_cuda: # CPU Mode
            backend_type = ExecutionBackendType.CPU
            # CPU Hard Ceiling: <= 720p
            if req_res in ["1080p", "1440p", "4k"]:
                downgrade_reasons.append(f"CPU execution constrained: requested {req_res} downgraded to 720p ceiling for resource safety.")
                target_res = (1280, 720)
            elif complexity_tier == SceneComplexityTier.SIMPLE and req_res == "auto":
                target_res = (854, 480)
            else:
                target_res = (min(w_in, 1280), min(h_in, 720)) if req_res == "auto" else res_targets.get(req_res, (1280, 720))

            profile_name = "CPU_FAST" if target_res[1] <= 480 else "CPU_QUALITY"
        else: # CUDA / GPU Mode
            backend_type = ExecutionBackendType.CUDA
            if hardware.vram_gb < 4.0:
                max_res = (1280, 720)
                if req_res in ["1080p", "1440p", "4k"]:
                    downgrade_reasons.append(f"VRAM constrained ({hardware.vram_gb:.1f}GB < 4.0GB): downgraded to 720p.")
            elif hardware.vram_gb < 8.0:
                max_res = (1920, 1080)
                if req_res in ["1440p", "4k"]:
                    downgrade_reasons.append(f"VRAM constrained ({hardware.vram_gb:.1f}GB < 8.0GB): downgraded to 1080p.")
            else:
                max_res = (3840, 2160)

            if req_res in res_targets:
                req_pair = res_targets[req_res]
                target_res = (min(req_pair[0], max_res[0]), min(req_pair[1], max_res[1]))
            else:
                target_res = (min(w_in, max_res[0]), min(h_in, max_res[1]))

            if target_res[1] >= 2160:
                profile_name = "GPU_ULTRA"
            elif target_res[1] >= 1440:
                profile_name = "GPU_HIGH"
            else:
                profile_name = "GPU_BALANCED"

        recon_res = (min(w_in, target_res[0]), min(h_in, target_res[1]))
        is_native = bool(recon_res[0] >= target_res[0] and recon_res[1] >= target_res[1])

        profile = QualityProfile(
            profile_name=profile_name,
            output_resolution=target_res,
            reconstruction_resolution=recon_res,
            depth_model="Depth-Anything-V2-Small-hf",
            segmentation_model="sam2-hiera-tiny",
            depth_refinement_strength=0.85,
            splat_resolution=target_res,
            inpainting_quality="HIGH" if hardware.has_cuda else "STANDARD",
            temporal_quality="STABLE"
        )

        mem_est = estimate_render_memory_bytes(target_res, frame_count, is_cuda=hardware.has_cuda)

        return QualityDecision(
            hardware=hardware,
            backend=backend_type,
            scene_complexity_tier=complexity_tier,
            source_resolution=(w_in, h_in),
            reconstruction_resolution=recon_res,
            output_resolution=target_res,
            selected_quality_profile=profile,
            estimated_ram_gb=mem_est["estimated_ram_gb"],
            estimated_vram_gb=mem_est["estimated_vram_gb"],
            downgrade_reason="; ".join(downgrade_reasons) if downgrade_reasons else None,
            is_native_reconstruction=is_native
        )


@dataclass
class ExecutionBackend:
    backend_type: ExecutionBackendType
    device: torch.device
    precision: torch.dtype = torch.float32

    @classmethod
    def auto_select(cls) -> 'ExecutionBackend':
        if torch.cuda.is_available():
            return cls(backend_type=ExecutionBackendType.CUDA, device=torch.device("cuda"))
        return cls(backend_type=ExecutionBackendType.CPU, device=torch.device("cpu"))


class SceneComplexityAnalyzer:
    """
    Evaluates image-space signals (depth variance, subject count, edge density, disocclusion risk)
    to classify scene reconstruction difficulty tier.
    """

    @staticmethod
    def analyze(
        rgb_array: np.ndarray,
        depth_map: np.ndarray,
        subject_mask: np.ndarray
    ) -> Tuple[SceneComplexityTier, float, Dict[str, Any]]:
        depth_std = float(np.std(depth_map))
        sub_area_pct = float(np.sum(subject_mask) / max(1, subject_mask.size))

        # Compute boundary edge density
        sub_uint = (subject_mask * 255).astype(np.uint8)
        contours, _ = cv2_find_contours(sub_uint)
        peri = float(sum(len(c) for c in contours))
        edge_density = peri / max(1.0, float(np.sum(subject_mask)))

        # Depth discontinuity density
        gx = cv2.Sobel(depth_map, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(depth_map, cv2.CV_32F, 0, 1, ksize=3)
        grad_mag = np.sqrt(gx**2 + gy**2)
        depth_discont_pct = float(np.mean(grad_mag > 0.5))

        score = 0.35 * min(1.0, depth_std / 2.0) + 0.25 * sub_area_pct + 0.25 * min(1.0, edge_density * 5.0) + 0.15 * min(1.0, depth_discont_pct * 10.0)

        if score >= 0.80:
            tier = SceneComplexityTier.TIER4_EXTREME
        elif score >= 0.55:
            tier = SceneComplexityTier.TIER3_COMPLEX
        elif score >= 0.25:
            tier = SceneComplexityTier.TIER2_MODERATE
        else:
            tier = SceneComplexityTier.TIER1_SIMPLE

        details = {
            "complexity_score": round(score, 4),
            "depth_std": round(depth_std, 4),
            "subject_area_pct": round(sub_area_pct, 4),
            "edge_density": round(edge_density, 4)
        }
        return tier, score, details


def cv2_find_contours(mask_uint8: np.ndarray):
    import cv2
    return cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
