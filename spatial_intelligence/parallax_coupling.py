"""
Attachment Graph, Motion Coupling & Eligibility Subsystem for First-Principles Cinematic 2.5D Renderer.

Defines AttachmentType, MotionEligibility, MotionCouplingGroup, and reasoning algorithms
that connect semantically related object regions into unified 3D motion fields while respecting
continuous pixel-level 3D depth geometry.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Tuple, Optional, Any
import numpy as np
import cv2


class AttachmentType(str, Enum):
    RIGIDLY_ATTACHED = "RIGIDLY_ATTACHED"
    SOFTLY_ATTACHED = "SOFTLY_ATTACHED"
    INDEPENDENT = "INDEPENDENT"
    SUPPORTED_BY = "SUPPORTED_BY"
    RESTING_ON = "RESTING_ON"
    BEHIND = "BEHIND"
    IN_FRONT_OF = "IN_FRONT_OF"
    PART_OF = "PART_OF"
    SAME_SURFACE = "SAME_SURFACE"
    UNKNOWN = "UNKNOWN"


class MotionEligibility(str, Enum):
    PRIMARY_CAMERA_PARALLAX = "PRIMARY_CAMERA_PARALLAX"
    SECONDARY_PARALLAX = "SECONDARY_PARALLAX"
    BACKGROUND_PARALLAX = "BACKGROUND_PARALLAX"
    STATIC_REFERENCE = "STATIC_REFERENCE"
    ATTACHED_TO_GROUP = "ATTACHED_TO_GROUP"
    INDEPENDENT_ELEMENT = "INDEPENDENT_ELEMENT"
    MOTION_SUPPRESSED = "MOTION_SUPPRESSED"
    UNCERTAIN = "UNCERTAIN"


@dataclass
class MotionCouplingGroup:
    """
    Unified motion group linking semantically and physically connected regions.
    Members share the same SE(3) camera-induced motion field while obeying local 3D pixel depth.
    """
    group_id: str
    group_name: str
    primary_region_id: str
    member_region_ids: List[str]
    attachment_types: Dict[str, AttachmentType]
    group_motion_eligibility: MotionEligibility
    rigid_motion_shared: bool
    confidence: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "group_id": self.group_id,
            "group_name": self.group_name,
            "primary_region_id": self.primary_region_id,
            "member_region_ids": self.member_region_ids,
            "attachment_types": {k: v.value for k, v in self.attachment_types.items()},
            "group_motion_eligibility": self.group_motion_eligibility.value,
            "rigid_motion_shared": self.rigid_motion_shared,
            "confidence": round(float(self.confidence), 4),
        }


def infer_region_attachments(
    regions: List[Any],
    depth_map: np.ndarray,
    rgb_array: np.ndarray
) -> List[Dict[str, Any]]:
    """
    Infers attachment relationships between pairs of spatial parallax regions based on
    boundary contact, containment, spatial proximity, and depth continuity.
    """
    attachments = []
    n = len(regions)

    for i in range(n):
        rA = regions[i]
        for j in range(i + 1, n):
            rB = regions[j]

            # Boundary contact check (dilation intersection)
            mA_dil = cv2.dilate((rA.mask * 255).astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
            mB_dil = cv2.dilate((rB.mask * 255).astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
            contact = np.any(mA_dil & rB.mask) or np.any(mB_dil & rA.mask)

            depth_diff = abs(rA.depth_mean - rB.depth_mean)

            if rA.semantic_role == "PRIMARY_SUBJECT" and rB.semantic_role == "PRIMARY_SUBJECT":
                att_type = AttachmentType.RIGIDLY_ATTACHED
                conf = 0.95
            elif contact and depth_diff < 0.8:
                att_type = AttachmentType.SAME_SURFACE if depth_diff < 0.2 else AttachmentType.SOFTLY_ATTACHED
                conf = 0.80
            elif contact and rA.depth_mean < rB.depth_mean:
                att_type = AttachmentType.SUPPORTED_BY if "SURFACE" in rB.semantic_role else AttachmentType.IN_FRONT_OF
                conf = 0.75
            else:
                att_type = AttachmentType.INDEPENDENT
                conf = 0.90

            attachments.append({
                "source_region_id": rA.region_id,
                "target_region_id": rB.region_id,
                "attachment_type": att_type,
                "confidence": conf,
                "depth_delta": round(float(depth_diff), 4),
                "boundary_contact": bool(contact)
            })

    return attachments


def construct_motion_coupling_groups(
    regions: List[Any],
    attachments: List[Dict[str, Any]]
) -> List[MotionCouplingGroup]:
    """
    Groups regions into MotionCouplingGroups ensuring semantically attached parts
    (e.g., face, torso, hands, ornaments) share a unified rigid motion group.
    """
    groups = []

    # Primary subject regions
    primary_regions = [r for r in regions if r.semantic_role == "PRIMARY_SUBJECT"]
    if primary_regions:
        main_reg = primary_regions[0]
        member_ids = [r.region_id for r in primary_regions]
        att_map = {r.region_id: AttachmentType.RIGIDLY_ATTACHED for r in primary_regions}

        groups.append(
            MotionCouplingGroup(
                group_id="GROUP_PRIMARY_SUBJECT",
                group_name="PRIMARY_SUBJECT_RIGID_GROUP",
                primary_region_id=main_reg.region_id,
                member_region_ids=member_ids,
                attachment_types=att_map,
                group_motion_eligibility=MotionEligibility.PRIMARY_CAMERA_PARALLAX,
                rigid_motion_shared=True,
                confidence=0.95
            )
        )

    # Background & environment regions
    env_regions = [r for r in regions if r.semantic_role != "PRIMARY_SUBJECT"]
    for idx, r in enumerate(env_regions):
        groups.append(
            MotionCouplingGroup(
                group_id=f"GROUP_ENV_{idx:02d}",
                group_name=f"ENVIRONMENT_GROUP_{r.region_id}",
                primary_region_id=r.region_id,
                member_region_ids=[r.region_id],
                attachment_types={r.region_id: AttachmentType.INDEPENDENT},
                group_motion_eligibility=MotionEligibility.BACKGROUND_PARALLAX,
                rigid_motion_shared=False,
                confidence=0.85
            )
        )

    return groups
