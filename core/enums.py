from enum import Enum

class RenderMode(str, Enum):
    AUTO = "auto"
    MODE_2_5D = "2.5d"
    MODE_3D = "3d"

class MotionStyle(str, Enum):
    CINEMATIC_PUSH_IN = "Cinematic Push-In"
    DOLLY_IN = "Dolly In"
    DOLLY_OUT = "Dolly Out"
    HORIZONTAL_PAN = "Horizontal Pan"
    VERTICAL_PAN = "Vertical Pan"
    ORBIT = "Orbit"
    MICRO_ORBIT = "Micro Orbit"

class MotionStrength(str, Enum):
    SUBTLE = "Subtle"
    CINEMATIC = "Cinematic"
    STRONG = "Strong"

class QualityTier(str, Enum):
    AUTO = "auto"
    FAST = "fast"
    BALANCED = "balanced"
    HIGH = "high"
    ULTRA = "ultra"
