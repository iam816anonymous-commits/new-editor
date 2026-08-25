from typing import Tuple, Dict, Any, List, Optional, Union
import numpy as np

Point2D = Tuple[float, float]
Point3D = Tuple[float, float, float]
Shape2D = Tuple[int, int]
BoundingBox = Tuple[int, int, int, int]

RGBImageArray = np.ndarray
DepthMapArray = np.ndarray
MaskArray = np.ndarray
Matrix3x3 = np.ndarray
Matrix4x4 = np.ndarray
Vector3 = np.ndarray
