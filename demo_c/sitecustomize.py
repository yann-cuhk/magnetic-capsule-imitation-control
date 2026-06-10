import numpy as np
import warnings

warnings.filterwarnings(
    "ignore",
    message="scipy.misc is deprecated and will be removed in 2.0.0",
    category=DeprecationWarning,
)

if not hasattr(np, "mat"):
    np.mat = np.asmatrix
