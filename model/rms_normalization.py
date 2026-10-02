import numpy as np
from typing import List

def debug(x: Any):
    print(f"Value: {x} | Type: {type(x)}")

class Solution:

    def rms_norm(self, x: List[float], gamma: List[float], eps: float) -> List[float]:
        # Implement RMS Normalization (similar to LayerNorm but without mean centering or beta)
        # Normalize x, then scale by gamma
        # Return result rounded to 4 decimal places as a list
        x = np.array(x)
        gamma = np.array(gamma)
        mean_sq = np.mean(x ** 2) + eps
        x_hat = x / np.sqrt(mean_sq)
        output = gamma * x_hat
        return np.round(output, 4).tolist()
