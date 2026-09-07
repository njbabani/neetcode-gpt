import numpy as np
from typing import List


class Solution:
    def linear(self, x, W, b):
        return W @ x + b

    def forward_and_backward(
        self,
        x: List[float],
        W1: List[List[float]],
        b1: List[float],
        W2: List[List[float]],
        b2: List[float],
        y_true: List[float]
    ) -> dict:

        # Convert inputs to NumPy arrays
        x = np.array(x, dtype=float)
        W1 = np.array(W1, dtype=float)
        b1 = np.array(b1, dtype=float)
        W2 = np.array(W2, dtype=float)
        b2 = np.array(b2, dtype=float)
        y_true = np.array(y_true, dtype=float)

        # Forward pass
        z1 = self.linear(x, W1, b1)
        a1 = np.maximum(z1, 0)
        z2 = self.linear(a1, W2, b2)

        # Loss
        mse = np.mean((z2 - y_true) ** 2)

        # Backward pass
        dL_dz2 = 2 / len(z2) * (z2 - y_true)

        # Layer 2 gradients
        dL_dW2 = np.outer(dL_dz2, a1)
        dL_db2 = dL_dz2

        # Backprop through layer 2
        dL_da1 = W2.T @ dL_dz2

        # Backprop through ReLU
        dL_dz1 = dL_da1 * (z1 > 0)

        # Layer 1 gradients
        dL_dW1 = np.outer(dL_dz1, x)
        dL_db1 = dL_dz1

        return {
            "loss": round(float(mse), 4),
            "dW1": np.round(dL_dW1, 4).tolist(),
            "db1": np.round(dL_db1, 4).tolist(),
            "dW2": np.round(dL_dW2, 4).tolist(),
            "db2": np.round(dL_db2, 4).tolist(),
        }
