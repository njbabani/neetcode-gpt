import numpy as np
from numpy.typing import NDArray
from typing import Tuple


class Solution:
    def sigmoid(self, z):
        return 1/(1 + np.exp(-z))

    def backward(self, x: NDArray[np.float64], w: NDArray[np.float64], b: float, y_true: float) -> Tuple[NDArray[np.float64], float]:
        # x: 1D input array
        # w: 1D weight array
        # b: scalar bias
        # y_true: true target value
        #
        # Forward: z = dot(x, w) + b, y_hat = sigmoid(z)
        # Loss: L = 0.5 * (y_hat - y_true)^2
        # Return: (dL_dw rounded to 5 decimals, dL_db rounded to 5 decimals)
        
        # Forward pass
        z = np.dot(x, w) + b
        y_hat = self.sigmoid(z)

        # Loss
        L = 0.5 * (y_hat - y_true) ** 2

        # Gradients
        dL_dw = np.round((y_hat - y_true) * y_hat * (1 - y_hat) * x, 5)
        dL_db = np.round((y_hat - y_true) * y_hat * (1 - y_hat), 5)

        return (dL_dw, dL_db)
