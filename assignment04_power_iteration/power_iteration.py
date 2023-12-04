import numpy as np

def get_dominant_eigenvalue_and_eigenvector(data, num_steps):
    """
    data: np.ndarray – symmetric diagonalizable real-valued matrix
    num_steps: int – number of power method steps

    Returns:
    eigenvalue: float – dominant eigenvalue estimation after `num_steps` steps
    eigenvector: np.ndarray – corresponding eigenvector estimation
    """
    ### YOUR CODE HERE
    n = data.shape[0]
    x = np.random.rand(n)
    x /= np.linalg.norm(x)
    for i in range(num_steps):
        value = np.dot(x,data @ x)
        x_new = data @ x / np.linalg.norm(data @x)
        if np.linalg.norm(x - x_new) < 1e-6:
            break

        x = x_new
    return float(value), x