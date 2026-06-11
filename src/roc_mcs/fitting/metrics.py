# metrics.py
import numpy as np

def r_factor(y_exp: np.ndarray, y_fit: np.ndarray) -> float:
    return np.sum(np.abs(y_exp - y_fit)) / np.sum(np.abs(y_exp))

def rmse(y_exp: np.ndarray, y_fit: np.ndarray) -> float:
    return np.sqrt(np.mean((y_exp - y_fit) ** 2))

def nrmse(y_exp: np.ndarray, y_fit: np.ndarray) -> float:
    return rmse(y_exp, y_fit) / np.mean(np.abs(y_exp))

def cosine_similarity(y_exp: np.ndarray, y_fit: np.ndarray) -> float:
    return np.sum(y_exp * y_fit) / np.sqrt(np.sum(y_exp**2) * np.sum(y_fit**2))


def chi_square(y_exp: np.ndarray, y_fit: np.ndarray, sigma: np.ndarray) -> float:
    """ Classical chi-square. """
    return np.sum(((y_exp - y_fit) / sigma) ** 2)

def reduced_chi_square(y_exp: np.ndarray, y_fit: np.ndarray, sigma: np.ndarray, n_params: int) -> float:
    """  Classical reduced chi-square. """
    dof = len(y_exp) - n_params
    if dof <= 0:
        raise ValueError("Degrees of freedom must be positive.")
    return chi_square(y_exp, y_fit, sigma) / dof

def poisson_reduced_chi_square(y_exp: np.ndarray, y_fit: np.ndarray, n_params: int) -> float:
    """
    Reduced chi-square assuming Poisson counting statistics.

    sigma_i = sqrt(I_exp,i)
    """
    dof = len(y_exp) - n_params
    if dof <= 0:
        raise ValueError("Degrees of freedom must be positive.")
    y_exp_safe = np.clip(y_exp, 1.0, None)
    return np.sum((y_exp_safe - y_fit) ** 2 / y_exp_safe) / dof



# ==================================================
# summary
# ==================================================
def compute_fit_metrics(y_exp: np.ndarray, y_fit: np.ndarray, n_params: int,) -> dict[str, float]:
    return {
        "R_factor": r_factor(y_exp, y_fit),
        "RMSE": rmse(y_exp, y_fit),
        "NRMSE": nrmse(y_exp, y_fit),
        "chi2_red": poisson_reduced_chi_square(y_exp, y_fit, n_params=n_params,),  # "χ²_red"
        "eta": cosine_similarity(y_exp, y_fit),   # "η"
    }