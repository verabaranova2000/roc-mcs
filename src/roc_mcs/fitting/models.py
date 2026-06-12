import numpy as np
from scipy.special import voigt_profile

from roc_mcs.fitting.kernels import ultrasound_RC

# ==================================================
# Base profiles
# ==================================================
"""
Нормированные профили:   ∫f(θ)dθ = 1

np.trapezoid(...)
    Численная перенормировка, потому что интегрируем не по всей прямой, 
    а по конечному диапазону theta, т.е. хвосты обрезаются -> площадь не дооценивается
"""

def profile_lorentz(th, gamma):
    L = (1 / np.pi) * gamma / (th**2 + gamma**2)
    L /= np.trapezoid(L, th)
    return L

def profile_gauss(th, sigma):
    G = np.exp(-(th**2) / (2 * sigma**2))
    G /= np.trapezoid(G, th)
    return G

def profile_voigt(th, sigma, gamma):
    V = voigt_profile(th, sigma, gamma)
    V /= np.trapezoid(V, th)
    return V  

def profile_pvoigt(th, H, eta):
    sigma = H / (2*np.sqrt(2*np.log(2)))
    gamma = H / 2
    G = profile_gauss(th, sigma)
    L = profile_lorentz(th, gamma)
    return eta*L + (1-eta)*G
    

def profile_emg(theta, sigma, gamma, lam):
    V = voigt_profile(theta, sigma, gamma)
    tail = np.exp(-lam * np.clip(theta, 0, None))
    return V * tail

def split_voigt(x, beta_Gl, beta_Cl, beta_Gr, beta_Cr):
    """
    Split Voigt almost exactly in the spirit of Sánchez-Bajo (2025).
    Unit-area split Voigt: continuous at x=0, area = 1.
    """
    _EPS = 1e-14
    def _pos(x, eps=_EPS):
        return np.maximum(np.asarray(x, dtype=float), eps)
    
    x = np.asarray(x, dtype=float)

    # Article parameters -> SciPy parameters
    sigma_l = _pos(beta_Gl / np.sqrt(2 * np.pi))
    gamma_l = _pos(beta_Cl / np.pi)
    sigma_r = _pos(beta_Gr / np.sqrt(2 * np.pi))
    gamma_r = _pos(beta_Cr / np.pi)

    # Peak values of the two Voigt halves at x=0
    v0_l = _pos(voigt_profile(0.0, sigma_l, gamma_l))
    v0_r = _pos(voigt_profile(0.0, sigma_r, gamma_r))

    y = np.empty_like(x)
    left = x <= 0

    # Continuity at the join: both halves equal 1 at x=0 before area normalization
    y[left] = voigt_profile(x[left], sigma_l, gamma_l) / v0_l
    y[~left] = voigt_profile(x[~left], sigma_r, gamma_r) / v0_r

    # Integral breadth of the split profile: beta = (beta_l + beta_r)/2
    beta_l = 1.0 / v0_l
    beta_r = 1.0 / v0_r
    beta = 0.5 * (beta_l + beta_r)

    return y / beta

# ==================================================
# No-US models
# ==================================================
def model_lorentz(theta, S, theta0, gamma):
    return S * profile_lorentz(theta - theta0, gamma)

def model_gauss(theta, S, theta0, sigma):
    return S * profile_gauss(theta - theta0, sigma)

def model_voigt(theta, S, theta0, sigma, gamma):
    return S * profile_voigt(theta - theta0, sigma, gamma)

def model_pvoigt(theta, S, theta0, H, eta):
    return S * profile_pvoigt(theta - theta0, H, eta)    

def model_emg(theta, S, theta0, sigma, gamma, lam):
    x = theta - theta0
    return S * profile_emg(x, sigma, gamma, lam)

def model_split_voigt(theta, S, theta0, beta_Gl, beta_Cl, beta_Gr, beta_Cr):
    x = theta - theta0
    return S * split_voigt(x, beta_Gl, beta_Cl, beta_Gr, beta_Cr)    

# ==================================================
# US models
# ==================================================
def model_lorentz_us(theta, S, theta0, gamma, Delta):
    I0 = model_lorentz(theta, S, theta0, gamma)
    return ultrasound_RC(I0, theta, Delta)

def model_gauss_us(theta, S, theta0, sigma, Delta):
    I0 = model_gauss(theta, S, theta0, sigma)
    return ultrasound_RC(I0, theta, Delta)

def model_voigt_us(theta, S, theta0, sigma, gamma, Delta):
    I0 = model_voigt(theta, S, theta0, sigma, gamma)
    return ultrasound_RC(I0, theta, Delta) 