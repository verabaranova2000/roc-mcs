# kernels.py
from scipy.signal import fftconvolve
import numpy as np

# ==================================================
# Ultrasound kernel
# ==================================================
def harmonic_kernel(dtheta, Delta):
    """
    Delta  = Δθₚ  (амплитуда oscillatory Bragg-angle shift)
             Δθₚ = tanθᵦ₀ · u₀ · k · cos(kxₚ)
    dtheta = δθ

    Возвращает распределение P(δθ) = 1 / [π sqrt(Delta² - δθ²)].
    """  
    P = np.zeros_like(dtheta)
    mask = np.abs(dtheta) < Delta
    P[mask] = 1 / (np.pi * np.sqrt(Delta**2 - dtheta[mask]**2))
    P /= np.trapezoid(P, dtheta)
    return P

def ultrasound_RC(I0, theta, Delta):
    """
    Берем исходную rocking curve, сдвигаем её во времени, усредняем.
    
    fftconvolve - это реализация I_avg = F⊗P
                  I_avg(θ) = ∫F(θ−δθ)P(δθ)dδθ
    """
    dtheta = theta[1]-theta[0]
    P = harmonic_kernel(theta,Delta)
    I_US = fftconvolve(I0, P, mode='same')*dtheta
    return I_US
