MODEL_TRAJECTORY = {
    "lorentz": {
        "param_keys": ("S", "theta0", "gamma"),
        "rel_spans": (0.10, 0.03, 0.20),
        "ns": (20, 10, 20),
        "rel_keep": 1.05,
        "max_keep": None,
        "q_alpha": 0.01,
    },

    "gauss": {
        "param_keys": ("S", "theta0", "sigma"),
        "rel_spans": (0.10, 0.03, 0.20),
        "ns": (20, 11, 20),
        "rel_keep": 1.05,
        "max_keep": None,
        "q_alpha": 0.01,
    },

    "gauss_us": {
        "param_keys": ("S", "theta0", "sigma", "Delta"),
        "rel_spans": (0.10, 0.05, 0.20, 0.20),
        "ns": (20, 10, 20, 20),
        "rel_keep": 1.05,
        "max_keep": None,
        "q_alpha": 0.01,
    },

    "voigt": {
        "param_keys": ("S", "theta0", "sigma", "gamma"),
        "rel_spans": (0.10, 0.03, 0.20, 0.20),
        "ns": (25, 11, 25, 25),
        "rel_keep": 1.05,
        "max_keep": None,
        "q_alpha": 0.01,
    },

    "pvoigt": {
        "param_keys": ("S", "theta0", "H", "eta"),
        "rel_spans": (0.10, 0.03, 0.20, 0.15),
        "ns": (20, 11, 20, 20),
        "rel_keep": 1.05,
        "max_keep": None,
        "q_alpha": 0.01,
    },

    "emg": {
        "param_keys": ("S", "theta0", "sigma", "gamma", "lam"),
        "rel_spans": (0.10, 0.03, 0.20, 0.20, 0.25),
        "ns": (21, 9, 21, 21, 21),
        "rel_keep": 1.05,
        "max_keep": None,
        "q_alpha": 0.01,
    },

    "split_voigt": {
        "param_keys": ("S", "theta0", "beta_Gl", "beta_Cl", "beta_Gr", "beta_Cr"),
        "rel_spans": (0.10, 0.03, 0.20, 0.20, 0.20, 0.20),
        "ns": (9, 9, 9, 9, 9, 9),
        "rel_keep": 1.05,
        "max_keep": None,
        "q_alpha": 0.01,
    },
}