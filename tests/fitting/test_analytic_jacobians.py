from __future__ import annotations

from time import perf_counter
import numpy as np
import pandas as pd
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

from roc_mcs.fitting.registry import MODEL_SPECS
import roc_mcs.fitting.backends.jax_backend as jaxb


def _block(x):
    if isinstance(x, (tuple, list)):
        for z in x: _block(z)
    elif hasattr(x, "block_until_ready"):
        x.block_until_ready()
    return x


def _seed_curve(theta):
    th = np.asarray(theta, float)
    c = float(np.median(th))
    w = max(float(np.ptp(th)) / 10, float(np.median(np.diff(th))))
    return 10.0 + 1000.0 * np.exp(-0.5 * ((th - c) / w) ** 2)


def _sample_params(spec, theta, n, rng):
    th = np.asarray(theta, float)
    dtheta = float(np.median(np.diff(th)))
    guess = spec.make_initial_guess(th, _seed_curve(th))
    bounds = spec.make_bounds(guess, dtheta=dtheta)
    cols = []
    for name in spec.param_names:
        g = float(guess[name]); lo, hi = map(float, bounds[name])
        scale = max(abs(g), abs(dtheta), 1.0)
        if np.isfinite(lo) and np.isfinite(hi):
            if hi < lo: raise ValueError(f"{spec.name}.{name}: upper<lower")
            if hi == lo: x = np.full(int(n), lo)
            else:
                eps = max((hi - lo) * 1e-6, np.finfo(float).eps * max(1, abs(lo), abs(hi)))
                a, b = lo + eps, hi - eps
                x = rng.uniform(a, b, int(n)) if b > a else np.full(int(n), 0.5 * (lo + hi))
        elif np.isfinite(lo):
            a = lo + max(scale * 1e-6, 1e-12)
            b = max(a + scale, min(max(g, a) + 2 * scale, a + 4 * scale))
            x = rng.uniform(a, b, int(n))
        elif np.isfinite(hi):
            b = hi - max(scale * 1e-6, 1e-12)
            a = min(b - scale, max(min(g, b) - 2 * scale, b - 4 * scale))
            x = rng.uniform(a, b, int(n))
        else:
            x = g + rng.normal(0, 0.5 * scale, int(n))
        cols.append(x)
    return np.column_stack(cols)


def _one_spec(spec, theta, *, n=4096, repeats=7, seed=20261016, rtol=1e-8, atol=1e-10):
    if spec.jax_func is None or getattr(spec, "jax_jac_func", None) is None:
        raise ValueError(f"{spec.name}: jax_func или analytic jax_jac_func не зарегистрирован")
    rng = np.random.default_rng(int(seed))
    P = _sample_params(spec, theta, int(n), rng)
    th, pj = jnp.asarray(theta, jnp.float64), jnp.asarray(P, jnp.float64)

    def ref_single(p):
        def f(q): return spec.jax_func(th, *q)
        pred = f(p)
        return pred, jax.jacfwd(f)(p)

    ref = jax.jit(jax.vmap(ref_single))
    ana = jax.jit(jax.vmap(lambda p: spec.jax_jac_func(th, *p)))

    t = perf_counter(); rr = ref(pj); _block(rr); compile_ref = perf_counter() - t
    t = perf_counter(); aa = ana(pj); _block(aa); compile_ana = perf_counter() - t
    pred_r, Jr = map(np.asarray, rr); pred_a, Ja = map(np.asarray, aa)
    finite = bool(all(np.all(np.isfinite(x)) for x in (pred_r, Jr, pred_a, Ja)))
    dp, dJ = pred_a - pred_r, Ja - Jr
    allclose = finite and np.allclose(pred_a, pred_r, rtol=rtol, atol=atol) and np.allclose(Ja, Jr, rtol=rtol, atol=atol)
    numpy_ok, numpy_max = np.nan, np.nan
    batch_jac = getattr(spec, "batch_jac_func", None)
    if batch_jac is not None:
        Jn = np.asarray(batch_jac(np.asarray(theta, float), *[P[:, i] for i in range(P.shape[1])]), float)
        numpy_ok = bool(np.all(np.isfinite(Jn)) and np.allclose(Ja, Jn, rtol=rtol, atol=max(atol, 1e-9)))
        numpy_max = float(np.max(np.abs(Ja - Jn)))

    tref, tana = [], []
    for _ in range(int(repeats)):
        t = perf_counter(); z = ref(pj); _block(z); tref.append(perf_counter() - t)
        t = perf_counter(); z = ana(pj); _block(z); tana.append(perf_counter() - t)
    sr, sa = float(np.median(tref)), float(np.median(tana))
    den = np.maximum(np.abs(Jr), atol)
    return {"model": spec.name, "n": int(n), "finite": finite, "allclose": bool(allclose),
            "max_abs_pred": float(np.max(np.abs(dp))), "max_abs_jac": float(np.max(np.abs(dJ))),
            "max_rel_jac": float(np.max(np.abs(dJ) / den)), "numpy_jac_allclose": numpy_ok,
            "max_abs_vs_numpy": numpy_max, "compile_autodiff_s": compile_ref,
            "compile_analytic_s": compile_ana, "steady_autodiff_s": sr,
            "steady_analytic_s": sa, "speedup": sr / sa if sa > 0 else np.inf}


def benchmark_registered_analytic_jacobians(theta, *, n=4096, repeats=7, seed=20261016,
                                              rtol=1e-8, atol=1e-10, require_pass=True):
    """Сравнивает зарегистрированные analytic JAX Jacobian с jacfwd и NumPy-Jacobian."""
    th = np.asarray(theta, float)
    if th.ndim != 1 or len(th) < 5 or np.any(~np.isfinite(th)) or np.any(np.diff(th) <= 0):
        raise ValueError("theta должна быть конечной возрастающей 1D-сеткой минимум из 5 точек")
    if int(n) < 1 or int(repeats) < 1: raise ValueError("n и repeats должны быть положительными")
    specs = [s for s in MODEL_SPECS.values() if getattr(s, "jax_jac_func", None) is not None]
    if not specs: raise RuntimeError("В MODEL_SPECS нет моделей с jax_jac_func")
    rows = [_one_spec(s, th, n=n, repeats=repeats, seed=int(seed) + i, rtol=rtol, atol=atol)
            for i, s in enumerate(specs)]
    df = pd.DataFrame(rows)
    if require_pass:
        bad = (~df["finite"]) | (~df["allclose"])
        has_numpy = df["numpy_jac_allclose"].notna()
        numpy_bad = pd.Series(False, index=df.index)
        numpy_bad.loc[has_numpy] = ~df.loc[has_numpy, "numpy_jac_allclose"].astype(bool)
        bad |= numpy_bad
        if bad.any(): raise AssertionError("Проверка analytic Jacobian не пройдена: " + ", ".join(df.loc[bad, "model"].astype(str)))
    return df


def test_registry_analytic_contract():
    assert "pvoigt" in MODEL_SPECS, "Модель pvoigt отсутствует в MODEL_SPECS"
    assert MODEL_SPECS["pvoigt"].jax_func is not None, "У pvoigt не зарегистрирован jax_func"
    assert MODEL_SPECS["pvoigt"].jax_jac_func is not None, "У pvoigt не зарегистрирован analytic jax_jac_func"
    assert jaxb.JAX_SOLVER_CONFIG.get("jacobian_mode") == "auto", "Ожидался jacobian_mode='auto' по умолчанию"
    for name, spec in MODEL_SPECS.items():
        if getattr(spec, "jax_jac_func", None) is not None:
            assert spec.jax_func is not None, f"{name}: есть jax_jac_func, но отсутствует jax_func"


def test_registered_analytic_jacobians_reference_grid():
    theta = np.arange(-50.0, 52.0, 2.0)
    report = benchmark_registered_analytic_jacobians(theta, n=128, repeats=1, require_pass=True)
    assert report["finite"].all()
    assert report["allclose"].all()
    numpy_rows = report["numpy_jac_allclose"].dropna()
    assert numpy_rows.astype(bool).all()


if __name__ == "__main__":
    report = benchmark_registered_analytic_jacobians(np.arange(-50.0, 52.0, 2.0), n=2048, repeats=5)
    print(report.to_string(index=False))
