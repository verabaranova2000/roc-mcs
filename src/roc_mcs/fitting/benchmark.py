import numpy as np
import pandas as pd
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from time import perf_counter


from roc_mcs.fitting.backends.jax_backend import JAX_SOLVER_CONFIG
from roc_mcs.fitting.preparation import iter_fit_batches, pad_batch_arrays



MICROBATCH_BY_K = {1: 128, 2: 4, 3: 4}
CONV_REASON_NAMES = {1: "gradient", 2: "step", 3: "cost", 4: "max_iter"}

def get_microbatch_size(K):
    K = int(K)
    if K not in MICROBATCH_BY_K:
        raise ValueError(f"Неподдерживаемый K={K}")
    return MICROBATCH_BY_K[K]

def _sync_jax_tree(tree):
    """Дожидается фактического завершения JAX-вычисления."""
    return jax.tree_util.tree_map(
        lambda x: x.block_until_ready() if hasattr(x, "block_until_ready") else x,
        tree,
    )


def benchmark_jax_full(
    data_yxt,
    theta,
    B=128,
    prepared_fit=None,
    prepared_fit_time_s=0.0,
    solver_config=JAX_SOLVER_CONFIG,
    solver_factory=None,
    solver_name="JAX",
    old_time_s=1482.123,
    old_success=49333,
):
    """Полный JAX benchmark с раздельным учётом подготовки, compile и fit."""
    if solver_factory is None:
        raise ValueError("solver_factory must be provided")
    if prepared_fit is None:
        raise ValueError("benchmark_jax_full requires prepared_fit")

    expected_keys = {(int(x), int(y)) for group in prepared_fit.groups.values() for x, y in zip(group.x, group.y)}
    seen = set()

    stats = {
        K: {
            "pixels": 0,
            "success": 0,
            "time_s": 0.0,
            "batches": 0,
            "cost": 0.0,
            "n_iter": [],
            "optimality": [],
            "reason": [],
        }
        for K in (1, 2, 3)
    }

    batch_stats = {
        K: {
            "max_iter": [],
            "mean_iter": [],
            "time": [],
            "max_over_mean": [],
        }
        for K in (1, 2, 3)
    }

    microbatch_stats = {
        K: {
            "max_iter": [],
            "mean_iter": [],
            "max_over_mean": [],
            "effective_fraction": [],
            "size": [],
        }
        for K in (1, 2, 3)
    }

    solvers = {}
    compiled = set()
    compile_time = {K: 0.0 for K in (1, 2, 3)}

    batch_prep_time = 0.0
    validation_time = 0.0
    padding_time = 0.0
    input_prep_time = 0.0

    t_global0 = perf_counter()
    n_done_total = 0
    total_pixels = len(expected_keys)
    progress_step = max(B, total_pixels // 20)
    next_progress = progress_step

    theta_j = jnp.asarray(theta, dtype=jnp.float64)
    jax.block_until_ready(theta_j)

    print("=" * 120)
    print(f"FULL {solver_name.upper()} BENCHMARK")
    print("=" * 120)
    print(f"pixels                  : {total_pixels}")
    print(f"external batch size     : {B}")
    print(f"microbatch by K         : {MICROBATCH_BY_K}")
    print(f"backend                 : {jax.default_backend()}")
    print(f"devices                 : {jax.devices()}")
    print(f"old time                : {old_time_s:.3f} s")
    print(f"old px/s                : {old_success/max(old_time_s, 1e-15):.1f}")
    print(f"prepared fit time       : {prepared_fit_time_s:.3f} s")
    print(f"config                  : {solver_config}")
    print("=" * 120)

    batch_iter = iter_fit_batches(data_yxt=data_yxt, prepared=prepared_fit, max_batch_pixels=B)
    while True:
        t_batch0 = perf_counter()
        try:
            batch = next(batch_iter)
        except StopIteration:
            break
        batch_prep_time += perf_counter() - t_batch0

        K = int(batch.K)
        if K not in stats:
            raise ValueError(f"Неподдерживаемый K={K}")

        t_validate0 = perf_counter()
        keys = list(zip(batch.x.astype(int), batch.y.astype(int)))
        if len(keys) != len(set(keys)):
            raise AssertionError(f"K={K}: duplicate pixels inside batch")
        for key in keys:
            if key in seen:
                raise AssertionError(f"Pixel {key} appears in multiple batches")
            seen.add(key)
        validation_time += perf_counter() - t_validate0

        t_pad0 = perf_counter()
        obs, valid, x0, lb, ub, n_actual = pad_batch_arrays(batch, B)
        padding_time += perf_counter() - t_pad0

        if K not in solvers:
            print(
                f"\n[{perf_counter()-t_global0:8.1f}s] "
                f"START {solver_name} K={K}",
                flush=True,
            )
            solvers[K], _, _, _ = solver_factory(
                K,
                max_iter=solver_config["max_iter"],
                ftol=solver_config["ftol"],
                xtol=solver_config["xtol"],
                gtol=solver_config["gtol"],
            )

        t_input0 = perf_counter()
        obs_j = jnp.asarray(obs, dtype=jnp.float64)
        valid_j = jnp.asarray(valid, dtype=bool)
        x0_j = jnp.asarray(x0, dtype=jnp.float64)
        lb_j = jnp.asarray(lb, dtype=jnp.float64)
        ub_j = jnp.asarray(ub, dtype=jnp.float64)
        jax.block_until_ready((obs_j, valid_j, x0_j, lb_j, ub_j, theta_j))
        input_prep_time += perf_counter() - t_input0

        args = (obs_j, valid_j, x0_j, lb_j, ub_j, theta_j)

        if K not in compiled:
            t_compile0 = perf_counter()
            executable = solvers[K].lower(*args).compile()
            compile_time[K] = perf_counter() - t_compile0
            solvers[K] = executable
            compiled.add(K)

            print(
                f"[{perf_counter()-t_global0:8.1f}s] "
                f"JAX K={K}: compile={compile_time[K]:.3f}s "
                f"microbatch={get_microbatch_size(K)}",
                flush=True,
            )

        t_fit0 = perf_counter()
        out = _sync_jax_tree(solvers[K](*args))
        elapsed = perf_counter() - t_fit0

        iters = np.asarray(out["n_iter"])[:n_actual]

        bs = batch_stats[K]
        bs["max_iter"].append(int(iters.max()))
        bs["mean_iter"].append(float(iters.mean()))
        bs["time"].append(elapsed)
        if iters.mean() > 0.0:
            bs["max_over_mean"].append(float(iters.max() / iters.mean()))

        mb = get_microbatch_size(K)
        ms = microbatch_stats[K]

        for start in range(0, n_actual, mb):
            group = iters[start:min(start + mb, n_actual)]
            if group.size == 0:
                continue

            group_mean = float(group.mean())
            group_max = int(group.max())

            ms["size"].append(int(group.size))
            ms["max_iter"].append(group_max)
            ms["mean_iter"].append(group_mean)
            ms["max_over_mean"].append(group_max / group_mean if group_mean > 0.0 else 1.0)
            ms["effective_fraction"].append(group_mean / group_max if group_max > 0 else 1.0)

        success = np.asarray(out["success"])[:n_actual]
        cost = np.asarray(out["cost"])[:n_actual]
        n_iter = np.asarray(out["n_iter"])[:n_actual]
        optimality = np.asarray(out["optimality"])[:n_actual]
        reason = np.asarray(out["convergence_reason"])[:n_actual]

        s = stats[K]
        s["pixels"] += n_actual
        s["success"] += int(success.sum())
        s["time_s"] += elapsed
        s["batches"] += 1
        s["cost"] += float(cost.sum())
        s["n_iter"].extend(n_iter.tolist())
        s["optimality"].extend(optimality.tolist())
        s["reason"].extend(reason.tolist())

        n_done_total += n_actual

        if n_done_total >= next_progress or n_done_total == total_pixels:
            elapsed_global = perf_counter() - t_global0
            rate = n_done_total / max(elapsed_global, 1e-15)
            print(
                f"[{elapsed_global:8.1f}s] "
                f"{n_done_total:>6}/{total_pixels} "
                f"({100*n_done_total/total_pixels:5.1f}%) "
                f"| K={K} | {rate:7.1f} px/s",
                flush=True,
            )
            next_progress += progress_step

    if seen != expected_keys:
        raise AssertionError(f"Coverage mismatch: missing={len(expected_keys-seen)}, extra={len(seen-expected_keys)}")

    total_fit = sum(s["time_s"] for s in stats.values())
    total_success = sum(s["success"] for s in stats.values())
    total_cost = sum(s["cost"] for s in stats.values())
    total_compile = sum(compile_time.values())

    total_elapsed = perf_counter() - t_global0
    steady_wall = max(total_elapsed - total_compile, 0.0)
    pipeline_time = prepared_fit_time_s + total_elapsed

    speedup_fit = old_time_s / max(total_fit, 1e-15)
    speedup_wall = old_time_s / max(total_elapsed, 1e-15)
    speedup_pipeline = old_time_s / max(pipeline_time, 1e-15)

    fit_px_s = total_pixels / max(total_fit, 1e-15)
    fit_success_px_s = total_success / max(total_fit, 1e-15)
    wall_px_s = total_pixels / max(total_elapsed, 1e-15)
    wall_success_px_s = total_success / max(total_elapsed, 1e-15)

    other_time = (
        total_elapsed
        - batch_prep_time
        - validation_time
        - padding_time
        - input_prep_time
        - total_compile
        - total_fit
    )

    print("\n" + "=" * 120)
    print("JAX PER-K RESULTS")
    print("=" * 120)

    rows = []

    for K in (1, 2, 3):
        s = stats[K]
        reasons = np.asarray(s["reason"], dtype=int)
        iters = np.asarray(s["n_iter"], dtype=float)
        opts = np.asarray(s["optimality"], dtype=float)

        reason_counts = {
            CONV_REASON_NAMES.get(code, f"unknown_{code}"): int(np.sum(reasons == code))
            for code in (1, 2, 3, 4)
        }
        px_s = s["pixels"] / max(s["time_s"], 1e-15)

        rows.append({
            "K": K,
            "pixels": s["pixels"],
            "batches": s["batches"],
            "microbatch_size": get_microbatch_size(K),
            "success": s["success"],
            "success_rate": s["success"] / max(1, s["pixels"]),
            "time_s": s["time_s"],
            "px_s": px_s,
            "n_iter_median": np.median(iters) if len(iters) else np.nan,
            "n_iter_p95": np.percentile(iters, 95) if len(iters) else np.nan,
            "n_iter_max": np.max(iters) if len(iters) else np.nan,
            "optimality_p95": np.percentile(opts, 95) if len(opts) else np.nan,
            "optimality_max": np.max(opts) if len(opts) else np.nan,
            "cost": s["cost"],
            "gradient": reason_counts["gradient"],
            "step": reason_counts["step"],
            "cost_stop": reason_counts["cost"],
            "max_iter": reason_counts["max_iter"],
            "compile_time_s": compile_time[K],
        })

        print(
            f"K={K}: pixels={s['pixels']:>6} | "
            f"time={s['time_s']:>9.3f}s | "
            f"px/s={px_s:>8.1f} | "
            f"mb={get_microbatch_size(K):>3} | "
            f"success={s['success']}/{s['pixels']} | "
            f"iter med/p95/max={rows[-1]['n_iter_median']:.1f}/"
            f"{rows[-1]['n_iter_p95']:.1f}/"
            f"{rows[-1]['n_iter_max']:.0f} | "
            f"reason G/S/C/M={reason_counts['gradient']}/"
            f"{reason_counts['step']}/"
            f"{reason_counts['cost']}/"
            f"{reason_counts['max_iter']}",
            flush=True,
        )

    df = pd.DataFrame(rows)

    print("\n" + "=" * 120)
    print("JAX FULL RESULT")
    print("=" * 120)
    print(f"compile K=1/2/3    : {[round(compile_time[K], 3) for K in (1, 2, 3)]}")
    print(f"compile total      : {total_compile:.3f} s")
    print(f"prepared fit time  : {prepared_fit_time_s:.3f} s")
    print(f"batch prep time    : {batch_prep_time:.3f} s")
    print(f"validation time    : {validation_time:.3f} s")
    print(f"padding time       : {padding_time:.3f} s")
    print(f"input prep time    : {input_prep_time:.3f} s")
    print(f"fit time           : {total_fit:.3f} s")
    print(f"wall time (cold)   : {total_elapsed:.3f} s")
    print(f"wall time (steady) : {steady_wall:.3f} s")
    print(f"pipeline time      : {pipeline_time:.3f} s")
    print(f"other overhead     : {other_time:.3f} s")

    print("\nTHROUGHPUT")
    print("=" * 120)
    print(f"fit px/s           : {fit_px_s:.1f}")
    print(f"fit success px/s   : {fit_success_px_s:.1f}")
    print(f"cold wall px/s     : {wall_px_s:.1f}")
    print(f"cold wall succ px/s: {wall_success_px_s:.1f}")

    print("\nSPEEDUP")
    print("=" * 120)
    print(f"fit speedup        : {speedup_fit:.2f}x")
    print(f"cold wall speedup  : {speedup_wall:.2f}x")
    print(f"pipeline speedup   : {speedup_pipeline:.2f}x")

    print("\nOUTER BATCH ITERATION STATISTICS")
    print("=" * 120)

    for K in (1, 2, 3):
        bs = batch_stats[K]
        if not bs["max_iter"]:
            print(f"\nK={K}: no batches")
            continue

        print(f"\nK={K} | external B={B}")
        print(f"  mean batch max_iter : {np.mean(bs['max_iter']):.2f}")
        print(f"  p50  batch max_iter : {np.percentile(bs['max_iter'], 50):.2f}")
        print(f"  p95  batch max_iter : {np.percentile(bs['max_iter'], 95):.2f}")
        print(f"  max  batch max_iter : {np.max(bs['max_iter']):.0f}")
        print(f"  mean batch mean_iter: {np.mean(bs['mean_iter']):.2f}")
        print(f"  mean batch time     : {np.mean(bs['time']):.4f} s")
        print(f"  mean max/mean iter  : {np.mean(bs['max_over_mean']):.2f}x")
        print(f"  p95  max/mean iter  : {np.percentile(bs['max_over_mean'], 95):.2f}x")

    print("\nMICROBATCH ITERATION STATISTICS")
    print("=" * 120)

    for K in (1, 2, 3):
        ms = microbatch_stats[K]
        if not ms["max_iter"]:
            print(f"\nK={K}: no microbatches")
            continue

        print(f"\nK={K} | microbatch={get_microbatch_size(K)}")
        print(f"  n microbatches           : {len(ms['max_iter'])}")
        print(f"  mean micro max_iter      : {np.mean(ms['max_iter']):.2f}")
        print(f"  p50  micro max_iter      : {np.percentile(ms['max_iter'], 50):.2f}")
        print(f"  p95  micro max_iter      : {np.percentile(ms['max_iter'], 95):.2f}")
        print(f"  max  micro max_iter      : {np.max(ms['max_iter']):.0f}")
        print(f"  mean micro mean_iter     : {np.mean(ms['mean_iter']):.2f}")
        print(f"  mean micro max/mean      : {np.mean(ms['max_over_mean']):.2f}x")
        print(f"  p95  micro max/mean      : {np.percentile(ms['max_over_mean'], 95):.2f}x")
        print(f"  mean effective fraction  : {np.mean(ms['effective_fraction']):.3f}")

    print("\nFINAL")
    print("=" * 120)
    print(f"success              : {total_success}/{total_pixels} ({total_success/max(1,total_pixels):.2%})")
    print(f"total cost           : {total_cost:.6f}")
    print(f"all pixels covered   : {len(seen) == total_pixels}")
    print("\n" + df.to_string(index=False))

    return {
        "per_k": df,
        "stats": stats,
        "batch_stats": batch_stats,
        "microbatch_stats": microbatch_stats,
        "microbatch_size_by_K": dict(MICROBATCH_BY_K),
        "compile_time_s": compile_time,
        "compile_total_time_s": total_compile,
        "prepared_fit_time_s": prepared_fit_time_s,
        "batch_prep_time_s": batch_prep_time,
        "validation_time_s": validation_time,
        "padding_time_s": padding_time,
        "input_prep_time_s": input_prep_time,
        "fit_time_s": total_fit,
        "wall_time_s": total_elapsed,
        "steady_wall_time_s": steady_wall,
        "pipeline_time_s": pipeline_time,
        "other_overhead_s": other_time,
        "success": total_success,
        "success_rate": total_success / max(1, total_pixels),
        "fit_px_s": fit_px_s,
        "fit_success_px_s": fit_success_px_s,
        "wall_px_s": wall_px_s,
        "wall_success_px_s": wall_success_px_s,
        "speedup_fit_vs_old": speedup_fit,
        "speedup_wall_vs_old": speedup_wall,
        "speedup_pipeline_vs_old": speedup_pipeline,
        "total_cost": total_cost,
        "total_pixels": total_pixels,
    }