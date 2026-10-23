# Copyright(C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
"""Local W2 approximation using weighted centroids and exact or Sinkhorn transport.

The mass-corrected transport plan is a primal upper bound; c-transforms give a
numerical lower bound on unregularized OT for the compressed distributions.
These bounds do not establish reproduction of Meta's unpublished settings.
"""

import math
import time
import numpy as np


def weighted_voxels(points, pitch):
    """Represent every input point by its cell's centroid, preserving cell mass."""
    points = np.asarray(points, dtype=np.float64)
    if (
        points.ndim != 2
        or points.shape[1] != 3
        or not len(points)
        or not np.isfinite(points).all()
        or not math.isfinite(pitch)
        or pitch <= 0
    ):
        raise ValueError(
            "Expected finite nonempty [N,3] points and positive grid pitch"
        )
    q = np.floor(points / pitch).astype(np.int64)
    q -= q.min(0)
    shape = q.max(0) + 1
    keys = (q[:, 0] * shape[1] + q[:, 1]) * shape[2] + q[:, 2]
    _, inverse, count = np.unique(keys, return_inverse=True, return_counts=True)
    centers = np.stack(
        [np.bincount(inverse, weights=points[:, k]) / count for k in range(3)], 1
    )
    rms = float(np.sqrt(np.mean(np.sum((points - centers[inverse]) ** 2, axis=1))))
    return centers, count.astype(np.float64) / len(points), rms


def sinkhorn_w2(
    x, a, y, b, *, blur=0.01, scaling=0.95, device="cpu", max_iterations=10000
):
    """Use GeomLoss potentials to precondition POT's float64 Sinkhorn solve.

    Adding row/column constants to the cost does not change the optimal balanced
    coupling. Return its unregularized RMS transport cost, not a regularized loss
    or a debiased Sinkhorn divergence. A c-transform supplies a feasible LP dual.
    """
    import torch
    import ot
    from geomloss import SamplesLoss

    if (
        not math.isfinite(blur)
        or blur <= 0
        or not 0 < scaling < 1
        or max_iterations < 1
    ):
        raise ValueError("Positive blur/iteration count and 0 < scaling < 1 required")
    started = time.monotonic()
    epsilon = blur**2
    with torch.no_grad():
        xx, aa, yy, bb = [
            torch.as_tensor(v, dtype=torch.float64, device=device) for v in (x, a, y, b)
        ]
        aa = aa / aa.sum()
        bb = bb / bb.sum()
        if not (
            torch.isfinite(xx).all()
            and torch.isfinite(yy).all()
            and torch.all(aa > 0)
            and torch.all(bb > 0)
        ):
            raise ValueError("Finite coordinates and strictly positive masses required")
        # Warm start only: float64 POT subsequently checks both mass constraints.
        f, g = SamplesLoss(
            "sinkhorn",
            p=2,
            blur=max(blur, 0.08),
            scaling=scaling,
            debias=False,
            potentials=True,
            backend="tensorized",
        )(aa.float(), xx.float(), bb.float(), yy.float())
        f = f.reshape(-1).double()
        g = g.reshape(-1).double()
        cost = torch.clamp(
            ((xx * xx).sum(1)[:, None] + (yy * yy).sum(1)[None, :]) / 2 - xx @ yy.T,
            min=0,
        )
        levels = [0.08, 0.04, 0.02]
        levels = [level for level in levels if level > blur] + [blur]
        stage_reports = []
        for level in levels:
            epsilon_level = level**2
            shifted = (
                cost
                - f[:, None]
                - g[None, :]
                - epsilon_level * (aa.log()[:, None] + bb.log()[None, :])
            )
            plan, log = ot.sinkhorn(
                aa,
                bb,
                shifted,
                epsilon_level,
                method="sinkhorn",
                numItermax=max_iterations
                if level == blur
                else min(300, max_iterations),
                stopThr=1e-10 if level == blur else 1e-7,
                log=True,
                warn=False,
            )
            del shifted
            if not (
                torch.isfinite(plan).all()
                and torch.all(log["u"] > 0)
                and torch.all(log["v"] > 0)
            ):
                raise ArithmeticError("Invalid Sinkhorn scaling")
            f = f + epsilon_level * log["u"].log()
            g = g + epsilon_level * log["v"].log()
            stage_reports.append(
                dict(
                    blur=level,
                    iterations=int(log["niter"]),
                    marginal_l1=max(
                        float(torch.abs(plan.sum(1) - aa).sum()),
                        float(torch.abs(plan.sum(0) - bb).sum()),
                    ),
                )
            )
            if level != blur:
                del plan
        if not torch.isfinite(plan).all():
            raise ArithmeticError("Nonfinite Sinkhorn coupling")
        row_error = float(torch.abs(plan.sum(1) - aa).sum())
        column_error = float(torch.abs(plan.sum(0) - bb).sum())
        # Round the near-feasible coupling to the prescribed marginals, preserving
        # nonnegativity; residual mass is at most the recorded solver tolerance.
        plan *= torch.minimum(aa / plan.sum(1), torch.ones_like(aa))[:, None]
        plan *= torch.minimum(bb / plan.sum(0), torch.ones_like(bb))[None, :]
        da = torch.clamp(aa - plan.sum(1), min=0)
        db = torch.clamp(bb - plan.sum(0), min=0)
        residual_mass = float(da.sum())
        if residual_mass > 0:
            plan += da[:, None] * (db / da.sum())[None, :]
        rounded_error = max(
            float(torch.abs(plan.sum(1) - aa).sum()),
            float(torch.abs(plan.sum(0) - bb).sum()),
        )
        if rounded_error > 1e-10:
            raise ArithmeticError(f"Coupling rounding failed: {rounded_error}")
        upper = float(torch.sum(plan * cost))
        # Any f followed by these c-transforms gives f_i+g_j <= cost_ij.
        dual_f = f + epsilon * aa.log()
        dual_g = torch.min(cost - dual_f[:, None], dim=0).values
        dual_f = torch.min(cost - dual_g[None, :], dim=1).values
        lower = float(torch.sum(aa * dual_f) + torch.sum(bb * dual_g))
        if not (math.isfinite(lower) and math.isfinite(upper)):
            raise ArithmeticError("Nonfinite transport bounds")
        if lower > upper + 1e-10:
            raise ArithmeticError(
                f"Invalid transport bounds: dual {lower} > primal {upper}"
            )
        lower = min(lower, upper)
        result = dict(
            w2=math.sqrt(max(2 * upper, 0)),
            w2_lower=math.sqrt(max(2 * lower, 0)),
            transport_half_squared=upper,
            dual_half_squared=lower,
            stages=stage_reports,
            sinkhorn_marginals_converged=max(row_error, column_error) <= 1e-6,
            row_l1_error=row_error,
            column_l1_error=column_error,
            rounded_l1_error=rounded_error,
            rounding_mass=residual_mass,
            iterations=int(log["niter"]),
            blur=blur,
            epsilon=epsilon,
            scaling=scaling,
            seconds=time.monotonic() - started,
        )
        return result


def earth_mover(
    a,
    b,
    *,
    pitch=0.04,
    max_centroids=24576,
    blur=0.01,
    scaling=0.95,
    device="cpu",
    max_iterations=2000,
    solver="exact",
    lp_iterations=10000000,
):
    """Compress full surface samples, preserve their mass, and record uncertainty."""
    x, aw, a_rms = weighted_voxels(a, pitch)
    y, bw, b_rms = weighted_voxels(b, pitch)
    if max(len(x), len(y)) > max_centroids:
        raise ValueError(
            "EMD centroid limit exceeded: increase --emd-pitch or --emd-max-centroids"
        )
    if solver == "sinkhorn":
        result = sinkhorn_w2(
            x,
            aw,
            y,
            bw,
            blur=blur,
            scaling=scaling,
            device=device,
            max_iterations=max_iterations,
        )
    elif solver == "exact":
        result = exact_w2(
            x,
            aw,
            y,
            bw,
            max_iterations=lp_iterations,
            mass_scale=len(a) if len(a) == len(b) else None,
        )
    else:
        raise ValueError("EMD solver must be sinkhorn or exact")
    result.update(
        samples_predicted=len(a),
        samples_reference=len(b),
        centroids_predicted=len(x),
        centroids_reference=len(y),
        grid_pitch=pitch,
        quantization_rms_predicted=a_rms,
        quantization_rms_reference=b_rms,
        # Triangle inequality supplies a lower bound for the original empirical
        # distributions. Lifting each centroid coupling independently within its
        # cells adds exactly their within-cell variances to squared W2 cost.
        full_sample_w2_lower=max(0.0, result["w2_lower"] - a_rms - b_rms),
        full_sample_w2_upper=math.sqrt(result["w2"] ** 2 + a_rms**2 + b_rms**2),
    )
    return result


def self_check():
    """Check against exact CPU network-simplex OT and analytical controls."""
    import itertools
    import ot
    from scipy.spatial.distance import cdist

    rng = np.random.default_rng(42)
    identity = rng.normal(size=(128, 3))
    translated = rng.normal(size=(128, 3))
    shift = np.array([0.1, -0.2, 0.15])
    cases = [
        ("identity", identity, None, identity.copy(), None),
        ("translation", translated, None, translated + shift, None),
        (
            "different_shapes",
            rng.normal(size=(256, 3)) * 0.3,
            None,
            rng.normal(size=(256, 3)) * 0.2 + shift,
            None,
        ),
        (
            "same_support_different_mass",
            np.array([[-1.0, 0, 0], [1.0, 0, 0]]),
            np.array([0.75, 0.25]),
            np.array([[-1.0, 0, 0], [1.0, 0, 0]]),
            np.array([0.25, 0.75]),
        ),
    ]
    checks = []
    for name, x, a, y, b in cases:
        a = np.full(len(x), 1 / len(x)) if a is None else a
        b = np.full(len(y), 1 / len(y)) if b is None else b
        exact = float(np.sqrt(ot.emd2(a, b, cdist(x, y, "sqeuclidean"))))
        if name == "translation":
            np.testing.assert_allclose(exact, np.linalg.norm(shift), rtol=0, atol=1e-12)
        np.testing.assert_allclose(
            exact_w2(x, a, y, b)["w2"], exact, rtol=0, atol=1e-12
        )
        if name == "same_support_different_mass":
            np.testing.assert_allclose(exact, math.sqrt(2), rtol=0, atol=1e-12)
        for blur in [0.02, 0.01, 0.005]:
            result = sinkhorn_w2(x, a, y, b, blur=blur)
            assert result["w2_lower"] <= exact + 1e-7 and exact <= result["w2"] + 1e-7
            assert abs(result["w2"] - exact) <= 1e-3, (name, exact, result)
            checks.append(dict(case=name, exact_w2=exact, **result))
    x = rng.normal(size=(3, 3))
    y = rng.normal(size=(3, 3))
    costs = cdist(x, y, "sqeuclidean")
    enumerated = min(
        np.mean(costs[np.arange(3), permutation])
        for permutation in itertools.permutations(range(3))
    )
    np.testing.assert_allclose(
        ot.emd2(np.full(3, 1 / 3), np.full(3, 1 / 3), costs),
        enumerated,
        rtol=0,
        atol=1e-12,
    )
    x = rng.normal(size=(256, 3)) * 0.3
    y = x * np.array([1.0, 0.8, 1.2]) + shift
    result = earth_mover(x, y, pitch=0.2, blur=0.01)
    exact = float(
        np.sqrt(
            ot.emd2(
                np.full(256, 1 / 256), np.full(256, 1 / 256), cdist(x, y, "sqeuclidean")
            )
        )
    )
    assert result["full_sample_w2_lower"] <= exact + 1e-7
    assert exact <= result["full_sample_w2_upper"] + 1e-7
    centers, weights, rms = weighted_voxels(x, 0.2)
    np.testing.assert_allclose(weights.sum(), 1.0, rtol=0, atol=1e-12)
    np.testing.assert_allclose(weights @ centers, x.mean(0), rtol=0, atol=1e-12)
    assert 0 <= rms <= math.sqrt(3) * 0.2
    return dict(
        exact_controls=checks,
        brute_force_three_point_cost=float(enumerated),
        compressed_distribution=dict(exact_w2=exact, **result),
    )


def exact_w2(x, a, y, b, *, max_iterations=10000000, mass_scale=None):
    """CPU network-simplex reference for a bounded weighted point problem."""
    import ot
    from scipy.spatial.distance import cdist

    started = time.monotonic()
    x, a, y, b = (np.asarray(value, dtype=np.float64) for value in (x, a, y, b))
    a = a / a.sum()
    b = b / b.sum()
    if not (
        np.isfinite(x).all()
        and np.isfinite(y).all()
        and np.all(a > 0)
        and np.all(b > 0)
    ):
        raise ValueError("Finite coordinates and positive masses required")
    factor = 1.0
    native_a, native_b = a, b
    if mass_scale is not None:
        if int(mass_scale) != mass_scale or not 1 <= mass_scale < 2**52:
            raise ValueError("Integer empirical sample count required")
        factor = float(mass_scale)
        native_a, native_b = np.rint(a * factor), np.rint(b * factor)
        if not (
            native_a.sum() == native_b.sum() == factor
            and np.all(native_a > 0)
            and np.all(native_b > 0)
            and np.allclose(native_a / factor, a, rtol=0, atol=1e-12)
            and np.allclose(native_b / factor, b, rtol=0, atol=1e-12)
        ):
            raise ValueError(
                "Masses do not represent the stated empirical sample counts"
            )
        a, b = native_a / factor, native_b / factor
    # Relabeling support points leaves OT unchanged and avoids pathological
    # lexicographic pivot order. Integer sample counts eliminate tiny residual
    # flows when the weights came from a uniformly sampled empirical measure.
    rng = np.random.default_rng(42)
    ix, iy = rng.permutation(len(x)), rng.permutation(len(y))
    x, a, native_a = x[ix], a[ix], native_a[ix]
    y, b, native_b = y[iy], b[iy], native_b[iy]
    cost = cdist(x, y, "sqeuclidean")
    value, log = ot.emd2(native_a, native_b, cost, numItermax=max_iterations, log=True)
    value = float(value) / factor
    if int(log["result_code"]) != 1 or log["warning"] is not None:
        raise ArithmeticError(
            "Exact OT solver did not certify an optimum: " + str(log["warning"])
        )
    u, v = log["u"], log["v"]
    violation = max(
        float(np.max(u[i : i + 256, None] + v[None, :] - cost[i : i + 256]))
        for i in range(0, len(x), 256)
    )
    dual = float(np.dot(a, u) + np.dot(b, v) - max(violation, 0.0))
    if not math.isfinite(value) or abs(float(value) - dual) > 1e-7:
        raise ArithmeticError("Exact OT primal/dual mismatch")
    return dict(
        w2=math.sqrt(max(float(value), 0)),
        w2_lower=math.sqrt(max(min(dual, float(value)), 0)),
        transport_squared=float(value),
        dual_squared=dual,
        dual_max_violation=violation,
        result_code=int(log["result_code"]),
        solver="POT network simplex, float64",
        native_mass_scale=factor,
        support_permutation_seed=42,
        seconds=time.monotonic() - started,
    )
