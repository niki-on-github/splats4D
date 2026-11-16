#!/usr/bin/env python3
"""Synthetic check of tools/fdanyone_to_frames.py — no 4DAnyone install or GPU needed.

Builds a random `4dgs-render/1` model, bakes it, and compares every frame
against references computed a different way:
  * conditional covariance / velocity from the 4D precision matrix
    (inverse of the joint covariance) instead of the Schur complement;
  * the covariance the viewer reconstructs from the written scale + u8
    quaternion (same formula as viewer/src/worker.ts);
  * a scalar per-gaussian evaluation of position and alpha.

Usage: uv run --with numpy python tools/test_fdanyone_to_frames.py
"""
import json
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import fdanyone_to_frames as bake  # noqa: E402

N, T, FPS = 4000, 24, 24
SPACE_SCALE, THRESHOLD = 0.1, 0.05


def synthetic_model(path, rng):
    times = np.arange(T) / FPS
    arrays = dict(
        xyz=rng.normal(0, 8, (N, 3)),
        time=rng.uniform(-0.2, times[-1] + 0.2, (N, 1)),
        scale=rng.uniform(np.log(0.02), np.log(0.6), (N, 3)),
        time_scale=rng.uniform(np.log(0.02), np.log(0.6), (N, 1)),
        left=rng.normal(0, 1, (N, 4)),
        right=rng.normal(0, 1, (N, 4)),
        opacity=rng.normal(1, 2, (N, 1)),
        dc=rng.normal(0, 1.5, (N, 1, 3)),
        sh=rng.normal(0, 0.1, (N, 15, 3)),
    )
    w2c = np.eye(4)
    w2c[:3, 3] = [0.1, -0.2, 3.0]
    metadata = dict(
        format="4dgs-render/1", step=1, degree=3, times=times.tolist(), fps=str(FPS),
        cameras=[dict(name="00", K=[[900, 0, 270], [0, 900, 480], [0, 0, 1]], w2c=w2c.tolist(), height=960, width=540)],
        config=dict(image_scale=1.0, space_scale=SPACE_SCALE, near=0.2, far=20.0, marginal_threshold=THRESHOLD),
    )
    encoded = np.frombuffer(json.dumps(metadata).encode(), dtype=np.uint8)
    np.savez_compressed(path, metadata=encoded, **{k: v.astype(np.float32) for k, v in arrays.items()})
    # reference math uses the float32 values that were actually stored
    return times, {k: v.astype(np.float32).astype(np.float64) for k, v in arrays.items()}


def reference(arrays):
    """Conditional 3D gaussian from the precision matrix: cov = inv(P_xx), velocity = -inv(P_xx) P_xt."""
    scale4 = np.exp(np.concatenate((arrays["scale"], arrays["time_scale"]), -1))
    rot = bake.rotation4(arrays["left"], arrays["right"])
    assert np.allclose(rot @ rot.transpose(0, 2, 1), np.eye(4), atol=1e-9), "rotation4 is not orthogonal"
    joint = (rot * scale4[:, None, :] ** 2) @ rot.transpose(0, 2, 1)
    precision = np.linalg.inv(joint)
    spatial = np.linalg.inv(precision[:, :3, :3])
    velocity = -(spatial @ precision[:, :3, 3:])[:, :, 0]
    return spatial, velocity, joint[:, 3, 3]


def viewer_covariance(scale, quat_u8):
    """Sigma as viewer/src/worker.ts builds it from a .splat record (without its 4x factor)."""
    r0, r1, r2, r3 = ((quat_u8.astype(np.float64) - 128) / 128).T
    sx, sy, sz = scale.astype(np.float64).T
    m = np.stack(
        (
            (1 - 2 * (r2 * r2 + r3 * r3)) * sx, 2 * (r1 * r2 + r0 * r3) * sx, 2 * (r1 * r3 - r0 * r2) * sx,
            2 * (r1 * r2 - r0 * r3) * sy, (1 - 2 * (r1 * r1 + r3 * r3)) * sy, 2 * (r2 * r3 + r0 * r1) * sy,
            2 * (r1 * r3 + r0 * r2) * sz, 2 * (r2 * r3 - r0 * r1) * sz, (1 - 2 * (r1 * r1 + r2 * r2)) * sz,
        ),
        -1,
    ).reshape(-1, 3, 3)
    return m.transpose(0, 2, 1) @ m


def main():
    rng = np.random.default_rng(7)
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        times, arrays = synthetic_model(tmp / "latest.npz", rng)
        manifest = bake.convert(tmp / "latest.npz", tmp / "frames")
        frames = [np.frombuffer((tmp / "frames" / f["file"]).read_bytes(), dtype=np.uint8).reshape(-1, 32)
                  for f in manifest["frames"]]

    spatial, velocity, variance = reference(arrays)
    base = 1 / (1 + np.exp(-arrays["opacity"][:, 0]))
    mean_t = arrays["time"][:, 0]

    # expected per-frame alpha, straight from the model definition
    dt = times[:, None] - mean_t[None, :]
    marginal = np.exp(-0.5 * dt**2 / variance)
    alpha = np.where(marginal > THRESHOLD, np.rint(base * marginal * 255), 0).astype(np.uint8)
    keep = alpha.max(0) >= 1
    n = int(keep.sum())
    assert 0 < n < N, "test model should have both kept and pruned gaussians"
    assert manifest["num_splats"] == n and manifest["count"] == T and manifest["fps"] == FPS
    assert all(f.shape == (n, 32) for f in frames), "constant N across frames"

    # static attributes: identical bytes in every frame
    for f in frames[1:]:
        assert np.array_equal(f[:, 12:27], frames[0][:, 12:27]) and np.array_equal(f[:, 28:], frames[0][:, 28:])

    # shape: full-precision decomposition, then what the viewer sees through the u8 quaternion
    metric = spatial[keep] * SPACE_SCALE**2
    scale = frames[0][:, 12:24].copy().view(np.float32)
    quat_u8 = frames[0][:, 28:32]
    model = bake.Baked({k: v[keep] for k, v in arrays.items()}, SPACE_SCALE, THRESHOLD)
    rot = viewer_covariance(np.ones((n, 3)), np.rint(model.quat * 128 + 128))  # sanity: unit scales -> identity
    assert np.allclose(rot, np.eye(3), atol=0.05)
    exact = viewer_covariance(model.scale, model.quat * 128 + 128)  # unquantised quaternion
    norm = np.linalg.norm(metric, axis=(1, 2))
    err_exact = np.linalg.norm(exact - metric, axis=(1, 2)) / norm
    assert err_exact.max() < 1e-6, f"decomposition error {err_exact.max():.2e}"
    err_u8 = np.linalg.norm(viewer_covariance(scale, quat_u8) - metric, axis=(1, 2)) / norm
    assert err_u8.max() < 0.08, f"u8 quaternion covariance error {err_u8.max():.3f}"

    # colour
    rgb = np.rint(np.clip(arrays["dc"][keep, 0] * bake.C0 + 0.5, 0, 1) * 255).astype(np.uint8)
    assert np.array_equal(frames[0][:, 24:27], rgb)

    # per-frame position and alpha
    xyz, vel = arrays["xyz"][keep] * SPACE_SCALE, velocity[keep] * SPACE_SCALE
    worst = 0.0
    for i, f in enumerate(frames):
        a = alpha[i, keep]
        assert np.array_equal(f[:, 27], a), f"alpha mismatch in frame {i}"
        pos = f[:, 0:12].copy().view(np.float32).astype(np.float64)
        live = a > 0
        expect = xyz + vel * dt[i, keep][:, None]
        worst = max(worst, np.abs(pos - expect)[live].max(initial=0.0))
    assert worst < 1e-5, f"position error {worst:.2e} m"

    # inactive gaussians do not move: position equals the window edge, so it is constant per side
    pos_all = np.stack([f[:, 0:12].copy().view(np.float32) for f in frames])
    alive = alpha[:, keep] > 0
    marg = marginal[:, keep] > THRESHOLD
    for side in (dt[:, keep] < 0, dt[:, keep] > 0):
        idle = ~marg & side
        hi = np.where(idle[..., None], pos_all, -np.inf).max(0)
        lo = np.where(idle[..., None], pos_all, np.inf).min(0)
        assert not (idle.any(0)[:, None] & (hi != lo)).any(), "inactive gaussian moved"

    hint = manifest["camera"]
    assert np.allclose(hint["position"], [-0.1, 0.2, -3.0]) and np.allclose(hint["up"], [0, -1, 0])

    print(f"OK  kept {n}/{N}  alive/frame {alive.sum(1).mean():.0f}  max pos err {worst:.1e} m"
          f"  cov err exact {err_exact.max():.1e} / u8 quat {err_u8.max():.3f}")


if __name__ == "__main__":
    main()
