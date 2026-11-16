#!/usr/bin/env python3
"""Bake a 4DAnyone 4DGS render model (checkpoints/latest.npz) into a series of
antimatter15 .splat frames that `splat4d encode` accepts.

4DAnyone (https://github.com/ant-research/4DAnyone, Apache-2.0) trains native
4D gaussians. Its `4dgs-render/1` .npz holds, per gaussian:
  xyz (3)  time (1)          4D mean, in training units (metres / space_scale)
  scale (3)  time_scale (1)  log of the four 4D scales
  left (4)  right (4)        two quaternions: 4D rotation R = L(left) Q(right)
  opacity (1)                logit
  dc (1,3)  sh (K,3)         spherical harmonics; only dc is kept here
plus a `metadata` JSON blob (fps, times, cameras, config).

Evaluation at time t (ported from fdanyone/reconstruction/gaussian.py,
covariance4 + evaluate_conditional), with S = F F^T, F = R diag(exp(scales)),
c = S[0:3,3], var = S[3,3]:
  velocity = c / var
  spatial  = S[0:3,0:3] - c c^T / var        constant in time
  dt       = t - time
  pos      = xyz + velocity * dt
  alpha    = sigmoid(opacity) * exp(-0.5 dt^2 / var), culled below
             config.marginal_threshold

So rotation, scale and colour are static per gaussian; only position (linear)
and alpha change. Every kept gaussian is written in every frame (the encoder
needs constant N and index correspondence): alpha is 0 while it is inactive and
its position is frozen at the edge of its active window, so those stretches
cost almost nothing after encoding and nothing drifts off along its velocity.

Lost in the bake: view-dependent colour (SH degree > 0), and rotation precision
(.splat stores quaternions as 4 x u8).

Usage:
  uv run --with numpy python tools/fdanyone_to_frames.py \
      <result>/4dgs/checkpoints/latest.npz data/frames/<name> \
      [--min-peak-alpha 1] [--max-splats N]
  splat4d encode -i data/frames/<name> -o <name>.splat4d
"""
import argparse
import json
import time as clock
from fractions import Fraction
from pathlib import Path

import numpy as np

C0 = 0.28209479177387814
NAMES = ("xyz", "time", "scale", "time_scale", "left", "right", "opacity", "dc", "sh")


def load_model(path):
    """-> (metadata dict, {name: float64 array})."""
    with np.load(path, allow_pickle=False) as archive:
        missing = {*NAMES, "metadata"} - set(archive.files)
        if missing:
            raise ValueError(f"{path}: not a 4DAnyone render model, missing {sorted(missing)}")
        metadata = json.loads(archive["metadata"].tobytes().decode("utf-8"))
        if metadata.get("format") != "4dgs-render/1":
            raise ValueError(f"{path}: unsupported format {metadata.get('format')!r}")
        arrays = {name: archive[name].astype(np.float64) for name in NAMES if name != "sh"}
    n = arrays["xyz"].shape[0]
    shapes = dict(xyz=(3,), time=(1,), scale=(3,), time_scale=(1,), left=(4,), right=(4,), opacity=(1,), dc=(1, 3))
    for name, shape in shapes.items():
        if arrays[name].shape != (n, *shape):
            raise ValueError(f"{path}: {name} has shape {arrays[name].shape}, expected {(n, *shape)}")
    return metadata, arrays


def rotation4(left, right):
    """4D rotation from two quaternions (left- and right-isoclinic), (N,4,4)."""
    left = left / np.linalg.norm(left, axis=-1, keepdims=True)
    right = right / np.linalg.norm(right, axis=-1, keepdims=True)
    a, b, c, d = left.T
    p, q, r, s = right.T
    lmat = np.stack((a, -b, -c, -d, b, a, -d, c, c, d, a, -b, d, -c, b, a), -1).reshape(-1, 4, 4)
    rmat = np.stack((p, q, r, s, -q, p, -s, r, -r, s, p, -q, -s, -r, q, p), -1).reshape(-1, 4, 4)
    return lmat @ rmat


def covariance4(log_scale, log_time_scale, left, right):
    """-> spatial (N,3,3), velocity (N,3), temporal variance (N,1), conditioned on time."""
    scale = np.exp(np.concatenate((log_scale, log_time_scale), -1))
    factor = rotation4(left, right) * scale[:, None, :]
    full = factor @ factor.transpose(0, 2, 1)
    temporal = full[:, 3, 3:]
    cross = full[:, :3, 3]
    velocity = cross / temporal
    spatial = full[:, :3, :3] - cross[:, :, None] * velocity[:, None, :]
    return spatial, velocity, temporal


def matrix_to_quat(m):
    """Proper rotation matrices (N,3,3) -> unit quaternions (N,4) as (w,x,y,z), w >= 0."""
    m00, m01, m02 = m[:, 0, 0], m[:, 0, 1], m[:, 0, 2]
    m10, m11, m12 = m[:, 1, 0], m[:, 1, 1], m[:, 1, 2]
    m20, m21, m22 = m[:, 2, 0], m[:, 2, 1], m[:, 2, 2]
    traces = np.stack((1 + m00 + m11 + m22, 1 + m00 - m11 - m22, 1 - m00 + m11 - m22, 1 - m00 - m11 + m22), -1)
    k = traces.argmax(-1)
    # each row is 4*lead*(w,x,y,z) for the branch whose lead component is largest
    branches = np.stack(
        (
            np.stack((traces[:, 0], m21 - m12, m02 - m20, m10 - m01), -1),
            np.stack((m21 - m12, traces[:, 1], m01 + m10, m02 + m20), -1),
            np.stack((m02 - m20, m01 + m10, traces[:, 2], m12 + m21), -1),
            np.stack((m10 - m01, m02 + m20, m12 + m21, traces[:, 3]), -1),
        ),
        1,
    )
    q = branches[np.arange(len(m)), k]
    q /= np.linalg.norm(q, axis=-1, keepdims=True)
    q[q[:, 0] < 0] *= -1
    return q


def decompose(spatial, chunk=1 << 18):
    """Symmetric PSD (N,3,3) -> linear scales (N,3), quaternions (N,4): spatial = R diag(s^2) R^T."""
    n = len(spatial)
    scales = np.empty((n, 3))
    quats = np.empty((n, 4))
    for lo in range(0, n, chunk):
        w, v = np.linalg.eigh(spatial[lo : lo + chunk])
        v[np.linalg.det(v) < 0, :, 0] *= -1  # eigenvectors form a reflection as often as a rotation
        scales[lo : lo + chunk] = np.sqrt(np.maximum(w, 1e-14))
        quats[lo : lo + chunk] = matrix_to_quat(v)
    return scales, quats


class Baked:
    """Time-independent per-gaussian quantities, in metres and seconds."""

    def __init__(self, arrays, space_scale, threshold):
        spatial, velocity, variance = covariance4(arrays["scale"], arrays["time_scale"], arrays["left"], arrays["right"])
        scales, self.quat = decompose(spatial)
        self.scale = scales * space_scale
        self.xyz = arrays["xyz"] * space_scale
        self.velocity = velocity * space_scale
        self.time = arrays["time"][:, 0]
        self.variance = variance[:, 0]
        self.base_alpha = 1.0 / (1.0 + np.exp(-arrays["opacity"][:, 0]))
        self.rgb = np.clip(arrays["dc"][:, 0, :] * C0 + 0.5, 0.0, 1.0)
        self.threshold = threshold
        # marginal > threshold  <=>  |dt| < window
        self.window = np.sqrt(-2.0 * self.variance * np.log(threshold))

    def __len__(self):
        return len(self.xyz)

    def select(self, keep):
        for name in ("scale", "quat", "xyz", "velocity", "time", "variance", "base_alpha", "rgb", "window"):
            setattr(self, name, getattr(self, name)[keep])

    def alpha(self, t):
        """u8 alpha at time t; 0 wherever 4DAnyone's renderer would cull the gaussian."""
        dt = t - self.time
        marginal = np.exp(-0.5 * dt * dt / self.variance)
        value = np.rint(self.base_alpha * marginal * 255.0)
        return np.where(marginal > self.threshold, value, 0.0).astype(np.uint8)

    def position(self, t):
        """Position at time t, held at the edge of the active window outside it."""
        dt = np.clip(t - self.time, -self.window, self.window)
        return self.xyz + self.velocity * dt[:, None]


def camera_hint(camera, centre):
    """Viewer camera hint from a 4DAnyone OpenCV camera record (metric w2c)."""
    c2w = np.linalg.inv(np.asarray(camera["w2c"], dtype=np.float64))
    position, forward, up = c2w[:3, 3], c2w[:3, 2], -c2w[:3, 1]
    distance = max(float(np.dot(centre - position, forward)), 0.1)
    hint = {
        "position": [round(float(v), 4) for v in position],
        "target": [round(float(v), 4) for v in position + forward * distance],
        "up": [round(float(v), 4) + 0.0 for v in up],
    }
    k, height = camera.get("K"), camera.get("height")
    if k and height:
        hint["fov"] = round(float(np.degrees(2 * np.arctan(height / (2 * k[1][1])))), 3)
    return hint


def convert(src, out_dir, min_peak_alpha=1, max_splats=None):
    t0 = clock.time()
    metadata, arrays = load_model(src)
    config = metadata["config"]
    times = np.asarray(metadata["times"], dtype=np.float64)
    fps = float(Fraction(str(metadata["fps"])))
    model = Baked(arrays, float(config["space_scale"]), float(config["marginal_threshold"]))
    total = len(model)
    print(f"{src}: {total} gaussians, {len(times)} frames @ {fps:g} fps, SH degree {metadata['degree']} (dc only kept)")

    # pass 1: which gaussians are ever visible at a sampled frame
    peak = np.zeros(total, dtype=np.uint8)
    weight = np.zeros(total)
    for t in times:
        a = model.alpha(t)
        np.maximum(peak, a, out=peak)
        weight += a
    keep = peak >= max(min_peak_alpha, 1)
    if max_splats is not None and keep.sum() > max_splats:
        # rank by time-integrated alpha x footprint (the two largest axes)
        footprint = model.scale[:, 1] * model.scale[:, 2]
        score = np.where(keep, weight * footprint, -1.0)
        keep = np.zeros(total, dtype=bool)
        keep[np.argpartition(score, -max_splats)[-max_splats:]] = True
    model.select(keep)
    n = len(model)
    if not n:
        raise ValueError("no gaussian is visible at any frame")
    print(f"kept {n} ({100 * n / total:.1f} %), dropped {total - n} never above alpha {max(min_peak_alpha, 1)}/255"
          + (f" or beyond --max-splats {max_splats}" if max_splats is not None else ""))

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    rec = np.zeros((n, 32), dtype=np.uint8)
    rec[:, 12:24] = model.scale.astype(np.float32).view(np.uint8).reshape(n, 12)
    rec[:, 24:27] = np.rint(model.rgb * 255.0).astype(np.uint8)
    rec[:, 28:32] = np.clip(np.rint(model.quat * 128.0 + 128.0), 0, 255).astype(np.uint8)

    manifest = {"fps": fps, "count": len(times), "num_splats": int(n), "frames": []}
    alive = []
    centre = np.zeros(3)
    for i, t in enumerate(times):
        a = model.alpha(t)
        pos = model.position(t)
        rec[:, 0:12] = pos.astype(np.float32).view(np.uint8).reshape(n, 12)
        rec[:, 27] = a
        name = f"frame_{i:04d}.splat"
        (out / name).write_bytes(rec.tobytes())
        manifest["frames"].append({"file": name, "timestamp": round(float(t - times[0]), 6)})
        alive.append(int(np.count_nonzero(a)))
        if alive[-1]:
            centre += np.average(pos, axis=0, weights=a) / len(times)

    cameras = metadata.get("cameras") or []
    if cameras:
        manifest["camera"] = camera_hint(cameras[0], centre)
    (out / "frames.json").write_text(json.dumps(manifest, indent=1))

    alive = np.asarray(alive)
    print(f"wrote {len(times)} frames x {n} splats = {len(times) * n * 32 / 1e6:.1f} MB in {clock.time() - t0:.1f}s -> {out}")
    print(f"alive per frame: min {alive.min()}  mean {alive.mean():.0f}  max {alive.max()}"
          f"  -> total/alive = {n / max(alive.mean(), 1):.2f}x")
    if "camera" in manifest:
        print(f"camera hint: {manifest['camera']}")
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("model", help="4DAnyone render model (checkpoints/latest.npz)")
    parser.add_argument("out_dir", help="output directory for frame_NNNN.splat + frames.json")
    parser.add_argument("--min-peak-alpha", type=int, default=1,
                        help="drop gaussians whose alpha never reaches this (0..255) at any frame (default 1)")
    parser.add_argument("--max-splats", type=int, default=None,
                        help="keep at most this many gaussians, ranked by time-integrated alpha x footprint")
    args = parser.parse_args()
    convert(args.model, args.out_dir, args.min_peak_alpha, args.max_splats)
