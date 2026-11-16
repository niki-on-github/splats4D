// WebXR (immersive-vr / immersive-ar) front end for the WebGL2 renderer.
//
// The streaming/decode/sort worker and the shaders are untouched: per eye we
// hand the reference shader a view/projection pair in ITS convention (OpenCV:
// +z forward, y down, focal in pixels) derived from the XR matrices:
//
//   F      = diag(1,-1,-1)                      (GL eye  <-> OpenCV eye)
//   M      = T(t) · Ry(yaw) · S(s) · F          (splat world -> XR space, level)
//   view'  = F · V_xr · M
//   proj'  = P_xr · F                           (so proj'·view' = P_xr·V_xr·M)
//   focal  = (P_xr[0]·w/2, P_xr[5]·h/2)
//
// With depthNear = 0.1 the shader's near fade (clamp(z/w + 1)) covers the same
// 0.1–0.2 m range as the reference projection, so no shader change is needed.
// One depth sort (head forward direction) serves both eyes.
//
// Controls: grip or trigger/pinch held = grab & move the scene; both hands =
// move + scale + turn; short trigger tap or A/X = play/pause; B/Y = re-place
// the scene in front of you; left stick = walk (relative to where you look,
// kept level); right stick ←→ = snap turn, ↑↓ = scrub time.

import type { GlSplatRenderer, GlSplatSet, GlView } from './renderer_gl';
import { invert4, multiply4 } from './camera';

export type XrMode = 'immersive-vr' | 'immersive-ar';

export interface XrHost {
  renderer: GlSplatRenderer;
  /** sets to draw this frame (empty while loading) */
  sets(): GlSplatSet[];
  /** current desktop view matrix; used to place the scene on entry */
  desktopView(): number[];
  /** advance playback / request decoded frames; called once per XR frame */
  tick(now: number): void;
  /** send a depth-sort direction to the worker (only elements 2, 6, 10 are read) */
  postSort(viewProj: Float32Array): void;
  togglePlay(): void;
  /** move the playhead by a fraction of the sequence duration */
  scrub(fraction: number): void;
  onStats(fps: number, viewportScale: number): void;
  onEnd(): void;
}

export interface XrOptions {
  /** XRWebGLLayer framebufferScaleFactor (1 = the device's recommended size) */
  framebufferScale: number;
  /** lower bound for the adaptive per-frame viewport scale */
  minViewportScale: number;
  /** target refresh rate to request when the device offers it (0 = leave as is) */
  frameRate: number;
  /** cap on the projected splat axis (reference: 1024) */
  maxSplatAxis: number;
}

export const XR_DEFAULTS: XrOptions = {
  framebufferScale: 1,
  minViewportScale: 0.5,
  frameRate: 72,
  maxSplatAxis: 512,
};

export async function xrSupport(): Promise<{ vr: boolean; ar: boolean }> {
  const xr = navigator.xr;
  if (!xr) return { vr: false, ar: false };
  const ok = (mode: XrMode) => xr.isSessionSupported(mode).catch(() => false);
  const [vr, ar] = await Promise.all([ok('immersive-vr'), ok('immersive-ar')]);
  return { vr, ar };
}

type Vec3 = [number, number, number];

/** level placement of the splat world in XR space */
interface Placement {
  t: Vec3;
  yaw: number;
  s: number;
}

/** heading of a direction in the xz plane; Ry(a) adds a to it */
const heading = (x: number, z: number) => Math.atan2(x, z);

function rotY(v: Vec3, a: number): Vec3 {
  const c = Math.cos(a);
  const s = Math.sin(a);
  return [c * v[0] + s * v[2], v[1], -s * v[0] + c * v[2]];
}

function modelMatrix(p: Placement): number[] {
  const c = Math.cos(p.yaw) * p.s;
  const s = Math.sin(p.yaw) * p.s;
  // T · Ry · S · F, column-major
  return [c, 0, -s, 0, 0, -p.s, 0, 0, -s, 0, -c, 0, p.t[0], p.t[1], p.t[2], 1];
}

/** F · A (negate rows 1, 2) */
function flipRows(a: ArrayLike<number>): number[] {
  const o = Array.from(a);
  for (const i of [1, 2, 5, 6, 9, 10, 13, 14]) o[i] = -o[i];
  return o;
}

/** A · F (negate columns 1, 2) */
function flipCols(a: ArrayLike<number>): number[] {
  const o = Array.from(a);
  for (let i = 4; i < 12; i++) o[i] = -o[i];
  return o;
}

/** scale by k and turn by dyaw about `from`, then carry that pivot to `to` */
function pivot(p: Placement, from: Vec3, to: Vec3, k: number, dyaw: number) {
  const s = Math.min(20, Math.max(0.02, p.s * k));
  k = s / p.s;
  const r = rotY([p.t[0] - from[0], p.t[1] - from[1], p.t[2] - from[2]], dyaw);
  p.t = [to[0] + k * r[0], to[1] + k * r[1], to[2] + k * r[2]];
  p.yaw += dyaw;
  p.s = s;
}

interface SourceState {
  holds: number; // select + squeeze currently pressed
  prev: Vec3 | null;
  tapStart: number;
  tapTravel: number;
  buttons: boolean[];
  turned: boolean; // snap-turn latch: one turn per stick flick
}

const WALK_SPEED = 1.5; // m/s at full stick deflection
const SNAP_TURN = Math.PI / 6;

export async function startXr(mode: XrMode, host: XrHost, options: Partial<XrOptions> = {}): Promise<XRSession> {
  const opt = { ...XR_DEFAULTS, ...options };
  const gl = host.renderer.gl;

  // must stay first: requestSession needs the click's user activation
  const session = await navigator.xr!.requestSession(mode, {
    optionalFeatures: ['local-floor', 'hand-tracking'],
  });
  await gl.makeXRCompatible();

  const layer = new XRWebGLLayer(session, gl, {
    antialias: false,
    depth: false,
    stencil: false,
    alpha: true,
    framebufferScaleFactor: opt.framebufferScale,
  });
  // the periphery is the cheapest place to save fill rate
  if ('fixedFoveation' in layer) (layer as XRWebGLLayer & { fixedFoveation: number }).fixedFoveation = 1;
  await session.updateRenderState({ baseLayer: layer, depthNear: 0.1, depthFar: 200 });

  const refSpace = await session
    .requestReferenceSpace('local-floor')
    .catch(() => session.requestReferenceSpace('local'));

  if (opt.frameRate > 0 && session.supportedFrameRates?.includes(opt.frameRate)) {
    void session.updateTargetFrameRate?.(opt.frameRate).catch(() => {});
  }

  // identical output for alpha-0 splats (they add nothing), but no fill cost
  host.renderer.setLod(opt.maxSplatAxis, 1);

  // ---- placement -----------------------------------------------------------
  const place: Placement = { t: [0, 0, 0], yaw: 0, s: 1 };
  let placed = false;

  /** put the desktop camera's viewpoint at the viewer's head, kept level */
  function placeFromHead(head: XRRigidTransform) {
    const inv = invert4(host.desktopView());
    const hm = head.matrix;
    const headFwd: Vec3 = [-hm[8], -hm[9], -hm[10]];
    if (!inv) return;
    // desktop camera position / forward in splat world, taken through F
    const cp: Vec3 = [inv[12], -inv[13], -inv[14]];
    const cf: Vec3 = [inv[8], -inv[9], -inv[10]];
    place.s = 1;
    place.yaw = heading(headFwd[0], headFwd[2]) - heading(cf[0], cf[2]);
    const r = rotY(cp, place.yaw);
    place.t = [head.position.x - r[0], head.position.y - r[1], head.position.z - r[2]];
    placed = true;
  }

  // ---- input ---------------------------------------------------------------
  const sources = new Map<XRInputSource, SourceState>();
  const stateOf = (src: XRInputSource) => {
    let st = sources.get(src);
    if (!st) {
      st = { holds: 0, prev: null, tapStart: 0, tapTravel: 0, buttons: [], turned: false };
      sources.set(src, st);
    }
    return st;
  };
  const press = (e: XRInputSourceEvent) => {
    const st = stateOf(e.inputSource);
    st.holds++;
    if (e.type === 'selectstart') {
      st.tapStart = performance.now();
      st.tapTravel = 0;
    }
  };
  const release = (e: XRInputSourceEvent) => {
    const st = stateOf(e.inputSource);
    st.holds = Math.max(0, st.holds - 1);
    if (e.type === 'selectend' && performance.now() - st.tapStart < 300 && st.tapTravel < 0.03) host.togglePlay();
  };
  session.addEventListener('selectstart', press);
  session.addEventListener('squeezestart', press);
  session.addEventListener('selectend', release);
  session.addEventListener('squeezeend', release);
  session.addEventListener('inputsourceschange', (e) => {
    for (const src of e.removed) sources.delete(src);
  });

  let replaceRequested = false;

  function handleInput(frame: XRFrame, head: XRRigidTransform, dt: number) {
    const held: { prev: Vec3; now: Vec3 }[] = [];
    for (const src of session.inputSources) {
      const st = stateOf(src);
      const pose = frame.getPose(src.gripSpace ?? src.targetRaySpace, refSpace);
      const now: Vec3 | null = pose ? [pose.transform.position.x, pose.transform.position.y, pose.transform.position.z] : null;
      if (now && st.prev) {
        st.tapTravel += Math.hypot(now[0] - st.prev[0], now[1] - st.prev[1], now[2] - st.prev[2]);
        if (st.holds > 0) held.push({ prev: st.prev, now });
      }
      st.prev = now;

      const gp = src.gamepad;
      if (!gp) continue;
      const down = (i: number) => !!gp.buttons[i]?.pressed;
      const edge = (i: number) => down(i) && !st.buttons[i];
      if (edge(4)) host.togglePlay();
      if (edge(5)) replaceRequested = true;
      st.buttons = gp.buttons.map((b) => b.pressed);

      const dead = (v: number | undefined) => (v !== undefined && Math.abs(v) > 0.25 ? v : 0);
      const ax = dead(gp.axes[2]);
      const ay = dead(gp.axes[3]);
      const hp: Vec3 = [head.position.x, head.position.y, head.position.z];
      if (src.handedness === 'left') {
        if (ax || ay) {
          // walk: move the viewer along the head's level forward/right axes,
          // i.e. slide the scene the opposite way
          const hm = head.matrix;
          const fl = Math.hypot(hm[8], hm[10]) || 1;
          const fx = -hm[8] / fl;
          const fz = -hm[10] / fl;
          const step = WALK_SPEED * dt;
          const mx = (fx * -ay - fz * ax) * step;
          const mz = (fz * -ay + fx * ax) * step;
          place.t = [place.t[0] - mx, place.t[1], place.t[2] - mz];
        }
      } else {
        // snap turn about the viewer (comfortable; no continuous rotation)
        const rawX = gp.axes[2] ?? 0;
        if (!st.turned && Math.abs(rawX) > 0.6) {
          pivot(place, hp, hp, 1, Math.sign(rawX) * SNAP_TURN);
          st.turned = true;
        } else if (Math.abs(rawX) < 0.3) {
          st.turned = false;
        }
        if (ay && Math.abs(ay) > Math.abs(ax)) host.scrub(-ay * dt * 0.25);
      }
    }

    if (held.length === 1) {
      const { prev, now } = held[0];
      pivot(place, prev, now, 1, 0);
    } else if (held.length >= 2) {
      const [a, b] = held;
      const mid = (p: Vec3, q: Vec3): Vec3 => [(p[0] + q[0]) / 2, (p[1] + q[1]) / 2, (p[2] + q[2]) / 2];
      const d0: Vec3 = [b.prev[0] - a.prev[0], b.prev[1] - a.prev[1], b.prev[2] - a.prev[2]];
      const d1: Vec3 = [b.now[0] - a.now[0], b.now[1] - a.now[1], b.now[2] - a.now[2]];
      const l0 = Math.hypot(...d0);
      const l1 = Math.hypot(...d1);
      if (l0 > 0.02 && l1 > 0.02) {
        pivot(place, mid(a.prev, b.prev), mid(a.now, b.now), l1 / l0, heading(d1[0], d1[2]) - heading(d0[0], d0[2]));
      }
    }
  }

  // ---- adaptive resolution -------------------------------------------------
  // Fill rate is the bottleneck: when frames are being dropped, shrink the
  // per-eye viewport; creep back up while the frame rate holds.
  let vpScale = 1;
  let vpCeil = 1;
  let justRaised = false;
  let stableSecs = 0;
  let statFrames = 0;
  let statT = 0;
  let warmup = 2;

  function adapt(now: number) {
    statFrames++;
    if (!statT) statT = now;
    if (now - statT < 1000) return;
    const fps = (statFrames * 1000) / (now - statT);
    statFrames = 0;
    statT = now;
    host.onStats(fps, vpScale);
    if (warmup > 0) {
      warmup--;
      return;
    }
    const target = session.frameRate ?? opt.frameRate ?? 72;
    if (fps < target * 0.9) {
      if (justRaised) vpCeil = Math.max(opt.minViewportScale, vpScale / 1.06);
      vpScale = Math.max(opt.minViewportScale, Math.min(vpCeil, vpScale * 0.88));
      justRaised = false;
      stableSecs = 0;
    } else if (++stableSecs >= 4 && vpScale < vpCeil) {
      vpScale = Math.min(vpCeil, vpScale * 1.06);
      justRaised = true;
      stableSecs = 0;
    } else if (stableSecs >= 4) {
      justRaised = false;
    }
  }

  // ---- frame loop ----------------------------------------------------------
  const sortDir = new Float32Array(16);
  let lastT = 0;

  function onFrame(now: number, frame: XRFrame) {
    session.requestAnimationFrame(onFrame);
    const dt = lastT ? Math.min((now - lastT) / 1000, 0.1) : 0;
    lastT = now;
    host.tick(now);

    const pose = frame.getViewerPose(refSpace);
    const fb = session.renderState.baseLayer!.framebuffer;
    if (!pose) {
      host.renderer.renderViews([], fb, []);
      return;
    }
    if (!placed || replaceRequested) {
      placeFromHead(pose.transform);
      replaceRequested = false;
    }
    handleInput(frame, pose.transform, dt);
    if (session.visibilityState === 'visible') adapt(now);

    const model = modelMatrix(place);

    // depth sort along the head's forward axis, in splat-world coordinates
    const hv = flipRows(multiply4(Array.from(pose.transform.inverse.matrix), model));
    const n = Math.hypot(hv[2], hv[6], hv[10]) || 1;
    sortDir[2] = hv[2] / n;
    sortDir[6] = hv[6] / n;
    sortDir[10] = hv[10] / n;
    host.postSort(sortDir);

    const views: GlView[] = [];
    for (const v of pose.views) {
      v.requestViewportScale?.(vpScale);
      const vp = session.renderState.baseLayer!.getViewport(v);
      if (!vp || vp.width === 0) continue;
      const P = v.projectionMatrix;
      views.push({
        x: vp.x,
        y: vp.y,
        w: vp.width,
        h: vp.height,
        projection: flipCols(P),
        view: flipRows(multiply4(Array.from(v.transform.inverse.matrix), model)),
        fx: (P[0] * vp.width) / 2,
        fy: (P[5] * vp.height) / 2,
      });
    }
    host.renderer.renderViews(host.sets(), fb, views);
  }

  session.addEventListener('end', () => {
    host.renderer.setLod(1024, 0);
    host.onEnd();
  });
  session.requestAnimationFrame(onFrame);
  return session;
}
