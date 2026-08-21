// WebGL2 volume ray caster for the phantom studio.
//
// Hand-rolled rather than pulled from a CDN: the studio must work with no
// network (Colab, an air-gapped lab machine), and a label volume needs
// exactly one rendering mode, so a general-purpose 3-D library would be
// almost entirely dead weight.
//
// How it works
//   * the volume is one R8 3-D texture, NEAREST-filtered — a label map must
//     never be interpolated, or the renderer invents tissues between two real
//     ones;
//   * colour and opacity come from a 256x1 RGBA "transfer function" texture,
//     so the same shader draws a categorical label map and a continuous
//     property field: only the 256 entries change, and changing them is a
//     texture upload, not a re-render of the volume;
//   * the camera is passed as an explicit basis (position + right/up/forward
//     + tan(fov/2)), so the shader builds rays with no matrix inverse at all;
//   * front-to-back compositing with early termination, and a central
//     -difference gradient on opacity for diffuse shading, which is what makes
//     the skin shell read as a surface instead of a fog.
//
// Opacity is corrected for step size (1 - (1-a)^(dt/dt0)) so changing the
// quality slider changes the noise, never the brightness.

const VERT = `#version 300 es
precision highp float;
const vec2 P[3] = vec2[3](vec2(-1.0,-1.0), vec2(3.0,-1.0), vec2(-1.0,3.0));
void main() { gl_Position = vec4(P[gl_VertexID], 0.0, 1.0); }
`;

const FRAG = `#version 300 es
precision highp float;
precision highp sampler3D;

uniform sampler3D uVolume;      // R8, indexed [z][y][x] in memory order
uniform sampler2D uTransfer;    // 256x1 RGBA: colour + opacity per value
uniform vec3  uHalf;            // half-extent of the box in world units
uniform vec3  uDims;            // voxel counts (x, y, z)
uniform vec3  uNotch;           // normalized [0,1] corner of the cutaway
uniform float uNotchOn;         // 0 = whole volume, 1 = bite the corner out
uniform vec3  uCamPos;
uniform vec3  uCamRight;
uniform vec3  uCamUp;
uniform vec3  uCamFwd;
uniform vec2  uResolution;
uniform float uTanHalfFov;
uniform float uSteps;
uniform float uDensity;         // global opacity multiplier
uniform float uLight;           // 0 = flat, 1 = full diffuse shading
uniform vec3  uBackground;

out vec4 fragColor;

// Position in the box -> normalized volume coords.
vec3 toUnit(vec3 p) { return (p / uHalf) * 0.5 + 0.5; }

// Sample, honouring the cutaway. The notch REMOVES the octant beyond the
// three slice planes rather than keeping only it: taking a bite out of the
// corner leaves seven eighths of the anatomy on screen with three clean
// internal faces, where clipping to a box would throw away everything but a
// small wedge.
float sampleVol(vec3 p) {
  vec3 u = toUnit(p);
  if (uNotchOn > 0.5 && all(greaterThan(u, uNotch))) return 0.0;
  return texture(uVolume, u.zyx).r;
}

vec4 transfer(float v) { return texture(uTransfer, vec2(v, 0.5)); }

// slab intersection against an axis-aligned box
bool hitBox(vec3 ro, vec3 rd, vec3 lo, vec3 hi, out float t0, out float t1) {
  vec3 inv = 1.0 / rd;
  vec3 a = (lo - ro) * inv;
  vec3 b = (hi - ro) * inv;
  vec3 s = min(a, b), e = max(a, b);
  t0 = max(max(s.x, s.y), s.z);
  t1 = min(min(e.x, e.y), e.z);
  return t1 > max(t0, 0.0);
}

void main() {
  vec2 ndc = (gl_FragCoord.xy / uResolution) * 2.0 - 1.0;
  float aspect = uResolution.x / uResolution.y;
  vec3 rd = normalize(uCamFwd
                      + ndc.x * uTanHalfFov * aspect * uCamRight
                      + ndc.y * uTanHalfFov * uCamUp);

  float t0, t1;
  if (!hitBox(uCamPos, rd, -uHalf, uHalf, t0, t1)) {
    fragColor = vec4(uBackground, 1.0);
    return;
  }
  t0 = max(t0, 0.0);

  float steps = uSteps;
  float dt = (t1 - t0) / steps;
  // reference step = one voxel along the shortest axis
  float dt0 = 2.0 * min(uHalf.x / uDims.x, min(uHalf.y / uDims.y, uHalf.z / uDims.z));
  float corr = dt / dt0;

  // Gradient taps sit ~1.6 voxels out, not 1. The volume is NEAREST-filtered
  // (a label map must not be interpolated), so the opacity field it feeds is
  // piecewise constant: a one-voxel central difference is zero almost
  // everywhere and flips sign with the ray's sub-voxel phase, which paints
  // concentric moire rings onto every curved shell. A wider tap averages
  // over the plateau and gives a stable normal.
  vec3 grad = vec3(3.2) * uHalf / uDims;
  vec3 acc = vec3(0.0);
  float alpha = 0.0;

  // jitter the entry point to trade banding for a little noise
  float jitter = fract(sin(dot(gl_FragCoord.xy, vec2(12.9898, 78.233))) * 43758.5453);
  float t = t0 + dt * jitter;

  for (int i = 0; i < 1024; ++i) {
    if (float(i) >= steps || alpha > 0.985) break;
    vec3 p = uCamPos + rd * t;
    vec4 c = transfer(sampleVol(p));
    float a = 1.0 - pow(max(1.0 - c.a * uDensity, 0.0), corr);
    if (a > 0.002) {
      vec3 rgb = c.rgb;
      if (uLight > 0.0) {
        // gradient of OPACITY, not of the raw value: for a label map the raw
        // value has no meaningful slope, but the visible/invisible boundary
        // does, and that boundary is the surface the eye expects to see lit.
        float ax = transfer(sampleVol(p + vec3(grad.x, 0.0, 0.0))).a
                 - transfer(sampleVol(p - vec3(grad.x, 0.0, 0.0))).a;
        float ay = transfer(sampleVol(p + vec3(0.0, grad.y, 0.0))).a
                 - transfer(sampleVol(p - vec3(0.0, grad.y, 0.0))).a;
        float az = transfer(sampleVol(p + vec3(0.0, 0.0, grad.z))).a
                 - transfer(sampleVol(p - vec3(0.0, 0.0, grad.z))).a;
        vec3 n = vec3(ax, ay, az);
        float len = length(n);
        if (len > 1e-5) {
          n /= len;
          vec3 L = normalize(uCamRight * 0.45 + uCamUp * 0.65 - uCamFwd);
          // Diffuse only: a specular lobe on a quantised normal field is the
          // noisiest term there is, and it buys nothing on tissue.
          float diff = 0.55 + 0.45 * max(dot(n, L), 0.0);
          rgb = mix(rgb, rgb * diff, uLight);
        }
      }
      acc += (1.0 - alpha) * a * rgb;
      alpha += (1.0 - alpha) * a;
    }
    t += dt;
  }
  fragColor = vec4(acc + (1.0 - alpha) * uBackground, 1.0);
}
`;

function compile(gl, type, src) {
  const sh = gl.createShader(type);
  gl.shaderSource(sh, src);
  gl.compileShader(sh);
  if (!gl.getShaderParameter(sh, gl.COMPILE_STATUS)) {
    throw new Error("shader: " + gl.getShaderInfoLog(sh));
  }
  return sh;
}

export class VolumeView {
  constructor(canvas) {
    this.canvas = canvas;
    const gl = canvas.getContext("webgl2", { antialias: false, alpha: false });
    if (!gl) throw new Error("WebGL2 is not available in this browser");
    this.gl = gl;

    const prog = gl.createProgram();
    gl.attachShader(prog, compile(gl, gl.VERTEX_SHADER, VERT));
    gl.attachShader(prog, compile(gl, gl.FRAGMENT_SHADER, FRAG));
    gl.linkProgram(prog);
    if (!gl.getProgramParameter(prog, gl.LINK_STATUS)) {
      throw new Error("link: " + gl.getProgramInfoLog(prog));
    }
    this.prog = prog;
    this.u = {};
    for (const name of ["uVolume", "uTransfer", "uHalf", "uDims", "uNotch", "uNotchOn",
                        "uCamPos", "uCamRight", "uCamUp", "uCamFwd", "uResolution",
                        "uTanHalfFov", "uSteps", "uDensity", "uLight", "uBackground"]) {
      this.u[name] = gl.getUniformLocation(prog, name);
    }
    this.vao = gl.createVertexArray();

    this.volTex = gl.createTexture();
    this.tfTex = gl.createTexture();
    gl.bindTexture(gl.TEXTURE_2D, this.tfTex);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);

    this.dims = [1, 1, 1];
    this.half = [0.5, 0.5, 0.5];
    this.notch = [1, 1, 1];
    this.notchOn = 0;
    this.density = 1.0;
    this.light = 1.0;
    this.quality = 320;   // replaced per-volume in setVolume()
    this.background = [0.071, 0.094, 0.122];
    // Default view looks back along -z, i.e. from where the transducer sits,
    // so the first thing on screen is the face the beam enters.
    this.camera = { azimuth: -2.25, elevation: 0.38, distance: 2.35, target: [0, 0, 0] };
    this._installControls();
  }

  /**
   * Upload a new label/scalar volume.
   *
   * `data` is a Uint8Array of nx*ny*nz in C order, `shape` its DOWNSAMPLED
   * dims, and `extent` the PHYSICAL size of the full volume. The two are
   * passed separately on purpose: the server block-reduces only the axes
   * that exceed the texture budget, so a 126x210x168 build arrives as
   * 126x105x84 — deriving the box from the texture dims would squash the
   * breast by 2x along y and z.
   */
  setVolume(data, shape, extent) {
    const gl = this.gl;
    const [nx, ny, nz] = shape;
    this.dims = [nx, ny, nz];
    gl.bindTexture(gl.TEXTURE_3D, this.volTex);
    gl.pixelStorei(gl.UNPACK_ALIGNMENT, 1);
    // memory is z-fastest, so GL width = nz, height = ny, depth = nx
    gl.texImage3D(gl.TEXTURE_3D, 0, gl.R8, nz, ny, nx, 0, gl.RED, gl.UNSIGNED_BYTE, data);
    for (const p of [gl.TEXTURE_MIN_FILTER, gl.TEXTURE_MAG_FILTER]) {
      gl.texParameteri(gl.TEXTURE_3D, p, gl.NEAREST);
    }
    for (const p of [gl.TEXTURE_WRAP_S, gl.TEXTURE_WRAP_T, gl.TEXTURE_WRAP_R]) {
      gl.texParameteri(gl.TEXTURE_3D, p, gl.CLAMP_TO_EDGE);
    }
    const m = Math.max(...extent);
    this.half = extent.map((e) => 0.5 * (e / m));
    // Step count follows the volume, not a fixed number: the ~1.5 mm skin
    // shell is one or two voxels thick, and a ray stepping more than about
    // 0.7 voxels at a time walks straight through it, which shows up as a
    // stippled surface rather than a solid one.
    this.quality = Math.min(1000, Math.max(200, Math.ceil(2.5 * Math.max(nx, ny, nz))));
    this.render();
  }

  /** `entries` is a Uint8Array of 1024 bytes (256 RGBA). */
  setTransfer(entries) {
    const gl = this.gl;
    gl.bindTexture(gl.TEXTURE_2D, this.tfTex);
    gl.pixelStorei(gl.UNPACK_ALIGNMENT, 1);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, 256, 1, 0, gl.RGBA, gl.UNSIGNED_BYTE, entries);
    this.render();
  }

  /** Cut the octant beyond `corner` (normalized) out of the volume. */
  setNotch(corner, enabled) {
    this.notch = corner;
    this.notchOn = enabled ? 1 : 0;
    this.render();
  }

  resetCamera() {
    this.camera.azimuth = -2.25;
    this.camera.elevation = 0.38;
    this.camera.distance = 2.35;
    this.render();
  }

  _installControls() {
    const c = this.canvas;
    let dragging = false;
    let last = [0, 0];
    c.addEventListener("pointerdown", (e) => {
      dragging = true;
      last = [e.clientX, e.clientY];
      c.setPointerCapture(e.pointerId);
    });
    c.addEventListener("pointerup", (e) => {
      dragging = false;
      c.releasePointerCapture(e.pointerId);
    });
    c.addEventListener("pointermove", (e) => {
      if (!dragging) return;
      const dx = e.clientX - last[0];
      const dy = e.clientY - last[1];
      last = [e.clientX, e.clientY];
      this.camera.azimuth -= dx * 0.008;
      this.camera.elevation = Math.max(-1.5, Math.min(1.5, this.camera.elevation + dy * 0.008));
      this.render();
    });
    c.addEventListener("wheel", (e) => {
      e.preventDefault();
      this.camera.distance = Math.max(1.05, Math.min(9.0,
        this.camera.distance * (1 + Math.sign(e.deltaY) * 0.09)));
      this.render();
    }, { passive: false });
  }

  render() {
    const gl = this.gl;
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    const w = Math.max(1, Math.round(this.canvas.clientWidth * dpr));
    const h = Math.max(1, Math.round(this.canvas.clientHeight * dpr));
    if (this.canvas.width !== w || this.canvas.height !== h) {
      this.canvas.width = w;
      this.canvas.height = h;
    }
    gl.viewport(0, 0, w, h);
    gl.useProgram(this.prog);
    gl.bindVertexArray(this.vao);

    const { azimuth, elevation, distance } = this.camera;
    const ce = Math.cos(elevation);
    const pos = [
      distance * ce * Math.cos(azimuth),
      distance * Math.sin(elevation),
      distance * ce * Math.sin(azimuth),
    ];
    const fwd = norm([-pos[0], -pos[1], -pos[2]]);
    const right = norm(cross(fwd, [0, 1, 0]));
    const up = cross(right, fwd);

    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_3D, this.volTex);
    gl.uniform1i(this.u.uVolume, 0);
    gl.activeTexture(gl.TEXTURE1);
    gl.bindTexture(gl.TEXTURE_2D, this.tfTex);
    gl.uniform1i(this.u.uTransfer, 1);

    gl.uniform3fv(this.u.uHalf, this.half);
    gl.uniform3fv(this.u.uDims, this.dims);
    gl.uniform3fv(this.u.uNotch, this.notch);
    gl.uniform1f(this.u.uNotchOn, this.notchOn);
    gl.uniform3fv(this.u.uCamPos, pos);
    gl.uniform3fv(this.u.uCamRight, right);
    gl.uniform3fv(this.u.uCamUp, up);
    gl.uniform3fv(this.u.uCamFwd, fwd);
    gl.uniform2f(this.u.uResolution, w, h);
    gl.uniform1f(this.u.uTanHalfFov, Math.tan((36 * Math.PI) / 360));
    gl.uniform1f(this.u.uSteps, this.quality);
    gl.uniform1f(this.u.uDensity, this.density);
    gl.uniform1f(this.u.uLight, this.light);
    gl.uniform3fv(this.u.uBackground, this.background);

    gl.drawArrays(gl.TRIANGLES, 0, 3);
  }
}

function cross(a, b) {
  return [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
}
function norm(v) {
  const l = Math.hypot(v[0], v[1], v[2]) || 1;
  return [v[0] / l, v[1] / l, v[2] / l];
}
