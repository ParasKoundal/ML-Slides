/* core.js: the deck engine and the drawing kit.
 *
 * A lecture is a list of scenes. A scene has a number of beats; the presenter's
 * "next" key advances one beat, and every drawing is a function of
 * (beat, time since the beat started), so jumping backwards always lands on a
 * clean, finished frame. Live simulations keep their own state inside the scene.
 *
 * Stage coordinates are a fixed 1920 x 1080 canvas, scaled to the window.
 */
'use strict';

const W = 1920, H = 1080, TAU = Math.PI * 2;
const T = String.raw;   // M(T`\frac{a}{b}`): TeX without escaping backslashes
const COL = {
  bg: '#07090d', panel: '#0f141c', fg: '#e9eef5', dim: '#8792a3', faint: '#2a3240', grid: '#141a24',
  data: '#ffb547', model: '#3fd0ff', loss: '#ff4d7d', grad: '#b48cff', good: '#5ef0a5', cls0: '#3f8cff', cls1: '#ffb547',
};

/* ---------------------------------------------------------------- numbers and easing */
const clamp = (x, a = 0, b = 1) => Math.min(b, Math.max(a, x));
const lerp = (a, b, t) => a + (b - a) * t;
const seg = (p, a, b) => clamp((p - a) / (b - a));
const Ease = {
  lin: t => t,
  out: t => 1 - Math.pow(1 - t, 3),
  in: t => t * t * t,
  inOut: t => (t < .5 ? 4 * t * t * t : 1 - Math.pow(-2 * t + 2, 3) / 2),
  back: t => { const c1 = 1.70158, c3 = c1 + 1; return 1 + c3 * Math.pow(t - 1, 3) + c1 * Math.pow(t - 1, 2); },
  spring: t => 1 - Math.exp(-6 * t) * Math.cos(10 * t),
};
const E = (p, a = 0, b = 1) => Ease.inOut(seg(p, a, b));
const fmt = (v, d = 2) => (Math.abs(v) < 1e-12 ? 0 : v).toFixed(d).replace('-', '−');
const sci = (v, d = 1) => { const [m, e] = v.toExponential(d).split('e'); const sup = { '-': '⁻', '+': '', 0: '⁰', 1: '¹', 2: '²', 3: '³', 4: '⁴', 5: '⁵', 6: '⁶', 7: '⁷', 8: '⁸', 9: '⁹' }; return m.replace('-', '−') + ' × 10' + [...e].map(c => sup[c]).join(''); };
const linspace = (a, b, n) => Array.from({ length: n }, (_, i) => a + (b - a) * i / (n - 1));
const sum = a => a.reduce((s, v) => s + v, 0);
const mean = a => sum(a) / a.length;

function hexA(hex, a) {
  const n = parseInt(hex.slice(1), 16);
  return `rgba(${n >> 16 & 255},${n >> 8 & 255},${n & 255},${a})`;
}
function mixHex(h1, h2, t) {
  const a = parseInt(h1.slice(1), 16), b = parseInt(h2.slice(1), 16);
  const r = Math.round(lerp(a >> 16 & 255, b >> 16 & 255, t)), g = Math.round(lerp(a >> 8 & 255, b >> 8 & 255, t)), bl = Math.round(lerp(a & 255, b & 255, t));
  return `rgb(${r},${g},${bl})`;
}
/* colour ramp for loss surfaces and probability fields */
const RAMP = ['#0a1830', '#0f3b5c', '#11718a', '#2fae9c', '#a8dc82', '#ffe38a', '#ffb547'];
function ramp(t, stops = RAMP) {
  t = clamp(t) * (stops.length - 1);
  const i = Math.min(stops.length - 2, Math.floor(t));
  return mixHex(stops[i], stops[i + 1], t - i);
}
function rampRGB(t, stops = RAMP) {
  const s = ramp(t, stops).match(/\d+/g).map(Number);
  return s;
}

/* ---------------------------------------------------------------- drawing kit */
const D = {
  g: null,
  dot(x, y, r, col, a = 1, glow = 0) {
    const g = this.g; g.save(); g.globalAlpha *= a;
    if (glow) { g.shadowColor = col; g.shadowBlur = glow; }
    g.fillStyle = col; g.beginPath(); g.arc(x, y, Math.max(0, r), 0, TAU); g.fill(); g.restore();
  },
  ring(x, y, r, col, w = 2, a = 1) {
    const g = this.g; g.save(); g.globalAlpha *= a; g.strokeStyle = col; g.lineWidth = w;
    g.beginPath(); g.arc(x, y, r, 0, TAU); g.stroke(); g.restore();
  },
  line(pts, col, w = 3, a = 1, dash = null, glow = 0) {
    if (pts.length < 2) return;
    const g = this.g; g.save(); g.globalAlpha *= a; g.strokeStyle = col; g.lineWidth = w; g.lineJoin = 'round'; g.lineCap = 'round';
    if (dash) g.setLineDash(dash);
    if (glow) { g.shadowColor = col; g.shadowBlur = glow; }
    g.beginPath(); g.moveTo(pts[0][0], pts[0][1]);
    for (let i = 1; i < pts.length; i++) g.lineTo(pts[i][0], pts[i][1]);
    g.stroke(); g.restore();
  },
  seg(x1, y1, x2, y2, col, w = 2, a = 1, dash = null) { this.line([[x1, y1], [x2, y2]], col, w, a, dash); },
  poly(pts, fill, a = 1, stroke = null, sw = 2) {
    const g = this.g; g.save(); g.globalAlpha *= a; g.beginPath(); g.moveTo(pts[0][0], pts[0][1]);
    for (let i = 1; i < pts.length; i++) g.lineTo(pts[i][0], pts[i][1]);
    g.closePath(); if (fill) { g.fillStyle = fill; g.fill(); }
    if (stroke) { g.strokeStyle = stroke; g.lineWidth = sw; g.stroke(); } g.restore();
  },
  rect(x, y, w, h, fill, a = 1, stroke = null, sw = 2, r = 0) {
    const g = this.g; g.save(); g.globalAlpha *= a; g.beginPath();
    if (r) g.roundRect(x, y, w, h, r); else g.rect(x, y, w, h);
    if (fill) { g.fillStyle = fill; g.fill(); }
    if (stroke) { g.strokeStyle = stroke; g.lineWidth = sw; g.stroke(); } g.restore();
  },
  text(s, x, y, o = {}) {
    const g = this.g; g.save(); g.globalAlpha *= (o.a ?? 1);
    g.font = `${o.weight || 500} ${o.size || 24}px ${o.font || '"IBM Plex Mono", monospace'}`;
    g.fillStyle = o.col || COL.dim; g.textAlign = o.align || 'left'; g.textBaseline = o.base || 'middle';
    if (o.spacing) g.letterSpacing = o.spacing + 'px';
    if (o.rot) { g.translate(x, y); g.rotate(o.rot); g.fillText(s, 0, 0); } else g.fillText(s, x, y);
    g.restore();
  },
  arrow(x1, y1, x2, y2, col, w = 4, head = 18, a = 1, glow = 0) {
    const L = Math.hypot(x2 - x1, y2 - y1); if (L < 1) return;
    const ux = (x2 - x1) / L, uy = (y2 - y1) / L, hh = Math.min(head, L * .6);
    this.line([[x1, y1], [x2 - ux * hh * .8, y2 - uy * hh * .8]], col, w, a, null, glow);
    this.poly([[x2, y2], [x2 - ux * hh - uy * hh * .55, y2 - uy * hh + ux * hh * .55], [x2 - ux * hh + uy * hh * .55, y2 - uy * hh - ux * hh * .55]], col, a);
  },
  /* a coil spring between two points: residuals as springs */
  spring(x1, y1, x2, y2, col, coils = 9, amp = 9, w = 2.5, a = 1) {
    const L = Math.hypot(x2 - x1, y2 - y1); if (L < 2) return;
    const ux = (x2 - x1) / L, uy = (y2 - y1) / L, nx = -uy, ny = ux, pts = [[x1, y1]];
    const lead = Math.min(10, L * .15), n = coils * 2;
    pts.push([x1 + ux * lead, y1 + uy * lead]);
    for (let i = 1; i < n; i++) {
      const t = lead + (L - 2 * lead) * i / n, s = (i % 2 ? 1 : -1) * amp;
      pts.push([x1 + ux * t + nx * s, y1 + uy * t + ny * s]);
    }
    pts.push([x2 - ux * lead, y2 - uy * lead], [x2, y2]);
    this.line(pts, col, w, a);
  },
  /* smooth curve through a function on a plot */
  fn(P, f, x0, x1, col, w = 3, a = 1, n = 160, dash = null, glow = 0) {
    const pts = [];
    for (let i = 0; i <= n; i++) { const x = lerp(x0, x1, i / n), y = f(x); if (isFinite(y)) pts.push([P.X(x), P.Y(y)]); }
    this.line(pts, col, w, a, dash, glow);
  },
};

/* ---------------------------------------------------------------- 2-D plot frame */
class Plot {
  constructor(x, y, w, h, xr, yr, o = {}) {
    Object.assign(this, { x, y, w, h, xr, yr, logy: !!o.logy, logx: !!o.logx });
  }
  X(v) { const [a, b] = this.xr; if (this.logx) { v = Math.log10(v); return this.x + (v - Math.log10(a)) / (Math.log10(b) - Math.log10(a)) * this.w; } return this.x + (v - a) / (b - a) * this.w; }
  Y(v) { const [a, b] = this.yr; if (this.logy) { v = Math.log10(Math.max(v, 1e-12)); return this.y + this.h - (v - Math.log10(a)) / (Math.log10(b) - Math.log10(a)) * this.h; } return this.y + this.h - (v - a) / (b - a) * this.h; }
  ix(px) { const [a, b] = this.xr; return a + (px - this.x) / this.w * (b - a); }
  iy(py) { const [a, b] = this.yr; return a + (this.y + this.h - py) / this.h * (b - a); }
  inside(px, py, pad = 0) { return px >= this.x - pad && px <= this.x + this.w + pad && py >= this.y - pad && py <= this.y + this.h + pad; }
  axes(o = {}) {
    const a = o.a ?? 1, g = D.g;
    if (o.grid) {
      for (const t of o.xt || []) D.seg(this.X(t), this.y, this.X(t), this.y + this.h, COL.grid, 1, a);
      for (const t of o.yt || []) D.seg(this.x, this.Y(t), this.x + this.w, this.Y(t), COL.grid, 1, a);
    }
    D.seg(this.x, this.y + this.h, this.x + this.w * (o.grow ?? 1), this.y + this.h, '#3a4352', 2, a);
    D.seg(this.x, this.y + this.h, this.x, this.y + this.h - this.h * (o.grow ?? 1), '#3a4352', 2, a);
    for (const t of o.xt || []) D.text(o.xf ? o.xf(t) : String(t), this.X(t), this.y + this.h + 26, { size: 20, align: 'center', a: a * .9 });
    for (const t of o.yt || []) D.text(o.yf ? o.yf(t) : String(t), this.x - 14, this.Y(t), { size: 20, align: 'right', a: a * .9 });
    if (o.xl) D.text(o.xl, this.x + this.w, this.y + this.h + 62, { size: 22, align: 'right', a, col: o.xlc || COL.dim, font: '"IBM Plex Sans", sans-serif' });
    if (o.yl) D.text(o.yl, this.x - (o.ylx ?? 0), this.y - 30, { size: 22, align: 'left', a, col: o.ylc || COL.dim, font: '"IBM Plex Sans", sans-serif' });
    void g;
  }
}

/* ---------------------------------------------------------------- 3-D: a small orthographic camera and shaded surfaces */
class Cam {
  constructor(cx, cy, scale, yaw = -0.7, pitch = 0.55, persp = 0.18) { Object.assign(this, { cx, cy, scale, yaw, pitch, persp }); }
  p(x, y, z) {
    const cy = Math.cos(this.yaw), sy = Math.sin(this.yaw), cp = Math.cos(this.pitch), sp = Math.sin(this.pitch);
    const x1 = x * cy - y * sy, y1 = x * sy + y * cy;
    const depth = y1 * cp + z * sp;            // larger = further away
    const up = z * cp - y1 * sp;
    const f = 1 / (1 + this.persp * depth);
    return [this.cx + this.scale * x1 * f, this.cy - this.scale * up * f, depth];
  }
}
/* surface z = f(u,v) over u,v in [-1,1]; zf maps values to height; colour by height */
function drawSurface(cam, grid, o = {}) {
  const n = grid.length - 1, quads = [];
  const zs = o.zscale ?? 1, a = o.a ?? 1, light = [-.4, -.5, .77];
  for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) {
    const u0 = -1 + 2 * i / n, u1 = -1 + 2 * (i + 1) / n, v0 = -1 + 2 * j / n, v1 = -1 + 2 * (j + 1) / n;
    const z00 = grid[i][j], z10 = grid[i + 1][j], z11 = grid[i + 1][j + 1], z01 = grid[i][j + 1];
    const P = [cam.p(u0, v0, z00 * zs), cam.p(u1, v0, z10 * zs), cam.p(u1, v1, z11 * zs), cam.p(u0, v1, z01 * zs)];
    const zc = (z00 + z10 + z11 + z01) / 4;
    // normal = (du, 0, dz/du) x (0, dv, dz/dv)
    const du = 2 / n, dzu = ((z10 - z00) + (z11 - z01)) / 2 * zs, dzv = ((z01 - z00) + (z11 - z10)) / 2 * zs;
    const nx = -dzu * du, ny = -du * dzv, nz = du * du;
    const nl = Math.hypot(nx, ny, nz) || 1;
    let lam = clamp(.35 + .75 * (nx * light[0] + ny * light[1] + nz * light[2]) / nl, .15, 1.15);
    if (o.spot) { const du_ = (u0 + u1) / 2 - o.spot[0], dv_ = (v0 + v1) / 2 - o.spot[1]; lam *= lerp(o.spot[3] ?? .12, 1, Math.exp(-(du_ * du_ + dv_ * dv_) / (o.spot[2] ** 2))); }
    quads.push({ P, d: (P[0][2] + P[1][2] + P[2][2] + P[3][2]) / 4, zc, lam });
  }
  quads.sort((q1, q2) => q2.d - q1.d);
  const g = D.g; g.save(); g.globalAlpha *= a;
  const zmin = o.zmin ?? 0, zmax = o.zmax ?? 1;
  for (const q of quads) {
    const t = (q.zc - zmin) / (zmax - zmin);
    const [r, gg, b] = rampRGB(o.flip ? 1 - t : t, o.stops);
    g.fillStyle = `rgb(${Math.min(255, r * q.lam) | 0},${Math.min(255, gg * q.lam) | 0},${Math.min(255, b * q.lam) | 0})`;
    g.beginPath(); g.moveTo(q.P[0][0], q.P[0][1]);
    for (let k = 1; k < 4; k++) g.lineTo(q.P[k][0], q.P[k][1]);
    g.closePath(); g.fill();
    if (o.wire) { g.strokeStyle = o.wire; g.lineWidth = .8; g.stroke(); } else { g.strokeStyle = g.fillStyle; g.lineWidth = .6; g.stroke(); }
  }
  g.restore();
}
/* contour-like heat image of f over a plot, cached as an offscreen canvas */
function heatImage(P, f, o = {}) {
  const res = o.res || 3, w = Math.ceil(P.w / res), h = Math.ceil(P.h / res);
  const c = document.createElement('canvas'); c.width = w; c.height = h;
  const g = c.getContext('2d'), img = g.createImageData(w, h);
  let lo = Infinity, hi = -Infinity; const vals = new Float32Array(w * h);
  for (let j = 0; j < h; j++) for (let i = 0; i < w; i++) {
    const v = f(P.ix(P.x + (i + .5) * res), P.iy(P.y + (j + .5) * res)); vals[j * w + i] = v;
    if (v < lo) lo = v; if (v > hi) hi = v;
  }
  if (o.lo !== undefined) lo = o.lo; if (o.hi !== undefined) hi = o.hi;
  const bands = o.bands || 0;
  for (let k = 0; k < w * h; k++) {
    let t = (o.map ? o.map(vals[k]) : (vals[k] - lo) / (hi - lo));
    t = clamp(t);
    let shade = 1;
    if (bands) { const fr = (t * bands) % 1; shade = fr < .06 ? 1.45 : 1; }
    const [r, g2, b] = rampRGB(t, o.stops);
    img.data[k * 4] = Math.min(255, r * shade); img.data[k * 4 + 1] = Math.min(255, g2 * shade); img.data[k * 4 + 2] = Math.min(255, b * shade); img.data[k * 4 + 3] = 255 * (o.alpha ?? 1);
  }
  g.putImageData(img, 0, 0);
  return c;
}
function drawImg(img, P, a = 1) { const g = D.g; g.save(); g.globalAlpha *= a; g.imageSmoothingEnabled = true; g.drawImage(img, P.x, P.y, P.w, P.h); g.restore(); }

/* ---------------------------------------------------------------- maths typesetting (pre-rendered SVG, see tools/build.mjs) */
function M(tex) {
  const svg = window.MATH && window.MATH[tex];
  if (svg) return `<span class="m">${svg}</span>`;
  return `<span class="m tex-fallback">${tex.replace(/</g, '&lt;')}</span>`;
}
/* split a sentence into kinetic words */
function words(s, cls = '', b = 0, extra = '') {
  let i = 0;
  const inner = s.split(/(\s+)/).map(w => (/^\s+$/.test(w) ? w : `<span class="kw" style="--i:${i++}">${w}</span>`)).join('');
  return `<div class="words ${cls}" data-b="${b}" ${extra}>${inner}</div>`;
}

/* ---------------------------------------------------------------- the deck */
const Deck = {
  scenes: [], chapters: [], rec: false, vnow: 0, idx: 0, beat: 0, beatStart: 0, sceneStart: 0, last: 0, talkStart: 0,
  chapter(title, sub) { this.chapters.push({ title, sub }); return this.chapters.length - 1; },
  add(s) { s.beats = s.beats || 1; s.ch = this.chapters.length - 1; this.scenes.push(s); return s; },

  clock() { return this.rec ? this.vnow : performance.now(); },
  init() {
    this.stage = document.getElementById('stage');
    this.cv = document.getElementById('cv');
    this.layer = document.getElementById('layer');
    this.g = this.cv.getContext('2d'); D.g = this.g;
    this.resize(); addEventListener('resize', () => this.resize());
    this.buildChrome();
    addEventListener('keydown', e => this.key(e));
    this.stage.addEventListener('pointerdown', e => this.pointer('down', e));
    addEventListener('pointermove', e => this.pointer('move', e));
    addEventListener('pointerup', e => this.pointer('up', e));
    let tx = null;
    this.stage.addEventListener('touchstart', e => { tx = e.touches[0].clientX; }, { passive: true });
    this.stage.addEventListener('touchend', e => { if (tx === null) return; const dx = e.changedTouches[0].clientX - tx; tx = null; if (Math.abs(dx) > 60) dx < 0 ? this.next() : this.prev(); }, { passive: true });
    addEventListener('hashchange', () => this.fromHash());
    if (this.rec) { document.getElementById('hint')?.remove(); this.go(0, 0); return; }
    this.fromHash(true);
    const loop = now => { this.frame(now); requestAnimationFrame(loop); };
    requestAnimationFrame(loop);
    let idle; if (this.rec) return; const wake = () => { this.chromeEl.classList.remove('idle'); clearTimeout(idle); idle = setTimeout(() => this.chromeEl.classList.add('idle'), 2500); };
    addEventListener('pointermove', wake); wake();
    setTimeout(() => document.getElementById('hint')?.classList.add('gone'), 7000);
  },
  resize() {
    const s = Math.min(innerWidth / W, innerHeight / H);
    this.scale = s;
    this.stage.style.transform = `translate(${-W * s / 2}px, ${-H * s / 2}px) scale(${s})`;
    const dpr = Math.min(2, (window.devicePixelRatio || 1) * s);
    this.cv.width = Math.round(W * dpr); this.cv.height = Math.round(H * dpr);
    this.dpr = dpr;
  },
  fromHash(first) {
    const h = location.hash.slice(1);
    let i = this.scenes.findIndex(s => s.id === h);
    if (i < 0) i = first ? 0 : this.idx;
    if (first || i !== this.idx) this.go(i, 0);
  },
  go(i, beat = 0, instant = false) {
    i = clamp(i, 0, this.scenes.length - 1);
    const prev = this.scenes[this.idx];
    if (prev && prev !== this.scenes[i] && prev.leave) prev.leave();
    const s = this.scenes[i];
    const changed = i !== this.idx || !this.entered;
    this.idx = i; this.entered = true;
    const now = this.clock();
    if (changed) {
      this.sceneStart = now;
      this.layer.className = '';
      this.layer.innerHTML = typeof s.html === 'function' ? s.html() : (s.html || '');
      s.state = {};
      if (s.enter) s.enter(s.state, this.layer);
      if (!this.talkStart && i > 0) this.talkStart = now;
      const id = '#' + s.id; if (!this.rec && location.hash !== id) history.replaceState(null, '', id);
    }
    this.setBeat(beat, instant);
    this.updateChrome();
  },
  setBeat(b, instant) {
    const s = this.scenes[this.idx];
    this.beat = clamp(b, 0, s.beats - 1);
    this.beatStart = instant ? -1e9 : this.clock();
    this.reveals = [...this.layer.querySelectorAll('[data-b]')].map(el => ({
      el, b: +el.dataset.b, x: el.dataset.x === undefined ? null : +el.dataset.x,
      d: parseFloat(el.style.getPropertyValue('--d')) || 0,
      kind: ['words', 'pop', 'blur', 'wipe', 'slam', 'rise', 'fade'].find(k => el.classList.contains(k)) || 'fade',
      kws: el.classList.contains('words') ? [...el.querySelectorAll('.kw')] : null, last: '',
    }));
    if (s.onBeat) s.onBeat(this.beat, s.state, instant);
    this.renderNotes();
  },
  next() {
    const s = this.scenes[this.idx];
    if (this.beat < s.beats - 1) this.setBeat(this.beat + 1);
    else if (this.idx < this.scenes.length - 1) this.go(this.idx + 1, 0);
  },
  prev() {
    if (this.beat > 0) this.setBeat(this.beat - 1, true);
    else if (this.idx > 0) { const p = this.scenes[this.idx - 1]; this.go(this.idx - 1, p.beats - 1, true); }
  },
  key(e) {
    if (e.target && /INPUT|TEXTAREA|SELECT/.test(e.target.tagName) && !/Arrow(Left|Right)|PageUp|PageDown/.test(e.key)) return;
    const k = e.key;
    if (['ArrowRight', 'PageDown', ' ', 'Enter', 'ArrowDown'].includes(k)) { e.preventDefault(); this.next(); }
    else if (['ArrowLeft', 'PageUp', 'Backspace', 'ArrowUp'].includes(k)) { e.preventDefault(); this.prev(); }
    else if (k === 'Home') this.go(0);
    else if (k === 'End') this.go(this.scenes.length - 1);
    else if (k === 'o' || k === 'O' || k === 'Escape') this.toggleOverview(k === 'Escape' ? false : undefined);
    else if (k === 'n' || k === 'N') this.toggleNotes();
    else if (k === 'f' || k === 'F') { if (!document.fullscreenElement) document.documentElement.requestFullscreen?.().catch(() => {}); else document.exitFullscreen?.(); }
    else if (k === 'b' || k === 'B' || k === '.') this.stage.style.visibility = this.stage.style.visibility === 'hidden' ? '' : 'hidden';
    else if (k === 'r' || k === 'R') { const s = this.scenes[this.idx]; this.entered = false; this.go(this.idx, 0); void s; }
  },
  toStage(e) { const r = this.stage.getBoundingClientRect(); return [(e.clientX - r.left) / this.scale, (e.clientY - r.top) / this.scale]; },
  pointer(type, e) {
    const s = this.scenes[this.idx];
    if (e.target.closest && e.target.closest('input,button,label,.ui')) return;
    const [x, y] = this.toStage(e);
    if (s.pointer && s.pointer(type, x, y, s.state)) { if (type === 'down') e.preventDefault(); return; }
    if (type === 'down' && e.button === 0 && !s.noClickAdvance && e.target.closest('#stage') && !e.target.closest('#chrome')) this.next();
  },
  frame(now) {
    const dt = Math.min(50, now - (this.last || now)); this.last = now;
    const s = this.scenes[this.idx]; if (!s) return;
    const g = this.g;
    g.setTransform(this.dpr, 0, 0, this.dpr, 0, 0);
    g.fillStyle = COL.bg; g.fillRect(0, 0, W, H);
    const bt = now - this.beatStart, t = now - this.sceneStart;
    const S = {
      beat: this.beat, bt, t, dt, now, st: s.state,
      /* progress of the current beat, after `delay` ms, over `dur` ms */
      p: (dur = 1000, delay = 0) => clamp((bt - delay) / dur),
      /* progress of an element that arrives on beat b */
      at: (b, dur = 1000, delay = 0) => (this.beat > b ? 1 : this.beat < b ? 0 : clamp((bt - delay) / dur)),
      /* 1 while beat in [b, x) with fade in/out */
      on: (b, x = 1e9, dur = 600) => (this.beat < b ? 0 : this.beat >= x ? (this.beat > x ? 0 : 1 - clamp(bt / dur)) : (this.beat > b ? 1 : clamp(bt / dur))),
    };
    this.tickReveals(bt);
    const fadeIn = clamp(t / 450);
    const lo = fadeIn.toFixed(3); if (this.layer.style.opacity !== lo) this.layer.style.opacity = lo;
    g.save(); g.globalAlpha = fadeIn;
    try { if (s.frame) s.frame(g, S); } catch (err) { console.error('scene', s.id, err); this.errors = (this.errors || 0) + 1; }
    g.restore();
    if (this.notesEl && !this.notesEl.hidden) this.tickClock(now);
  },

  /* ---- DOM reveals, driven by the same clock as the canvas so recordings are frame-exact */
  tickReveals(bt) {
    for (const r of this.reveals || []) {
      let p = this.beat < r.b ? 0 : this.beat > r.b ? 1 : clamp((bt - r.d) / (r.kind === 'words' ? 700 : 750));
      let out = 0;
      if (r.x !== null && this.beat >= r.x) out = this.beat > r.x ? 1 : clamp(bt / 450);
      const e = Ease.out(p), op = (r.kind === 'wipe' ? (p > 0 ? 1 : 0) : clamp(p * 1.4)) * (1 - out);
      let st;
      if (r.kind === 'words') {
        st = 'o' + op.toFixed(3);
        if (st !== r.last) r.el.style.opacity = out ? (1 - out).toFixed(3) : (p > 0 ? 1 : 0);
        const along = this.beat > r.b ? 1e9 : bt - r.d;
        r.kws.forEach((k, i) => {
          const q = clamp((along - i * 70) / 600), qe = Ease.out(q);
          const v = q.toFixed(3);
          if (k.dataset.v !== v) { k.dataset.v = v; k.style.opacity = clamp(q * 1.6).toFixed(3); k.style.transform = `translateY(${((1 - qe) * 0.45).toFixed(3)}em)`; k.style.filter = q < 1 ? `blur(${((1 - qe) * 8).toFixed(1)}px)` : 'none'; }
        });
        r.last = st; continue;
      }
      let tf = 'none', fl = 'none', cp = 'none';
      if (r.kind === 'rise') tf = `translateY(${((1 - e) * 40).toFixed(1)}px)`;
      else if (r.kind === 'pop') tf = `scale(${lerp(.6, 1, Ease.back(p)).toFixed(4)})`;
      else if (r.kind === 'blur') fl = p < 1 ? `blur(${((1 - e) * 18).toFixed(1)}px)` : 'none';
      else if (r.kind === 'slam') { const q = clamp((bt - r.d) / 450); const qq = this.beat > r.b ? 1 : q; tf = `scale(${lerp(1.8, 1, Ease.back(qq)).toFixed(4)})`; fl = qq < 1 ? `blur(${((1 - Ease.out(qq)) * 10).toFixed(1)}px)` : 'none'; }
      else if (r.kind === 'wipe') cp = `inset(0 ${((1 - Ease.inOut(p)) * 100).toFixed(2)}% 0 0)`;
      st = op.toFixed(3) + tf + fl + cp;
      if (st === r.last) continue;
      r.last = st; r.el.style.opacity = op.toFixed(3); r.el.style.transform = tf; r.el.style.filter = fl; if (r.kind === 'wipe') r.el.style.clipPath = cp;
    }
  },

  /* ---- chrome */
  buildChrome() {
    const c = document.createElement('div'); c.id = 'chrome';
    c.innerHTML = `<div class="row"><span id="chapLbl"></span><span id="cnt"></span></div><div id="track"></div>`;
    this.stage.appendChild(c); this.chromeEl = c;
    const tr = c.querySelector('#track'); this.segs = [];
    const n = this.scenes.length;
    this.chapters.forEach((ch, ci) => {
      const idxs = this.scenes.map((s, i) => [s, i]).filter(([s]) => s.ch === ci).map(([, i]) => i);
      const el = document.createElement('div'); el.className = 'seg'; el.style.flex = String(idxs.length);
      el.title = ch.title; el.innerHTML = '<i></i>';
      el.addEventListener('pointerdown', ev => { ev.stopPropagation(); this.go(idxs[0]); });
      tr.appendChild(el); this.segs.push({ el, idxs });
    });
    const head = document.createElement('div'); head.id = 'head'; tr.appendChild(head); this.headEl = head; void n;
    // overview
    const ov = document.createElement('div'); ov.id = 'overview'; ov.hidden = true; document.body.appendChild(ov); this.ovEl = ov;
    const nt = document.createElement('div'); nt.id = 'notes'; nt.hidden = true; document.body.appendChild(nt); this.notesEl = nt;
  },
  updateChrome() {
    const s = this.scenes[this.idx], ch = this.chapters[s.ch];
    document.getElementById('chapLbl').innerHTML = s.ch === 0 ? '<b>Learning by gradients</b>' : `${String(s.ch).padStart(2, '0')} · <b>${ch.title}</b>`;
    document.getElementById('cnt').textContent = `${this.idx + 1} / ${this.scenes.length}`;
    for (const sg of this.segs) {
      const k = sg.idxs.filter(i => i <= this.idx).length;
      sg.el.querySelector('i').style.width = (100 * k / sg.idxs.length) + '%';
    }
    const cur = this.segs.find(sg => sg.idxs.includes(this.idx));
    if (cur) {
      const tr = document.getElementById('track').getBoundingClientRect(), r = cur.el.getBoundingClientRect();
      const k = cur.idxs.indexOf(this.idx) + 1;
      this.headEl.style.left = ((r.left - tr.left) / this.scale + (r.width / this.scale) * k / cur.idxs.length - 7) + 'px';
    }
    this.chromeEl.style.display = s.chrome === false ? 'none' : '';
    if (!this.ovEl.hidden) this.renderOverview();
  },
  toggleOverview(force) {
    this.ovEl.hidden = force === undefined ? !this.ovEl.hidden : !force;
    if (!this.ovEl.hidden) this.renderOverview();
  },
  renderOverview() {
    const parts = this.chapters.map((ch, ci) => {
      const btns = this.scenes.map((s, i) => [s, i]).filter(([s]) => s.ch === ci)
        .map(([s, i]) => `<button data-i="${i}" class="${i === this.idx ? 'cur' : ''}">${i + 1}. ${s.title || s.id}</button>`).join('');
      return `<div class="chap"><h3>${ci === 0 ? 'Opening' : String(ci).padStart(2, '0') + ' · ' + ch.title}</h3><div class="grid">${btns}</div></div>`;
    }).join('');
    this.ovEl.innerHTML = `<h2>Learning by gradients · scenes</h2>${parts}<div class="keys">→ / space / click: next beat · ← : back · O: this overview · N: speaker notes and clock · F: full screen · B: black screen · R: replay scene</div>`;
    this.ovEl.querySelectorAll('button').forEach(b => b.addEventListener('click', () => { this.go(+b.dataset.i); this.toggleOverview(false); }));
  },
  toggleNotes() { this.notesEl.hidden = !this.notesEl.hidden; this.renderNotes(); },
  renderNotes() {
    if (!this.notesEl || this.notesEl.hidden) return;
    const s = this.scenes[this.idx];
    const notes = Array.isArray(s.notes) ? s.notes : [s.notes || ''];
    const ps = notes.map((n, i) => `<p class="${i === this.beat || (notes.length === 1) ? 'now' : i > this.beat ? 'later' : ''}">${notes.length > 1 ? `<b class="c-dim">[${i + 1}]</b> ` : ''}${n}</p>`).join('');
    this.notesEl.innerHTML = `<div class="top"><span>Scene ${this.idx + 1}: <b>${s.title || s.id}</b> · beat ${this.beat + 1} / ${s.beats} · target ${s.min || 1.5} min</span><span id="clock"></span></div>${ps}`;
  },
  tickClock(now) {
    const el = document.getElementById('clock'); if (!el) return;
    const sec = this.talkStart ? Math.floor((now - this.talkStart) / 1000) : 0;
    const plan = this.scenes.slice(0, this.idx).reduce((a, s) => a + (s.min || 1.5), 0);
    el.textContent = `talk ${Math.floor(sec / 60)}:${String(sec % 60).padStart(2, '0')} · plan ${plan.toFixed(0)} min`;
  },
};

/* ---------------------------------------------------------------- HTML snippets used by every scene */
const hdr = (eb, h, b = 0, w = 1760) => `<div class="abs" style="left:96px;top:66px;width:${w}px"><div class="eyebrow rise" data-b="${b}">${eb}</div>${words(h, 'h2', b, 'style="margin-top:16px"')}</div>`;
const sub = (s, b = 0, x) => `<div class="sub-wrap"><div class="sub rise" data-b="${b}"${x !== undefined ? ` data-x="${x}"` : ''}>${s}</div></div>`;
const at = (x, y, inner, b = 0, cls = 'rise', extra = '', xb) => `<div class="abs ${cls}" data-b="${b}"${xb !== undefined ? ` data-x="${xb}"` : ''} style="left:${x}px;top:${y}px;${extra}">${inner}</div>`;
const think = (x, y, q, a, b = 0, ab = b + 1) => `<div class="think pop" data-b="${b}" style="left:${x}px;top:${y}px"><div class="lbl">Pause and think</div><div class="q">${q}</div>${a ? `<div class="a rise" data-b="${ab}">${a}</div>` : ''}</div>`;
const slider = (id, label, min, max, step, val, extra = '') => `<label class="ctl ui" for="${id}"><span>${label}</span><input id="${id}" type="range" min="${min}" max="${max}" step="${step}" value="${val}" ${extra}><output id="${id}-o"></output></label>`;

/* marching squares: segments where f = level over a plot, on an nx x ny grid */
function contourSegs(P, f, level = 0.5, nx = 80, ny = 50) {
  const xs = linspace(P.xr[0], P.xr[1], nx), ys = linspace(P.yr[0], P.yr[1], ny), v = xs.map(x => ys.map(y => f(x, y) - level)), out = [];
  const interp = (x1, y1, a, x2, y2, b) => { const t = a / (a - b); return [P.X(x1 + (x2 - x1) * t), P.Y(y1 + (y2 - y1) * t)]; };
  for (let i = 0; i < nx - 1; i++) for (let j = 0; j < ny - 1; j++) {
    const a = v[i][j], b = v[i + 1][j], c = v[i + 1][j + 1], d = v[i][j + 1], pts = [];
    const x0 = xs[i], x1 = xs[i + 1], y0 = ys[j], y1 = ys[j + 1];
    if ((a > 0) !== (b > 0)) pts.push(interp(x0, y0, a, x1, y0, b));
    if ((b > 0) !== (c > 0)) pts.push(interp(x1, y0, b, x1, y1, c));
    if ((c > 0) !== (d > 0)) pts.push(interp(x1, y1, c, x0, y1, d));
    if ((d > 0) !== (a > 0)) pts.push(interp(x0, y1, d, x0, y0, a));
    if (pts.length >= 2) out.push([pts[0], pts[1]]);
    if (pts.length === 4) out.push([pts[2], pts[3]]);
  }
  return out;
}
function drawSegs(segs, col, w = 3, a = 1, glow = 0) {
  const g = D.g; g.save(); g.globalAlpha *= a; g.strokeStyle = col; g.lineWidth = w; g.lineCap = 'round';
  if (glow) { g.shadowColor = col; g.shadowBlur = glow; }
  g.beginPath(); for (const [p, q] of segs) { g.moveTo(p[0], p[1]); g.lineTo(q[0], q[1]); } g.stroke(); g.restore();
}
/* bind a range input to a callback; returns a setter */
function bindSlider(root, id, fmtf, cb) {
  const el = root.querySelector('#' + id), out = root.querySelector('#' + id + '-o');
  const upd = () => { const v = +el.value; if (out) out.textContent = fmtf ? fmtf(v) : v; cb && cb(v); };
  el.addEventListener('input', upd); upd();
  return v => { el.value = v; upd(); };
}
