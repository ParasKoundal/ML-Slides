/* lab.js: every data set and model in the lecture, computed in the browser.
 *
 * All data are SIMULATED with fixed seeds (labelled as such on screen), so the
 * deck shows the same numbers every time. Nothing here is hand-drawn: the
 * lines, surfaces, gradients and trained networks are the real algorithms.
 */
'use strict';

/* ---------------------------------------------------------------- random numbers */
function RNG(seed) {
  let a = seed >>> 0;
  const r = () => { a |= 0; a = a + 0x6D2B79F5 | 0; let t = Math.imul(a ^ a >>> 15, 1 | a); t = t + Math.imul(t ^ t >>> 7, 61 | t) ^ t; return ((t ^ t >>> 14) >>> 0) / 4294967296; };
  r.n = () => { let u = 0, v = 0; while (u === 0) u = r(); v = r(); return Math.sqrt(-2 * Math.log(u)) * Math.cos(TAU * v); };
  r.u = (lo, hi) => lo + (hi - lo) * r();
  return r;
}
const sigmoid = z => 1 / (1 + Math.exp(-z));

/* ---------------------------------------------------------------- linear algebra */
function solve(A, b) {               // Gaussian elimination with partial pivoting
  const n = b.length, M = A.map((r, i) => [...r, b[i]]);
  for (let c = 0; c < n; c++) {
    let p = c; for (let r = c + 1; r < n; r++) if (Math.abs(M[r][c]) > Math.abs(M[p][c])) p = r;
    [M[c], M[p]] = [M[p], M[c]];
    for (let r = c + 1; r < n; r++) { const f = M[r][c] / M[c][c]; for (let k = c; k <= n; k++) M[r][k] -= f * M[c][k]; }
  }
  const x = Array(n).fill(0);
  for (let r = n - 1; r >= 0; r--) { let s = M[r][n]; for (let k = r + 1; k < n; k++) s -= M[r][k] * x[k]; x[r] = s / M[r][r]; }
  return x;
}
function lstsq(X, y, ridge = 0) {    // normal equations (X^T X + ridge I) w = X^T y
  const p = X[0].length, A = Array.from({ length: p }, () => Array(p).fill(0)), b = Array(p).fill(0);
  X.forEach((row, i) => { for (let j = 0; j < p; j++) { b[j] += row[j] * y[i]; for (let k = 0; k < p; k++) A[j][k] += row[j] * row[k]; } });
  for (let j = 0; j < p; j++) A[j][j] += ridge;
  return solve(A, b);
}

/* ================================================================ 1. feed rate -> titre (linear regression) */
const FEED = (() => {
  const r = RNG(7), n = 14, x = [], y = [];
  for (let i = 0; i < n; i++) { const xi = 0.4 + 3.2 * (i + r()) / n; x.push(xi); y.push(1.2 + 1.35 * xi + 0.42 * r.n()); }
  const xm = mean(x), ym = mean(y);
  const cov = mean(x.map((xi, i) => (xi - xm) * (y[i] - ym))), vx = mean(x.map(xi => (xi - xm) ** 2));
  const w = cov / vx, b = ym - w * xm;
  const L = (w_, b_) => mean(x.map((xi, i) => (y[i] - (w_ * xi + b_)) ** 2));
  const grad = (w_, b_) => {
    let gw = 0, gb = 0;
    x.forEach((xi, i) => { const r_ = y[i] - (w_ * xi + b_); gw += -2 * xi * r_; gb += -2 * r_; });
    return [gw / n, gb / n];
  };
  const Exx = mean(x.map(v => v * v));
  // Hessian of L: 2 [[E x^2, E x],[E x, 1]]
  const Hs = [[2 * Exx, 2 * xm], [2 * xm, 2]];
  const tr = Hs[0][0] + Hs[1][1], det = Hs[0][0] * Hs[1][1] - Hs[0][1] ** 2;
  const lmax = tr / 2 + Math.sqrt(tr * tr / 4 - det), lmin = tr / 2 - Math.sqrt(tr * tr / 4 - det);
  return { x, y, n, w, b, L, grad, xm, ym, cov, vx, Lmin: L(w, b), lmax, lmin, start: [-0.2, 5.2] };
})();
function gdPath(gradf, w0, lr, steps) {
  const path = [w0.slice()]; let w = w0.slice();
  for (let k = 0; k < steps; k++) { const g = gradf(...w); w = w.map((v, i) => v - lr * g[i]); path.push(w.slice()); }
  return path;
}

/* ================================================================ 2. Arrhenius kinetics */
const ARR = (() => {
  const r = RNG(11), R = 8.314, Ea = 55000, A = 1.0e7;
  const T = linspace(300, 370, 8), k = T.map(t => A * Math.exp(-Ea / (R * t)) * Math.exp(0.06 * r.n()));
  const fit = (idx) => {
    const xs = idx.map(i => 1000 / T[i]), ys = idx.map(i => Math.log(k[i]));
    const xm = mean(xs), ym = mean(ys);
    const w = sum(xs.map((v, i) => (v - xm) * (ys[i] - ym))) / sum(xs.map(v => (v - xm) ** 2));
    return { w, b: ym - w * xm };
  };
  const all = fit(T.map((_, i) => i));
  const boots = [];
  const rb = RNG(5);
  for (let s = 0; s < 60; s++) { const idx = T.map(() => Math.floor(rb() * T.length)); if (new Set(idx).size > 2) boots.push(fit(idx)); }
  const EaOf = f => -f.w * 1000 * R / 1000;     // kJ/mol
  const eas = boots.map(EaOf).sort((a, b) => a - b);
  return { T, k, R, Ea, A, fit: all, Ea_fit: EaOf(all), A_fit: Math.exp(all.b), boots, ci: [eas[Math.floor(eas.length * .025)], eas[Math.floor(eas.length * .975)]] };
})();

/* ================================================================ 3. conditioning: raw units vs standardised */
const COND = (() => {
  // Hessian eigenvalues follow the feature variances: T in degC (sd 2.5), pH (sd 0.2)
  const raw = [2 * 2.5 ** 2, 2 * 0.2 ** 2], std = [2, 2 * 0.8];
  const make = (lam, rot) => {
    const c = Math.cos(rot), s = Math.sin(rot);
    const L = (u, v) => { const a = c * u + s * v, b = -s * u + c * v; return 0.5 * (lam[0] * a * a + lam[1] * b * b); };
    const G = (u, v) => { const a = c * u + s * v, b = -s * u + c * v, ga = lam[0] * a, gb = lam[1] * b; return [c * ga - s * gb, s * ga + c * gb]; };
    return { L, G, lam, kappa: lam[0] / lam[1] };
  };
  return { raw: make(raw, 0.0), std: make(std, 0.5) };
})();

/* ================================================================ 4. a bumpy landscape (non-convex) */
const BUMPY = {
  f: (u, v) => 0.55 * (u * u + v * v) + 0.22 * Math.sin(3.2 * u + 0.5) * Math.cos(2.8 * v - 0.3) + 0.12 * Math.sin(5 * u * v),
  g(u, v) { const h = 1e-4; return [(this.f(u + h, v) - this.f(u - h, v)) / (2 * h), (this.f(u, v + h) - this.f(u, v - h)) / (2 * h)]; },
};

/* ================================================================ 5. lactate -> pass / fail (1-D logistic) */
const LAC = (() => {
  const r = RNG(21), n = 30, x = [], y = [];
  for (let i = 0; i < n; i++) { const xi = r.u(0.6, 5.0); x.push(xi); y.push(r() < sigmoid(2.4 * (xi - 2.8)) ? 1 : 0); }
  let w = 0, b = 0;
  for (let it = 0; it < 4000; it++) {
    let gw = 0, gb = 0; x.forEach((xi, i) => { const p = sigmoid(w * xi + b); gw += (p - y[i]) * xi; gb += p - y[i]; });
    w -= 0.3 * gw / n; b -= 0.3 * gb / n;
  }
  const ce = (w_, b_) => -mean(x.map((xi, i) => { const p = clamp(sigmoid(w_ * xi + b_), 1e-9, 1 - 1e-9); return y[i] ? Math.log(p) : Math.log(1 - p); }));
  // least-squares line on the 0/1 labels, for the "a line predicts 130 %" moment
  const xm = mean(x), ym = mean(y), lw = sum(x.map((v, i) => (v - xm) * (y[i] - ym))) / sum(x.map(v => (v - xm) ** 2));
  return { x, y, n, w, b, ce, line: { w: lw, b: ym - lw * xm }, thresh: -b / w };
})();

/* ================================================================ 6. runaway risk from T and dissolved oxygen (2-D logistic) */
const RUN = (() => {
  const r = RNG(3), n = 120, X = [], y = [];
  for (let i = 0; i < n; i++) {
    const T = r.u(30, 40), DO = r.u(10, 60);
    const z = 0.9 * (T - 35) - 0.12 * (DO - 35);
    X.push([T, DO]); y.push(r() < sigmoid(2.2 * z) ? 1 : 0);
  }
  const mu = [35, 35], sd = [2.9, 14.4];
  const Z = X.map(([a, b]) => [(a - mu[0]) / sd[0], (b - mu[1]) / sd[1]]);
  return { X, Z, y, n, mu, sd };
})();

/* ================================================================ 7. the operating window (T, pH) */
const WIN = (() => {
  const r = RNG(42), n = 240, X = [], y = [];
  for (let i = 0; i < n; i++) {
    const T = r.u(30, 38), pH = r.u(6.6, 7.4);
    const d = ((T - 34.2) / 2.0) ** 2 + ((pH - 7.02) / 0.19) ** 2;
    X.push([T, pH]); y.push(d < 1 ? 1 : 0);
  }
  const mu = [34, 7.0], sd = [2.3, 0.23];
  const Z = X.map(([a, b]) => [(a - mu[0]) / sd[0], (b - mu[1]) / sd[1]]);
  return { X, Z, y, n, mu, sd, z: ([a, b]) => [(a - mu[0]) / sd[0], (b - mu[1]) / sd[1]] };
})();

/* ================================================================ a small multilayer perceptron, trained by hand-written backprop */
class MLP {
  /* sizes e.g. [2,16,1]; act 'relu' | 'tanh'; out 'sigmoid' (cross-entropy) | 'linear' (squared error) */
  constructor(sizes, act = 'relu', out = 'sigmoid', seed = 1, scale = 1) {
    const r = RNG(seed); this.sizes = sizes; this.act = act; this.out = out;
    this.W = []; this.b = [];
    for (let l = 0; l < sizes.length - 1; l++) {
      const fi = sizes[l], fo = sizes[l + 1], s = scale * Math.sqrt(2 / fi);
      this.W.push(Array.from({ length: fo }, () => Array.from({ length: fi }, () => r.n() * s)));
      this.b.push(Array.from({ length: fo }, () => (l === 0 ? r.n() * 0.5 : 0)));
    }
    this.mW = this.W.map(L => L.map(row => row.map(() => 0))); this.mb = this.b.map(v => v.map(() => 0));
    this.epoch = 0;
  }
  f(z) { return this.act === 'relu' ? Math.max(0, z) : Math.tanh(z); }
  df(z, a) { return this.act === 'relu' ? (z > 0 ? 1 : 0) : 1 - a * a; }
  forward(x) {
    const zs = [], as = [x];
    let a = x;
    for (let l = 0; l < this.W.length; l++) {
      const z = this.W[l].map((row, j) => this.b[l][j] + row.reduce((s, w, k) => s + w * a[k], 0));
      const last = l === this.W.length - 1;
      a = last ? (this.out === 'sigmoid' ? z.map(sigmoid) : z) : z.map(v => this.f(v));
      zs.push(z); as.push(a);
    }
    return { zs, as, y: a[0] };
  }
  predict(x) { return this.forward(x).y; }
  /* one full-batch gradient step; returns the loss before the step */
  step(X, Y, lr, mom = 0) {
    const gW = this.W.map(L => L.map(row => row.map(() => 0))), gb = this.b.map(v => v.map(() => 0));
    let loss = 0; const n = X.length;
    for (let i = 0; i < n; i++) {
      const { zs, as, y } = this.forward(X[i]);
      if (this.out === 'sigmoid') { const p = clamp(y, 1e-9, 1 - 1e-9); loss -= Y[i] ? Math.log(p) : Math.log(1 - p); }
      else loss += (y - Y[i]) ** 2;
      let delta = [this.out === 'sigmoid' ? y - Y[i] : 2 * (y - Y[i])];      // dL/dz at the output
      for (let l = this.W.length - 1; l >= 0; l--) {
        const aPrev = as[l];
        for (let j = 0; j < delta.length; j++) { gb[l][j] += delta[j]; for (let k = 0; k < aPrev.length; k++) gW[l][j][k] += delta[j] * aPrev[k]; }
        if (l > 0) {
          const nd = aPrev.map((_, k) => { let s = 0; for (let j = 0; j < delta.length; j++) s += this.W[l][j][k] * delta[j]; return s * this.df(zs[l - 1][k], aPrev[k]); });
          delta = nd;
        }
      }
    }
    for (let l = 0; l < this.W.length; l++) {
      for (let j = 0; j < this.W[l].length; j++) {
        for (let k = 0; k < this.W[l][j].length; k++) { this.mW[l][j][k] = mom * this.mW[l][j][k] + gW[l][j][k] / n; this.W[l][j][k] -= lr * this.mW[l][j][k]; }
        this.mb[l][j] = mom * this.mb[l][j] + gb[l][j] / n; this.b[l][j] -= lr * this.mb[l][j];
      }
    }
    this.epoch++;
    return loss / n;
  }
  loss(X, Y) {
    let s = 0;
    X.forEach((x, i) => { const y = this.predict(x); if (this.out === 'sigmoid') { const p = clamp(y, 1e-9, 1 - 1e-9); s -= Y[i] ? Math.log(p) : Math.log(1 - p); } else s += (y - Y[i]) ** 2; });
    return s / X.length;
  }
  acc(X, Y) { return mean(X.map((x, i) => ((this.predict(x) > .5 ? 1 : 0) === Y[i] ? 1 : 0))); }
  get nParams() { return this.W.reduce((s, L, l) => s + L.length * L[0].length + this.b[l].length, 0); }
}

/* ================================================================ 8. Haldane substrate inhibition kinetics */
const HAL = (() => {
  const mu = S => 0.45 * S / (1.0 + S + S * S / 6.0);
  const r = RNG(9), S = linspace(0.2, 20, 26), Y = S.map(s => mu(s) + 0.011 * r.n());
  // "a network with a frozen first layer": fixed ReLU hinges, output weights by least squares
  const hinges = (k, lo = 0, hi = 20) => linspace(lo, hi, k + 2).slice(0, k);
  const feat = (s, knots) => [1, ...knots.map(c => Math.max(0, s - c))];
  const fitRelu = (k, xs = S, ys = Y) => {
    const knots = hinges(k);
    const c = lstsq(xs.map(s => feat(s, knots)), ys, 1e-6);
    return { knots, c, f: s => feat(s, knots).reduce((a, v, i) => a + v * c[i], 0) };
  };
  // overfitting: polynomial degree d in a Chebyshev basis on [0,20]
  const ro = RNG(31), Str = linspace(0.5, 19.5, 10).map(s => s + ro.u(-.4, .4)), Ytr = Str.map(s => mu(s) + 0.02 * ro.n());
  const Sva = linspace(0.3, 19.7, 200), Yva = Sva.map(s => mu(s) + 0.02 * ro.n());
  const cheb = (s, d) => { const t = s / 10 - 1, T = [1, t]; for (let k = 2; k <= d; k++) T.push(2 * t * T[k - 1] - T[k - 2]); return T.slice(0, d + 1); };
  const poly = d => {
    const c = lstsq(Str.map(s => cheb(s, d)), Ytr, 1e-9);
    const f = s => cheb(s, d).reduce((a, v, i) => a + v * c[i], 0);
    const tr = mean(Str.map((s, i) => (f(s) - Ytr[i]) ** 2)), va = mean(Sva.map((s, i) => (f(s) - Yva[i]) ** 2));
    return { d, f, tr, va };
  };
  const polys = Array.from({ length: 10 }, (_, d) => (d === 0 ? null : poly(d)));
  return { mu, S, Y, fitRelu, Str, Ytr, Sva, Yva, polys };
})();

/* ================================================================ 9. a 2-3-1 tanh network whose hidden space we can look at in 3-D */
const WARP = (() => {
  let best = null;
  for (const seed of [3, 5, 8, 13, 21]) {
    const net = new MLP([2, 3, 1], 'tanh', 'sigmoid', seed, 1.2);
    for (let e = 0; e < 1500; e++) net.step(WIN.Z, WIN.y, 0.8, 0.9);
    const a = net.acc(WIN.Z, WIN.y);
    if (!best || a > best.acc) best = { net, acc: a };
  }
  return best;
})();

/* ================================================================ 10. the backprop example worked by hand */
const BP = (() => {
  const x = 2, y = 3, w1 = 0.5, w2 = 1.5, lr = 0.05;
  const z = w1 * x, h = Math.max(0, z), yh = w2 * h, L = (yh - y) ** 2;
  const dL_dyh = 2 * (yh - y), dL_dw2 = dL_dyh * h, dL_dh = dL_dyh * w2, dh_dz = z > 0 ? 1 : 0, dL_dz = dL_dh * dh_dz, dL_dw1 = dL_dz * x;
  const w1n = w1 - lr * dL_dw1, w2n = w2 - lr * dL_dw2, hn = Math.max(0, w1n * x), yhn = w2n * hn, Ln = (yhn - y) ** 2;
  return { x, y, w1, w2, lr, z, h, yh, L, dL_dyh, dL_dw2, dL_dh, dh_dz, dL_dz, dL_dw1, w1n, w2n, hn, yhn, Ln };
})();
