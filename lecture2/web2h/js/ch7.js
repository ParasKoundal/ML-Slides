/* ch7.js: Gradients for engineers. The same machinery pointed at engineering problems: fitting rate constants
 * of A -> B -> C by gradient descent; a network surrogate of a reactor and gradient ascent on its inputs;
 * a physics-informed loss; sensitivities as gradients with respect to the inputs. */
'use strict';

Deck.chapter('Gradients for engineers', 'kinetics, surrogates and physics');

/* ================================================================ series reaction A -> B -> C, first order */
const ABC = (() => {
  const conc = (k1, k2, t) => {
    const A = Math.exp(-k1 * t);
    const B = Math.abs(k2 - k1) < 1e-6 ? k1 * t * Math.exp(-k1 * t) : k1 / (k2 - k1) * (Math.exp(-k1 * t) - Math.exp(-k2 * t));
    return [A, B, 1 - A - B];
  };
  const r = RNG(71), K = [0.45, 0.12], t = linspace(1, 20, 11);
  const data = t.map(tt => conc(...K, tt).map(c => c + 0.02 * r.n()));
  const L = (l1, l2) => { const k1 = Math.exp(l1), k2 = Math.exp(l2); return mean(t.flatMap((tt, i) => conc(k1, k2, tt).map((c, s) => (c - data[i][s]) ** 2))); };
  const G = (l1, l2) => { const h = 1e-5; return [(L(l1 + h, l2) - L(l1 - h, l2)) / (2 * h), (L(l1, l2 + h) - L(l1, l2 - h)) / (2 * h)]; };
  // Adam on (ln k1, ln k2), from a poor first guess
  let p = [Math.log(0.06), Math.log(0.9)]; const path = [p.slice()], m = [0, 0], v = [0, 0];
  for (let s = 1; s <= 320; s++) {
    const g = G(...p);
    for (let j = 0; j < 2; j++) { m[j] = .9 * m[j] + .1 * g[j]; v[j] = .999 * v[j] + .001 * g[j] * g[j]; p[j] -= 0.06 * (m[j] / (1 - .9 ** s)) / (Math.sqrt(v[j] / (1 - .999 ** s)) + 1e-8); }
    path.push(p.slice());
  }
  return { conc, K, t, data, L, path, fit: path[path.length - 1].map(Math.exp) };
})();
const SPC = [['A', '#3f8cff'], ['B', COL.data], ['C', COL.grad]];

chapterCard(7, 'Gradients for engineers', 'kinetics, surrogates and physics',
  'Chapter seven. Everything so far has been about training models on data. But gradient descent and backpropagation do not care what the knobs mean. In this chapter we point the same machinery at engineering problems: fitting rate constants, optimising a process through a learned model, and putting physics into the loss.',
  (g, S) => {
    const P = new Plot(1180, 250, 600, 260, [0, 20], [0, 1]), q = S.p(2600, 300);
    SPC.forEach(([, col], s) => D.fn(P, t => ABC.conc(...ABC.K, t)[s], 0, 20 * q, col, 4, 1, 120, null, 10));
  });

/* ------------------------------------------------------------ 1. fitting rate constants by gradient descent */
const PAB = new Plot(150, 290, 820, 520, [0, 20], [-0.05, 1.05]), PAK = new Plot(1130, 290, 640, 520, [Math.log(0.03), Math.log(1.5)], [Math.log(0.03), Math.log(1.5)]);
Deck.add({
  id: 'ode-fit', title: 'Fitting rate constants by gradient descent', beats: 3, dur: [5400, 8200, 5600], min: 4.5,
  notes: [
    'A series reaction, A to B to C, both steps first order. B is the product we want. We sampled the reactor eleven times over twenty minutes and measured all three concentrations. The model has two knobs, the rate constants k1 and k2, and our first guess is poor: it has the second step much faster than the first, so the predicted B barely appears.',
    'The loss is the same mean squared error, summed over all three species. We descend on the logarithms of the rate constants, not the constants themselves. That keeps them positive automatically, and because rate constants span orders of magnitude, a step in ln k is a relative change, which conditions the problem much better. On the right is the loss surface over ln k1 and ln k2, and the path the optimiser takes, here Adam. On the left the model curves bend toward the data as it goes.',
    'It lands on k1 equal to ' + ABC.fit[0].toFixed(2) + ' and k2 equal to ' + ABC.fit[1].toFixed(2) + ' per minute; the simulated truth was 0.45 and 0.12. This particular model has a closed-form solution, so this is what curve_fit would also do. The point is what happens when it does not: a mechanism with 30 species and 80 rate constants, solved numerically. You can differentiate straight through the ODE solver, by backpropagation through its time steps or by the adjoint method, and every gradient comes from one backward pass, exactly as in chapter 5. Libraries such as torchdiffeq, JAX with diffrax, and Julia\'s SciML do precisely this.',
  ],
  html: () => `${hdr('Chapter 7 · kinetics · A → B → C', 'Rate constants by gradient descent')}
    ${at(1130, 250, `<div class="lbl">loss over ln k<sub>1</sub>, ln k<sub>2</sub></div>`, 1, 'rise', '', 2)}
    ${at(1130, 300, `<div class="card" style="width:640px"><h4 class="c-good">fitted</h4><p>k<sub>1</sub> = ${ABC.fit[0].toFixed(2)} min⁻¹ &nbsp;·&nbsp; k<sub>2</sub> = ${ABC.fit[1].toFixed(2)} min⁻¹<br>simulated truth: 0.45 and 0.12</p></div>`, 2, 'rise')}
    ${at(1130, 500, '<div class="card" style="width:640px"><h4 class="c-grad">no closed form? no problem</h4><p>differentiate through the ODE solver: backprop through time steps, or the adjoint method. torchdiffeq · diffrax · SciML</p></div>', 2, 'rise', '--d:700ms')}
    ${sub('Two knobs, k₁ and k₂. A poor first guess.', 0, 1)}
    ${sub('Descend on ln k: <b>positive for free, and better conditioned.</b>', 1, 2)}
    ${sub('Same loop, any mechanism: gradients flow through the solver.', 2)}`,
  enter(st) { st.heat = heatImage(PAK, (a, b) => Math.log(ABC.L(a, b)), { res: 5, stops: ['#06131f', '#0b2c45', '#14506a', '#2a7d86', '#5fb3a1'] }); },
  frame(g, S) {
    const a0 = S.at(0, 700), P = PAB;
    P.axes({ xt: [0, 5, 10, 15, 20], yt: [0, 0.5, 1], xl: 'time  (min)', yl: 'concentration / C_A0', a: a0, grid: true });
    const n = S.beat === 0 ? 0 : S.beat === 1 ? Math.floor(Ease.inOut(S.at(1, 7200, 500)) * (ABC.path.length - 1)) : ABC.path.length - 1;
    const [k1, k2] = ABC.path[n].map(Math.exp);
    SPC.forEach(([nm, col], s) => {
      D.fn(P, t => ABC.conc(k1, k2, t)[s], 0, 20, col, 3.5, a0, 120);
      ABC.t.forEach((t, i) => D.dot(P.X(t), P.Y(ABC.data[i][s]), 7, col, a0, 6));
      D.text(nm, P.X(20) + 18, P.Y(ABC.conc(k1, k2, 20)[s]), { size: 26, weight: 600, col, a: a0, font: '"IBM Plex Sans", sans-serif' });
    });
    if (S.beat === 1) {
      const a = S.at(1, 700);
      drawImg(S.st.heat, PAK, .95 * a);
      [0.03, 0.1, 0.3, 1, 1.5].forEach(k => { D.text(String(k), PAK.X(Math.log(k)), PAK.y + PAK.h + 26, { size: 20, align: 'center', a }); D.text(String(k), PAK.x - 14, PAK.Y(Math.log(k)), { size: 20, align: 'right', a }); });
      D.text('k₁  (log scale)', PAK.x + PAK.w, PAK.y + PAK.h + 60, { size: 22, align: 'right', a, font: '"IBM Plex Sans", sans-serif' });
      D.text('k₂', PAK.x - 60, PAK.y - 16, { size: 22, a, font: '"IBM Plex Sans", sans-serif' });
      D.line(ABC.path.slice(0, n + 1).map(([u, v]) => [PAK.X(u), PAK.Y(v)]), '#ffffff', 2, .9 * a);
      D.dot(PAK.X(ABC.path[n][0]), PAK.Y(ABC.path[n][1]), 10, COL.model, a, 18);
      D.dot(PAK.X(Math.log(ABC.K[0])), PAK.Y(Math.log(ABC.K[1])), 7, COL.good, a);
      D.seg(PAK.x, PAK.y + PAK.h, PAK.x + PAK.w, PAK.y, '#ffffff', 1, .25 * a, [5, 7]);
      D.text('k₁ = k₂', PAK.X(Math.log(1.0)), PAK.Y(Math.log(1.0)) + 26, { size: 18, col: '#c8d1de', a: .6 * a, align: 'center' });
    }
  },
});

/* ================================================================ a reactor surrogate: yield of B over (T, tau) */
const YIELD = (() => {
  const R = 8.314, Tr = 350;
  const kk = (T, k0, E) => k0 * Math.exp(-E / R * (1 / T - 1 / Tr));
  const lt = [Math.log(0.5), Math.log(10)];
  const toT = u => 355 + 25 * u, toTau = v => Math.exp(lerp(lt[0], lt[1], (v + 1) / 2));
  // A -> B (wanted, E 80 kJ/mol), A -> D (by-product, 30 kJ/mol, wins when cold), B -> C (130 kJ/mol, wins when hot)
  const true_ = (u, v) => {
    const T = toT(u), tau = toTau(v), k1 = kk(T, 0.5, 80000), k3 = kk(T, 0.25, 30000), k2 = kk(T, 0.15, 130000), ka = k1 + k3;
    return k1 / (k2 - ka) * (Math.exp(-ka * tau) - Math.exp(-k2 * tau));
  };
  const r = RNG(97), n = 70, X = [], Y = [];
  for (let i = 0; i < n; i++) { const u = r.u(-1, 1), v = r.u(-1, 1); X.push([u, v]); Y.push(true_(u, v) + 0.012 * r.n()); }
  const net = new MLP([2, 16, 1], 'tanh', 'linear', 5, 1);
  const marks = [0, 20, 60, 150, 400, 1200, 2500], snaps = [];
  for (let e = 0; e <= 2500; e++) { if (marks.includes(e)) snaps.push({ e, W: net.W.map(L => L.map(rw => rw.slice())), b: net.b.map(q => q.slice()) }); net.step(X, Y, 0.05, 0.9); }
  const evalS = (s, u, v) => { let o = s.b[1][0]; for (let j = 0; j < 16; j++) o += s.W[1][0][j] * Math.tanh(s.W[0][j][0] * u + s.W[0][j][1] * v + s.b[0][j]); return o; };
  const f = (u, v) => net.predict([u, v]);
  const grad = (u, v) => { const h = 1e-4; return [(f(u + h, v) - f(u - h, v)) / (2 * h), (f(u, v + h) - f(u, v - h)) / (2 * h)]; };
  const ascend = (u0, v0, box = true, steps = 120, lr = 0.08) => { let p = [u0, v0]; const path = [p]; for (let s = 0; s < steps; s++) { const g = grad(...p); p = [p[0] + lr * g[0] / Math.max(0.05, Math.hypot(...g)) * Math.min(1, Math.hypot(...g) * 4), p[1] + lr * g[1] / Math.max(0.05, Math.hypot(...g)) * Math.min(1, Math.hypot(...g) * 4)]; if (box) p = p.map(q => clamp(q, -1, 1)); path.push(p); } return path; };
  let best = [0, 0, -1]; for (const u of linspace(-1, 1, 81)) for (const v of linspace(-1, 1, 81)) { const y = true_(u, v); if (y > best[2]) best = [u, v, y]; }
  const rmse = Math.sqrt(mean(linspace(-1, 1, 21).flatMap(u => linspace(-1, 1, 21).map(v => (f(u, v) - true_(u, v)) ** 2))));
  return { true_, X, Y, net, snaps, evalS, f, grad, ascend, best, toT, toTau, lt, rmse };
})();
const PY = new Plot(150, 290, 900, 520, [-1.45, 1.45], [-1.2, 1.2]);
const yAxes = (P, a) => {
  [330, 345, 355, 365, 380].forEach(T => D.text(String(T), P.X((T - 355) / 25), P.y + P.h + 26, { size: 20, align: 'center', a }));
  [0.5, 1, 2, 5, 10].forEach(t => D.text(String(t), P.x - 14, P.Y((Math.log(t) - YIELD.lt[0]) / (YIELD.lt[1] - YIELD.lt[0]) * 2 - 1), { size: 20, align: 'right', a }));
  D.text('temperature  (K)', P.x + P.w, P.y + P.h + 60, { size: 22, align: 'right', a, font: '"IBM Plex Sans", sans-serif' });
  D.text('residence time  τ  (min, log scale)', P.x, P.y - 26, { size: 22, a, font: '"IBM Plex Sans", sans-serif' });
  D.rect(P.X(-1), P.Y(1), P.X(1) - P.X(-1), P.Y(-1) - P.Y(1), null, a, '#ffffff', 1.5);
};
const YSTOPS = ['#0a1830', '#0f3b5c', '#11718a', '#2fae9c', '#a8dc82', '#ffe38a'];
Deck.add({
  id: 'surrogate', title: 'A surrogate model and gradient ascent on the inputs', beats: 3, dur: [5200, 7000, 6400], min: 4.5,
  notes: [
    'A more realistic version of the same chemistry, in a plug-flow reactor where we choose the temperature and the residence time, and we want the most B. Now A also leaks to a by-product D through a low-activation-energy side reaction, which wins when the reactor is cold, and B degrades to C with a high activation energy, which wins when it is hot. So there is a best temperature in the middle. We ran 70 experiments, the dots, scattered over the operating box: 330 to 380 kelvin and half a minute to ten minutes. We fitted a small network, two inputs, 16 tanh neurons, one output, to the measured yields. The colours are the network\'s predicted yield. That trained network is a surrogate: a cheap, smooth, differentiable stand-in for the plant or for an expensive simulator.',
    'Now the trick. Freeze the network\'s weights and treat the inputs as the knobs. Backpropagation gives the gradient of predicted yield with respect to temperature and residence time, and we climb it: gradient ascent on the inputs, the mirror image of training. From three different starting conditions the paths climb the ridge and meet near the best point the network knows. The green ring is the true optimum of the simulated reactor, and the network found it to within a few kelvin. Notice the ridge: hotter needs a shorter residence time, because B is consumed faster.',
    'Now the danger. Start a run in the hot, slow corner and let it climb without the box constraint. It walks out of the region where we have data, and the network is happy to keep promising more yield out there: it is extrapolating, which we saw in chapter 6 it does badly. Two rules for surrogate optimisation. Keep the search inside the data, with bounds or a penalty for distance from the data. And never trust the optimum until you have run a confirmation experiment there, added it to the data, and refitted. That loop is the core of Bayesian optimisation and of self-driving labs.',
  ],
  html: () => `${hdr('Chapter 7 · surrogates', 'Freeze the weights, climb the inputs')}
    ${at(1180, 300, `<div class="lbl">surrogate</div><div class="lede" style="font-size:27px;margin-top:10px;width:600px">2 → 16 tanh → 1, fitted to 70 runs<br>error vs the true reactor: ${(YIELD.rmse * 100).toFixed(1)} % yield (RMS)</div>`, 0, 'rise')}
    ${at(1180, 500, `<div class="eq-s">${M(T`\mathbf{x} \leftarrow \mathbf{x} + \eta\, \nabla_{\!\mathbf{x}}\, \hat y(\mathbf{x})`)}</div><div class="lede" style="font-size:27px;margin-top:12px;width:600px">same backprop, but the gradient is taken with respect to the inputs</div>`, 1, 'rise')}
    ${at(1180, 720, `<div class="card" style="width:600px"><h4 class="c-good">true optimum</h4><p>${YIELD.toT(YIELD.best[0]).toFixed(0)} K · τ = ${YIELD.toTau(YIELD.best[1]).toFixed(1)} min · yield ${(YIELD.best[2] * 100).toFixed(0)} %</p></div>`, 1, 'rise', '--d:5200ms', 2)}
    ${at(1180, 720, '<div class="card" style="width:600px"><h4 class="c-loss">outside the box there is no data</h4><p>bound the search, then confirm the optimum with a real run</p></div>', 2, 'rise', '--d:3000ms')}
    ${sub('70 runs, a small network: a differentiable stand-in for the reactor.', 0, 1)}
    ${sub('Gradient ascent on T and τ: <b>three starts climb the same ridge.</b>', 1, 2)}
    ${sub('Unbounded, it walks off the data and believes its own extrapolation.', 2)}`,
  enter(st) {
    st.heat = heatImage(PY, (u, v) => YIELD.f(u, v), { res: 5, lo: 0, hi: 0.5, stops: YSTOPS });
    st.runs = [[-0.9, -0.85], [0.9, -0.9], [-0.6, 0.85]].map(([u, v]) => YIELD.ascend(u, v));
    st.wild = YIELD.ascend(0.9, 0.9, false, 160);
  },
  frame(g, S) {
    const a0 = S.at(0, 700), st = S.st;
    drawImg(st.heat, PY, a0); yAxes(PY, a0);
    YIELD.X.forEach(([u, v]) => D.dot(PY.X(u), PY.Y(v), 5, '#ffffff', .85 * a0));
    if (S.beat >= 1) {
      const n = S.beat === 1 ? Math.floor(S.at(1, 4800, 400) * 120) : 120;
      st.runs.forEach(p => { D.line(p.slice(0, n + 1).map(([u, v]) => [PY.X(u), PY.Y(v)]), COL.model, 3, .95); const [u, v] = p[n]; D.dot(PY.X(u), PY.Y(v), 10, COL.model, 1, 16); });
      D.ring(PY.X(YIELD.best[0]), PY.Y(YIELD.best[1]), 18, COL.good, 3, S.at(1, 600, 5000));
    }
    if (S.beat === 2) {
      const n = Math.floor(S.at(2, 4200, 500) * 160), pts = st.wild.slice(0, n + 1).filter(([u, v]) => Math.abs(u) < 1.45 && Math.abs(v) < 1.2);
      D.line(pts.map(([u, v]) => [PY.X(u), PY.Y(v)]), COL.loss, 3.5);
      if (pts.length) { const [u, v] = pts[pts.length - 1]; D.dot(PY.X(u), PY.Y(v), 11, COL.loss, 1, 18); }
    }
  },
});

/* ------------------------------------------------------------ 3. a physics-informed loss */
const PINN = (() => {
  const k = 0.3, r = RNG(53), td = [0.3, 1.0, 1.8, 2.6, 3.5], Ad = td.map(t => Math.exp(-k * t) + 0.02 * r.n());
  const cs = linspace(-2, 16, 19), wdt = 1.4;
  const phi = t => [1, ...cs.map(c => Math.exp(-(((t - c) / wdt) ** 2)))];
  const dphi = t => [0, ...cs.map(c => -2 * (t - c) / (wdt * wdt) * Math.exp(-(((t - c) / wdt) ** 2)))];
  const tc = linspace(0, 15, 40);
  const fit = lam => {
    const rows = [], ys = [];
    td.forEach((t, i) => { rows.push(phi(t)); ys.push(Ad[i]); });
    const s = Math.sqrt(lam);
    tc.forEach(t => { const a = phi(t), d = dphi(t); rows.push(d.map((v, j) => s * (v + k * a[j]))); ys.push(0); });
    const c = lstsq(rows, ys, 1e-4);
    return t => phi(t).reduce((a, v, j) => a + v * c[j], 0);
  };
  const lams = [0, ...linspace(-4, 0, 30).map(e => 10 ** e)], fits = lams.map(fit);
  return { k, td, Ad, tc, lams, fits };
})();
Deck.add({
  id: 'pinn', title: 'Put the physics in the loss', beats: 3, dur: [5000, 6600, 5600], min: 4,
  notes: [
    'Data are expensive; physics is free. Here are five early samples of a reactant that we know decays by first-order kinetics with k equal to 0.3 per hour, and we want the whole curve out to 15 hours. A flexible model, 20 knobs (a sum of smooth bumps), fitted to the five points alone, goes through them and then does whatever it likes where there is no data: here it dips, then levels off at about a quarter, as if the reaction had simply stopped. Nothing in the loss tells it otherwise.',
    'Add a second term to the loss. At 40 collocation times spread over the whole range, the small ticks, compute how badly the model breaks the rate law: the residual dA/dt plus k A, which should be zero. Square it, average it, and add it to the data loss with a weight lambda. As lambda grows the curve snaps onto the exponential decay, even where we have no data at all, and it still passes near the measurements. The dashed line is the truth.',
    'This is the idea behind physics-informed neural networks, PINNs. The model is usually a network, and dA/dt comes from automatic differentiation, the same backward pass we have been using, with respect to the input t. You can also make k a knob and learn it at the same time, or use conservation laws, energy balances, or monotonicity as the extra terms. The engineering judgement is in choosing which physics you trust enough to put in the loss, and how hard to weight it.',
  ],
  html: () => `${hdr('Chapter 7 · physics-informed', 'Put the rate law in the loss')}
    ${at(1220, 300, `<div class="eq-s">${M(T`L = \underbrace{\tfrac1n\textstyle\sum (\hat A - A)^2}_{\text{data}}`)}</div>`, 0, 'rise')}
    ${at(1220, 430, `<div class="eq-s">${M(T`\;+\;\color{#b48cff}{\lambda\,\tfrac1m\textstyle\sum \Big(\tfrac{d\hat A}{dt} + k\hat A\Big)^2}`)}</div><div class="lbl c-grad" style="margin-top:6px">physics residual at collocation times</div>`, 1, 'rise')}
    ${at(1220, 640, '<div class="card" style="width:560px"><h4 class="c-grad">PINN</h4><p>a network for Â(t), autodiff for dÂ/dt, and k can be a knob too</p></div>', 2, 'rise')}
    ${sub('Five points, twenty knobs: it fits the data and then wanders.', 0, 1)}
    ${sub('Penalise breaking the rate law: <b>the curve snaps to the physics.</b>', 1, 2)}
    ${sub('Physics-informed: data where you have it, the rate law everywhere.', 2)}`,
  frame(g, S) {
    const P = new Plot(150, 290, 960, 520, [0, 15], [-0.4, 1.1]), a0 = S.at(0, 700);
    P.axes({ xt: [0, 5, 10, 15], yt: [0, 0.5, 1], xl: 'time  (h)', yl: 'C_A / C_A0', a: a0, grid: true });
    D.seg(P.x, P.Y(0), P.x + P.w, P.Y(0), '#3a4352', 1.5, a0);
    D.fn(P, t => Math.exp(-PINN.k * t), 0, 15, '#c8d1de', 2, .6 * a0, 100, [8, 8]);
    const q = S.beat === 0 ? 0 : S.beat === 1 ? Ease.inOut(S.at(1, 4400, 800)) : 1;
    const x = q * (PINN.fits.length - 1), k = Math.min(PINN.fits.length - 2, Math.floor(x)), f = x - k;
    const col = mixHex(COL.loss, COL.grad, q);
    D.fn(P, t => lerp(PINN.fits[k](t), PINN.fits[k + 1](t), f), 0, 15, col, 4, S.at(0, 1200, 500), 200, null, 10);
    PINN.td.forEach((t, i) => D.dot(P.X(t), P.Y(PINN.Ad[i]), 9, COL.data, a0, 10));
    if (S.beat >= 1) PINN.tc.forEach(t => D.seg(P.X(t), P.y + P.h - 10, P.X(t), P.y + P.h, COL.grad, 2, S.at(1, 600)));
    if (S.beat >= 1) D.text('λ = ' + (q < 0.04 ? '0' : sci(PINN.lams[Math.round(x)] || 10, 0)), P.x + P.w - 10, P.y + 26, { size: 28, weight: 600, col: COL.grad, align: 'right', a: S.at(1, 600) });
  },
});

/* ------------------------------------------------------------ 4. sensitivities are input gradients */
const SENS = (() => {
  const op = [0.2, -0.25];   // the operating point, in scaled units
  const T = YIELD.toT(op[0]), tau = YIELD.toTau(op[1]);
  const g = YIELD.grad(...op);                 // per scaled unit
  const h = 1e-4, gt = [(YIELD.true_(op[0] + h, op[1]) - YIELD.true_(op[0] - h, op[1])) / (2 * h), (YIELD.true_(op[0], op[1] + h) - YIELD.true_(op[0], op[1] - h)) / (2 * h)];
  const dv = 2 / (YIELD.lt[1] - YIELD.lt[0]);  // scaled v per unit ln(tau)
  // the effect of realistic disturbances: +-2 K on T, +-10 % on tau, in % yield
  const eff = gg => [gg[0] / 25 * 2 * 100, gg[1] * dv * Math.log(1.1) * 100];
  return { op, T, tau, g, gt, net: eff(g), tru: eff(gt), y: YIELD.f(...op) };
})();
Deck.add({
  id: 'sensitivity', title: 'Sensitivities are gradients', beats: 3, dur: [5200, 5800, 5400], min: 3.5,
  notes: [
    'One more use of the same gradient. The arrows are the gradient of the surrogate\'s predicted yield with respect to temperature and residence time, at a grid of operating points. Each arrow points uphill, and its length says how sensitive the yield is there. Along the ridge they are short: the yield is forgiving. On the flanks they are long: small changes cost a lot.',
    'Pick the operating point, ' + SENS.T.toFixed(0) + ' kelvin and ' + SENS.tau.toFixed(1) + ' minutes. Multiply each sensitivity by a realistic disturbance: plus or minus two kelvin of temperature control, and plus or minus ten percent on residence time from flow variation. The bars show the change in yield each causes, from the network in cyan and from the true reactor model in green. They agree, and they say ' + (Math.abs(SENS.net[0]) > Math.abs(SENS.net[1]) ? 'temperature' : 'residence time') + ' is the one to control tightly here.',
    'This is local sensitivity analysis, and with backpropagation it is one backward pass for every input at once, whether there are two inputs or two thousand. That is the same reason backprop is cheap for training. Two cautions. A surrogate\'s slope is only as good as its data: check it against the mechanism or against plant trials where you can. And a gradient is local: for big moves, use the model itself, not its slope.',
  ],
  html: () => `${hdr('Chapter 7 · sensitivity', 'Which input matters? Ask the gradient')}
    ${at(1180, 290, `<div class="lbl">operating point</div><div class="big" style="margin-top:6px">${SENS.T.toFixed(0)} K · τ = ${SENS.tau.toFixed(1)} min · yield ${(SENS.y * 100).toFixed(0)} %</div>`, 1, 'rise')}
    ${at(1180, 420, '<div class="lbl">change in yield, % points</div>', 1, 'rise', '--d:400ms')}
    ${at(1180, 500, '<div class="cap">±2 K</div>', 1, 'rise', '--d:600ms')}
    ${at(1180, 600, '<div class="cap">±10 % τ</div>', 1, 'rise', '--d:600ms')}
    ${at(1180, 700, '<div class="lbl"><span class="c-model">■ surrogate network</span> &nbsp; <span class="c-good">■ true reactor</span></div>', 1, 'rise', '--d:900ms')}
    ${at(1180, 770, `<div class="eq-xs">${M(T`\frac{\partial \hat y}{\partial T},\ \frac{\partial \hat y}{\partial \tau}:\ \text{one backward pass}`)}</div>`, 2, 'rise')}
    ${sub('Arrows: the gradient of yield with respect to T and τ.', 0, 1)}
    ${sub('Scale by realistic disturbances: <b>this is what to control tightly.</b>', 1, 2)}
    ${sub('Every input\'s sensitivity in one backward pass. Check it against the mechanism.', 2)}`,
  enter(st) { st.heat = heatImage(PY, (u, v) => YIELD.f(u, v), { res: 5, lo: 0, hi: 0.5, stops: YSTOPS }); },
  frame(g, S) {
    const a0 = S.at(0, 700);
    drawImg(S.st.heat, PY, .85 * a0); yAxes(PY, a0);
    const ar = S.at(0, 1600, 600);
    for (const u of linspace(-0.9, 0.9, 10)) for (const v of linspace(-0.9, 0.9, 8)) {
      const [gu, gv] = YIELD.grad(u, v), sx = PY.w / 2.9, sy = PY.h / 2.4;
      const dx = gu * sx * .35, dy = -gv * sy * .35, L = Math.hypot(dx, dy), c = Math.min(1, 46 / Math.max(L, 1e-6));
      D.arrow(PY.X(u), PY.Y(v), PY.X(u) + dx * c * ar, PY.Y(v) + dy * c * ar, '#ffffff', 2, 10, .8 * a0);
    }
    if (S.beat >= 1) {
      const a = S.at(1, 700), [u, v] = SENS.op;
      D.ring(PY.X(u), PY.Y(v), 16, COL.data, 3, a); D.dot(PY.X(u), PY.Y(v), 6, COL.data, a);
      const x0 = 1420, sc = 360 / Math.max(...SENS.net.map(Math.abs), ...SENS.tru.map(Math.abs)), q = Ease.out(S.at(1, 1400, 900));
      [[SENS.net[0], SENS.tru[0], 512], [SENS.net[1], SENS.tru[1], 612]].forEach(([n, t, y]) => {
        D.rect(x0, y, Math.abs(n) * sc * q, 24, COL.model, a);
        D.rect(x0, y + 30, Math.abs(t) * sc * q, 24, COL.good, a);
        D.text('±' + Math.abs(n).toFixed(1), x0 + Math.abs(n) * sc * q + 12, y + 12, { size: 22, col: COL.model, a: a * q });
        D.text('±' + Math.abs(t).toFixed(1), x0 + Math.abs(t) * sc * q + 12, y + 42, { size: 22, col: COL.good, a: a * q });
      });
    }
  },
});

quizScene('quiz-7', 'Chapter 7', [
  ['Why fit ln k₁ and ln k₂ rather than k₁ and k₂ directly?', 'Positivity for free, and rate constants span decades: a step in ln k is a relative change, so the bowl is far better conditioned.'],
  ['Gradient ascent on your surrogate ends at the edge of the data. Do you move the plant there?', 'Not yet. That is where the model knows least. Run a confirmation experiment there, add it to the data, refit, and repeat.'],
], ['First question: about parameterisation.', 'Fitting logarithms keeps the constants positive and turns multiplicative uncertainty into additive steps. It is the same conditioning lesson as standardising inputs in chapter 2. With Arrhenius parameters, fitting ln A and E over a centred 1/T is the standard trick for the same reason.', 'Second question: about trusting a surrogate.', 'An optimum at the edge of the data is a hypothesis, not an answer. The confirm, add, refit loop is exactly what Bayesian optimisation formalises, with an uncertainty estimate deciding where to try next.']);
