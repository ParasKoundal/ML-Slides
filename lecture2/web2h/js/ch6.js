/* ch6.js: Watch it learn. A 2-16-1 network trains live on the operating-window data; momentum and Adam;
 * overfitting and validation; a network extrapolates into nonsense and a hybrid physics model does not;
 * reading loss curves; the PyTorch loop. */
'use strict';

Deck.chapter('Watch it learn', 'training, validation, and knowing when to trust it');

chapterCard(6, 'Watch it learn', 'training, validation, and knowing when to trust it',
  'Chapter six. We have every piece: a model, a loss, gradients, and a way to compute them. Now we watch training happen, and learn how to tell when to trust the result.',
  (g, S) => {
    const P = new Plot(1250, 260, 520, 240, [0, 1], [0, 1]), q = S.p(2600, 300);
    D.fn(P, x => .08 + .85 * Math.exp(-5 * x), 0, q, COL.loss, 4, 1, 120, null, 10);
    D.fn(P, x => .14 + .8 * Math.exp(-4.4 * x) + .25 * Math.max(0, x - .45) ** 2, 0, q, COL.data, 3, .8, 120, [8, 8]);
  });

/* ------------------------------------------------------------ 1. live training on the operating window */
const LIVE = (() => {
  const net = new MLP([2, 16, 1], 'relu', 'sigmoid', 7), E = 400, snaps = [], loss = [], acc = [];
  for (let e = 0; e <= E; e++) {
    snaps.push({ W: net.W.map(L => L.map(r => r.slice())), b: net.b.map(v => v.slice()) });
    acc.push(net.acc(WIN.Z, WIN.y));
    loss.push(net.step(WIN.Z, WIN.y, 0.5, 0.9));
  }
  const p = (s, T_, pH) => { const z = WIN.z([T_, pH]); let o = s.b[1][0]; for (let j = 0; j < 16; j++) o += s.W[1][0][j] * Math.max(0, s.W[0][j][0] * z[0] + s.W[0][j][1] * z[1] + s.b[0][j]); return sigmoid(o); };
  return { E, snaps, loss, acc, p };
})();
Deck.add({
  id: 'live-train', title: 'Watch a network learn', beats: 3, dur: [3600, 10000, 5200], min: 4,
  notes: [
    'This is the 2-16-1 network from chapter 4, on the same 240 batches, starting from random knobs. The background is its current belief: amber where it thinks a batch will be good. Right now it knows nothing; the loss is high and the boundary is wherever the random start put it.',
    'Press play. Every frame is one or more full gradient-descent steps on all 240 batches, with momentum. Watch three things at once: the boundary closing in around the window, the cross-entropy falling, and accuracy rising past the 80 percent that a straight line could only fake. Within a hundred or so epochs it has the window. After that it keeps sharpening the edges: the loss still falls while accuracy is already perfect.',
    'Here is what the hidden layer is doing. Each line is one ReLU neuron: the line where its weighted sum crosses zero, so where its hinge sits. Cyan neurons vote for good, pink for bad. The window is a polygon cut out by these lines, which is exactly the hinge picture from the kinetics example, now in two dimensions.',
  ],
  html: () => `${hdr('Chapter 6', 'Watch a network find the window')}
    ${at(1240, 590, '<div class="lbl">epoch</div><div id="lte" class="num mono" style="font-size:88px;margin-top:6px"></div>', 0, 'rise')}
    ${at(1560, 590, '<div class="lbl">accuracy</div><div id="lta" class="num c-good" style="font-size:88px;margin-top:6px"></div>', 0, 'rise')}
    ${sub('Random start: the network knows nothing yet.', 0, 1)}
    ${sub('Gradient descent, step by step: the boundary closes in on the window.', 1, 2)}
    ${sub('Each line is one hidden neuron\'s hinge. <b>The window is a polygon of hinges.</b>', 2)}`,
  frame(g, S) {
    const L = LIVE;
    const q = S.beat === 0 ? 0 : S.beat === 1 ? S.at(1, 9200, 300) : 1;
    const e = Math.round(L.E * Math.pow(q, 2.2)), s = L.snaps[e];
    const img = heatImage(PW, (T_, pH) => L.p(s, T_, pH), { res: 8, lo: 0, hi: 1, stops: FIELD_STOPS });
    drawImg(img, PW, .95 * S.at(0, 600));
    drawSegs(contourSegs(PW, (T_, pH) => L.p(s, T_, pH), .5, 70, 44), COL.data, 4, S.at(0, 600), 12);
    winAxes(); winPoints(PW, S.at(0, 600), 6);
    if (S.beat >= 2) {
      const a = S.at(2, 1200, 200), w2 = s.W[1][0], wm = Math.max(...w2.map(Math.abs));
      for (let j = 0; j < 16; j++) {
        const f = (T_, pH) => { const z = WIN.z([T_, pH]); return s.W[0][j][0] * z[0] + s.W[0][j][1] * z[1] + s.b[0][j]; };
        drawSegs(contourSegs(PW, f, 0, 40, 26), w2[j] > 0 ? COL.model : COL.loss, 2, a * (.25 + .75 * Math.abs(w2[j]) / wm), 0);
      }
    }
    // loss curve
    const PL_ = new Plot(1240, 270, 560, 230, [0, L.E], [0, 1]);
    PL_.axes({ xt: [0, 100, 200, 300, 400], yt: [0, .5, 1], xl: 'epoch', yl: 'cross-entropy', a: S.at(0, 600) });
    if (e > 0) D.line(L.loss.slice(0, e + 1).map((v, i) => [PL_.X(i), PL_.Y(Math.min(v, 1))]), COL.loss, 3.5, 1, null, 10);
    const ee = document.getElementById('lte'); if (ee) ee.textContent = String(e);
    const ea = document.getElementById('lta'); if (ea) ea.textContent = Math.round(100 * L.acc[e]) + '%';
  },
});

/* ------------------------------------------------------------ 2. momentum and Adam */
const OPT = (() => {
  const lam = [10, 0.12], rot = 1.35, c = Math.cos(rot), s = Math.sin(rot);
  const L = (u, v) => { const a = c * u + s * v, b = -s * u + c * v; return 0.5 * (lam[0] * a * a + lam[1] * b * b); };
  const G = (u, v) => { const a = c * u + s * v, b = -s * u + c * v, ga = lam[0] * a, gb = lam[1] * b; return [c * ga - s * gb, s * ga + c * gb]; };
  const start = [c * 0.7 + s * 2.6, s * 0.7 - c * 2.6], N = 60;
  const gd = [start.slice()]; { let w = start.slice(); for (let k = 0; k < N; k++) { const g_ = G(...w); w = [w[0] - .18 * g_[0], w[1] - .18 * g_[1]]; gd.push(w); } }
  const mo = [start.slice()]; { let w = start.slice(), m = [0, 0]; for (let k = 0; k < N; k++) { const g_ = G(...w); m = [.85 * m[0] + g_[0], .85 * m[1] + g_[1]]; w = [w[0] - .05 * m[0], w[1] - .05 * m[1]]; mo.push(w); } }
  const ad = [start.slice()]; { let w = start.slice(), m = [0, 0], v = [0, 0]; for (let k = 1; k <= N; k++) { const g_ = G(...w); m = m.map((x, i) => .9 * x + .1 * g_[i]); v = v.map((x, i) => .999 * x + .001 * g_[i] * g_[i]); w = w.map((x, i) => x - .12 * (m[i] / (1 - .9 ** k)) / (Math.sqrt(v[i] / (1 - .999 ** k)) + 1e-8)); ad.push(w); } }
  return { L, gd, mo, ad, N };
})();
Deck.add({
  id: 'optimizers', title: 'Momentum and Adam', beats: 3, dur: [4600, 4600, 5200], min: 3,
  notes: [
    'Back to a stretched valley like the one in chapter 2, one steep direction and one shallow. Plain gradient descent with the largest safe step: it zig-zags across the steep walls and crawls along the floor. Sixty steps and it is still far from the bottom.',
    'Momentum gives the ball inertia. Keep a running sum of past gradients and step along that. The zig-zags cancel out, the consistent downhill direction adds up, and the ball rolls along the valley floor. Mathematically it is a heavy ball with friction, a second-order ODE you know from process dynamics.',
    'Adam goes one step further: it keeps a running average of each knob\'s squared gradient and divides by its square root, so every knob gets its own step size. Steep directions are slowed, shallow ones sped up. It is a cheap, diagonal form of the conditioning fix from chapter 2, and it is the default optimizer for training most networks, LLMs included.',
  ],
  html: () => `${hdr('Chapter 6', 'Give the ball momentum')}
    ${at(1250, 290, `<div class="lbl c-loss">gradient descent</div><div class="eq-xs" style="margin-top:8px">${M(T`w \leftarrow w - \eta\, g`)}</div>`, 0, 'rise')}
    ${at(1250, 450, `<div class="lbl c-model">momentum</div><div class="eq-xs" style="margin-top:8px">${M(T`m \leftarrow \beta m + g,\quad w \leftarrow w - \eta\, m`)}</div>`, 1, 'rise')}
    ${at(1250, 610, `<div class="lbl c-good">Adam</div><div class="eq-xs" style="margin-top:8px">${M(T`w_i \leftarrow w_i - \eta\,\frac{\bar m_i}{\sqrt{\bar v_i}+\epsilon}`)}</div><div class="lede" style="margin-top:8px;font-size:26px">a separate step size for every knob</div>`, 2, 'rise')}
    ${sub('Plain gradient descent: zig-zags, then crawls.', 0, 1)}
    ${sub('Momentum: the zig-zags cancel, the downhill pull adds up.', 1, 2)}
    ${sub('Adam: every knob gets its own step size. <b>The default for deep learning.</b>', 2)}`,
  enter(st) {
    st.P = new Plot(150, 270, 1000, 600, [-3, 3], [-1.8, 1.8]);
    st.img = heatImage(st.P, OPT.L, { res: 5, lo: 0, hi: 30, stops: ['#06131f', '#0b2c45', '#14506a', '#2a7d86', '#5fb3a1'] });
  },
  frame(g, S) {
    const P = S.st.P, a0 = S.at(0, 700);
    drawImg(S.st.img, P, .9 * a0);
    [0.05, .3, 1, 3, 8, 16, 26].forEach(l => drawSegs(contourSegs(P, OPT.L, l, 90, 54), '#7fd3c4', 1, .25 * a0));
    D.dot(P.X(0), P.Y(0), 8, COL.fg, a0);
    const path = (pts, b, col) => {
      if (S.beat < b) return;
      const q = S.beat === b ? S.at(b, 3600, 300) : 1, n = Math.max(1, Math.round(q * OPT.N));
      const xy = pts.slice(0, n + 1).map(([u, v]) => [P.X(u), P.Y(v)]);
      D.line(xy, col, 3, S.beat === b ? 1 : .55, null, 8);
      xy.forEach(([x, y]) => D.dot(x, y, 4, col, S.beat === b ? 1 : .5));
      D.dot(xy[xy.length - 1][0], xy[xy.length - 1][1], 9, col, 1, 14);
    };
    const g0 = D.g; g0.save(); g0.beginPath(); g0.rect(P.x, P.y, P.w, P.h); g0.clip();
    path(OPT.gd, 0, COL.loss); path(OPT.mo, 1, COL.model); path(OPT.ad, 2, COL.good);
    g0.restore();
    D.text('60 steps each, same start', P.x, P.y + P.h + 30, { size: 22, a: a0 });
  },
});

/* ------------------------------------------------------------ 3. overfitting: polynomial degree on Haldane data */
const PO = new Plot(150, 280, 900, 560, [0, 20], [-0.05, 0.35]);
const OV_BEST = HAL.polys.slice(1).reduce((b, p) => (p.va < b.va ? p : b));
Deck.add({
  id: 'overfit', title: 'Overfitting and validation', beats: 3, dur: [6800, 5200, 4800], min: 4,
  notes: [
    'Back to growth kinetics, but now with only ten training points: the amber dots. The faint grey dots are 200 more measurements from the same process that the model never sees during fitting; we hold them back. Fit polynomials of increasing degree. Degree one is too stiff. Around degree five or six the curve looks like the kinetics. By degree nine, with ten knobs for ten points, it threads every training point exactly and swings wildly in between.',
    'Plot both errors against degree, on a log scale. Training error, pink, only ever falls; at degree nine it is essentially zero. Validation error, the grey points the model never saw, falls, bottoms out, and rises again. The gap between the two curves is overfitting: the model has memorised the noise in its ten points instead of learning the process.',
    'So the rule: always hold data out, and choose the model, its size, and when to stop training by the validation error, never the training error. The same picture appears when you train a network for too many epochs: training loss keeps falling while validation loss turns up. The standard remedies are more data, a smaller or regularised model, and early stopping at the validation minimum. A third, untouched test set gives the honest final number.',
  ],
  html: () => `${hdr('Chapter 6', 'Zero training error, wrong answer')}
    ${at(1220, 270, '<div class="lbl">polynomial degree</div><div id="ovd" class="num c-model" style="margin-top:6px"></div>', 0, 'rise', '', 1)}
    ${sub('Ten training points (amber). Grey: held-out data the fit never sees.', 0, 1)}
    ${sub('Training error always falls. <b>Validation error turns back up.</b>', 1, 2)}
    ${sub('Choose complexity, and when to stop, by validation error.', 2)}`,
  frame(g, S) {
    const a0 = S.at(0, 700);
    PO.axes({ xt: [0, 5, 10, 15, 20], yt: [0, 0.1, 0.2, 0.3], xl: 'substrate S  (g/L)', yl: 'growth rate μ  (1/h)', a: a0 });
    HAL.Sva.forEach((s, i) => D.dot(PO.X(s), PO.Y(HAL.Yva[i]), 3, '#55606f', .7 * a0));
    let d;
    if (S.beat === 0) { const q = S.at(0, 6000, 400); d = 1 + Math.min(8, Math.floor(q * 9)); }
    else if (S.beat === 1) { const q = S.at(1, 4200, 300); d = 1 + Math.min(8, Math.floor(q * 9)); }
    else d = OV_BEST.d;
    const p = HAL.polys[d], col = S.beat === 2 ? COL.good : COL.model;
    const g0 = D.g; g0.save(); g0.beginPath(); g0.rect(PO.x, PO.y, PO.w, PO.h); g0.clip();
    if (S.beat === 2) D.fn(PO, HAL.polys[9].f, 0.2, 19.8, COL.loss, 2, .35, 400);
    D.fn(PO, p.f, 0.2, 19.8, col, 4, a0, 400, null, 10);
    g0.restore();
    HAL.Str.forEach((s, i) => D.dot(PO.X(s), PO.Y(HAL.Ytr[i]), 9, COL.data, a0, 8));
    const el = document.getElementById('ovd'); if (el) el.textContent = String(d);
    if (S.beat >= 1) {
      const a = S.at(1, 700), PE = new Plot(1240, 300, 560, 470, [1, 9], [1e-7, 3e-3], { logy: true });
      PE.axes({ xt: [1, 3, 5, 7, 9], yt: [1e-6, 1e-4, 1e-2].filter(v => v < 3e-3), xl: 'polynomial degree', yl: 'mean squared error (log)', a, yf: v => '10' + { 1e-6: '⁻⁶', 1e-4: '⁻⁴', 1e-2: '⁻²' }[v] });
      const dm = S.beat === 1 ? d : 9;
      const tr = HAL.polys.slice(1, dm + 1).map(q => [PE.X(q.d), PE.Y(Math.max(q.tr, 1.2e-7))]);
      const va = HAL.polys.slice(1, dm + 1).map(q => [PE.X(q.d), PE.Y(q.va)]);
      D.line(tr, COL.loss, 3.5, a, null, 8); tr.forEach(([x, y]) => D.dot(x, y, 6, COL.loss, a));
      D.line(va, '#aab4c3', 3.5, a, null, 0); va.forEach(([x, y]) => D.dot(x, y, 6, '#aab4c3', a));
      D.text('training', PE.X(6.2), PE.Y(3e-6), { size: 22, col: COL.loss, a });
      D.text('validation', PE.X(1.2), PE.Y(3e-4), { size: 22, col: '#aab4c3', a });
      if (S.beat >= 2) { const b = S.at(2, 600); D.ring(PE.X(OV_BEST.d), PE.Y(OV_BEST.va), 16, COL.good, 3, b); D.text('best on held-out data', PE.X(OV_BEST.d), PE.Y(OV_BEST.va) - 40, { size: 22, align: 'center', col: COL.good, a: b }); }
    }
  },
});

/* ------------------------------------------------------------ 4. extrapolation, and a hybrid model that fixes it */
const HYB = (() => {
  const mu = (t, S_) => Math.exp(t[0]) * S_ / (Math.exp(t[1]) + S_ + S_ * S_ / Math.exp(t[2]));
  const L = t => mean(HAL.S.map((S_, i) => (mu(t, S_) - HAL.Y[i]) ** 2));
  let th = [Math.log(.2), Math.log(3), Math.log(30)], m = [0, 0, 0], v = [0, 0, 0];
  const path = [th.slice()];
  for (let k = 1; k <= 300; k++) {
    const gr = th.map((_, i) => { const h = 1e-5, a = th.slice(), b = th.slice(); a[i] += h; b[i] -= h; return (L(a) - L(b)) / (2 * h); });
    gr.forEach((gi, i) => { m[i] = .9 * m[i] + .1 * gi; v[i] = .999 * v[i] + .001 * gi * gi; th[i] -= .05 * (m[i] / (1 - .9 ** k)) / (Math.sqrt(v[i] / (1 - .999 ** k)) + 1e-8); });
    path.push(th.slice());
  }
  return { mu, L, path, relu: HAL.fitRelu(16) };
})();
const PX = new Plot(150, 280, 1050, 560, [0, 40], [-0.1, 0.32]);
Deck.add({
  id: 'extrapolate', title: 'Networks do not extrapolate; physics does', beats: 3, dur: [4200, 5200, 6400], min: 4,
  notes: [
    'The 16-neuron fit from chapter 4: an excellent interpolator. On the data range, zero to 20 grams per litre, it is within the noise everywhere.',
    'Now ask it about a fed-batch run that reaches 40 grams per litre. The network has never seen anything past 20, so it simply continues its last hinge in a straight line, and predicts a negative growth rate. That is physically impossible. A neural network is a very flexible interpolator with no idea what lies outside its data, and it will not warn you.',
    'The fix that works best in engineering is to build in what you know. Keep the Haldane form, mu max S over Ks plus S plus S squared over Ki, and let gradient descent fit just its three knobs, exactly the same loop as for a network. Watch it converge. Inside the data both models agree; outside, the hybrid model stays physical, because the physics is in its structure. Gradient descent does not care whether the knobs belong to a network or to your rate law: this is the basis of hybrid and physics-informed models, and it is often the right tool for chemical engineering data.',
  ],
  html: () => `${hdr('Chapter 6', 'A network extrapolates nonsense')}
    ${at(1290, 280, '<div class="lbl c-model">ReLU network, 16 neurons</div><div class="big mono" style="margin-top:8px">μ(40) = <span class="c-loss">' + fmt(HYB.relu.f(40), 3) + '</span></div>', 1, 'rise')}
    ${at(1290, 470, `<div class="lbl c-good">hybrid: Haldane form, 3 knobs</div><div class="eq-xs" style="margin-top:10px">${M(T`\mu = \frac{\mu_{\max} S}{K_s + S + S^2/K_i}`)}</div><div id="hyb" class="big mono" style="margin-top:12px"></div><div class="lede" style="margin-top:10px;font-size:26px">fitted by the same gradient loop</div>`, 2, 'rise')}
    ${sub('Inside the data, the network fits beautifully.', 0, 1)}
    ${sub('Past 20 g/L it continues its last hinge: <b>negative growth.</b>', 1, 2)}
    ${sub('Put the physics in the model, fit its knobs by gradient descent.', 2)}`,
  frame(g, S) {
    const a0 = S.at(0, 700);
    D.rect(PX.X(20), PX.y, PX.X(40) - PX.X(20), PX.h, '#0e131b', a0);
    D.text('no data here', PX.X(30), PX.y + 30, { size: 22, align: 'center', a: a0 });
    PX.axes({ xt: [0, 10, 20, 30, 40], yt: [-0.1, 0, 0.1, 0.2, 0.3], xl: 'substrate S  (g/L)', yl: 'growth rate μ  (1/h)', a: a0, yf: v => fmt(v, 1) });
    D.seg(PX.x, PX.Y(0), PX.x + PX.w, PX.Y(0), '#3a4352', 1.5, a0);
    D.fn(PX, HAL.mu, 0.05, S.beat >= 1 ? 40 : 20, '#8792a3', 2, .8 * a0, 300, [7, 7]);
    HAL.S.forEach((s, i) => D.dot(PX.X(s), PX.Y(HAL.Y[i]), 7, COL.data, a0, 6));
    D.fn(PX, HYB.relu.f, 0, 20, COL.model, 4, a0, 300, null, 10);
    if (S.beat >= 1) {
      const q = S.beat === 1 ? Ease.inOut(S.at(1, 1800, 300)) : 1;
      D.fn(PX, HYB.relu.f, 20, 20 + 20 * q, COL.loss, 4, 1, 200, null, 12);
      if (q > .98) { D.dot(PX.X(40), PX.Y(HYB.relu.f(40)), 9, COL.loss, 1, 12); D.text('impossible', PX.X(40) - 16, PX.Y(HYB.relu.f(40)) + 36, { size: 24, align: 'right', col: COL.loss, a: S.at(1, 600, 2000) }); }
    }
    if (S.beat >= 2) {
      const q = S.at(2, 4200, 400), k = Math.round(Math.pow(q, 1.6) * (HYB.path.length - 1)), th = HYB.path[k];
      D.fn(PX, s => HYB.mu(th, s), 0.05, 40, COL.good, 4, 1, 300, null, 12);
      const el = document.getElementById('hyb');
      if (el) el.innerHTML = `step ${k} · μ(40) = <span class="c-good">${fmt(HYB.mu(th, 40), 3)}</span>`;
    }
  },
});

/* ------------------------------------------------------------ 5. the loss-curve doctor */
const DOC = [
  { t: 'loss spikes or turns NaN', d: 'step size too big: lower η', col: COL.loss, tr: x => .9 * Math.exp(-x * 3) + .15 + .12 * Math.abs(Math.sin(x * 37)) * x + (x > .75 ? 3 * (x - .75) ** 2 * 6 : 0) },
  { t: 'barely moving', d: 'step size too small: raise η, check scaling', col: COL.data, tr: x => .95 - .25 * x },
  { t: 'training falls, validation rises', d: 'overfitting: stop early, regularise, more data', col: COL.grad, tr: x => .85 * Math.exp(-x * 5) + .06, va: x => .85 * Math.exp(-x * 5) + .16 + .5 * Math.max(0, x - .35) ** 1.5 },
  { t: 'both flatten together', d: 'converged: done, or try a bigger model', col: COL.good, tr: x => .8 * Math.exp(-x * 5) + .12, va: x => .8 * Math.exp(-x * 5) + .17 },
];
Deck.add({
  id: 'doctor', title: 'Read the loss curve', beats: 4, dur: [3400, 3400, 3800, 3600], min: 3,
  notes: [
    'The loss curve is your main diagnostic, so learn to read it. These are illustrative shapes. First: the loss jumps around, climbs, or becomes NaN. The step size is too big; you are overshooting the valley as in chapter 2. Lower the learning rate.',
    'Second: the loss falls in a slow, straight line. The step is too small, or the inputs are badly scaled so the valley is stretched. Raise the learning rate, standardise the features, or use Adam.',
    'Third: training loss keeps falling while validation loss turns up. That is overfitting, the polynomial picture again. Stop at the validation minimum, regularise, or get more data.',
    'Fourth: both curves flatten together with a small gap. Training has converged. If both are still too high, the model is too simple for the problem: more neurons, more layers, or better features.',
  ],
  html: () => `${hdr('Chapter 6', 'Read the loss curve like a trend chart')}
    ${DOC.map((c, k) => at(150 + (k % 2) * 860, 300 + Math.floor(k / 2) * 330, `<div class="lbl" style="color:${c.col}">${c.t}</div>`, k, 'rise')).join('')}
    ${DOC.map((c, k) => at(560 + (k % 2) * 860, 380 + Math.floor(k / 2) * 330, `<div class="lede" style="width:340px;font-size:28px">${c.d}</div>`, k, 'rise', '--d:900ms')).join('')}
    ${sub('Illustrative shapes. Each one tells you what to change.', 0)}`,
  frame(g, S) {
    DOC.forEach((c, k) => {
      if (S.beat < k) return;
      const P = new Plot(150 + (k % 2) * 860, 340 + Math.floor(k / 2) * 330, 360, 200, [0, 1], [0, 1.2]);
      const a = S.at(k, 500), q = S.beat === k ? S.at(k, 1800, 200) : 1;
      P.axes({ xt: [], yt: [], xl: 'epoch', yl: '', a });
      D.fn(P, x => Math.min(1.2, c.tr(x)), 0, q, c.col, 3.5, 1, 160, null, 8);
      if (c.va) D.fn(P, x => Math.min(1.2, c.va(x)), 0, q, '#aab4c3', 3, 1, 160, [7, 6]);
    });
  },
});

/* ------------------------------------------------------------ 6. the PyTorch loop */
const TORCH = [
  [1, 'model = nn.Sequential(nn.Linear(<span class="n">2</span>, <span class="n">16</span>), nn.ReLU(),'],
  [1, '                      nn.Linear(<span class="n">16</span>, <span class="n">1</span>))'],
  [2, 'loss_fn = nn.BCEWithLogitsLoss()'],
  [3, 'opt = torch.optim.Adam(model.parameters(), lr=<span class="n">1e-2</span>)'],
  [0, ''],
  [3, '<span class="k">for</span> epoch <span class="k">in</span> range(<span class="n">300</span>):'],
  [4, '    <span class="k">for</span> xb, yb <span class="k">in</span> loader:          <span class="c"># mini-batches</span>'],
  [2, '        loss = loss_fn(model(xb), yb)'],
  [3, '        opt.zero_grad()'],
  [3, '        loss.<span class="f">backward</span>()            <span class="c"># chapter 5</span>'],
  [3, '        opt.<span class="f">step</span>()                <span class="c"># chapter 2</span>'],
  [4, '    val = loss_fn(model(X_val), y_val)  <span class="c"># chapter 6</span>'],
];
const TORCH_COL = [null, '63,208,255', '255,77,125', '180,140,255', '255,181,71'];
Deck.add({
  id: 'torch', title: 'The training loop in PyTorch', beats: 4, dur: [3400, 3400, 3800, 3800], min: 3,
  notes: [
    'Here is everything in this lecture in twelve lines of PyTorch, the same network as the numpy version. First the model, chapter 4: a linear layer, a ReLU, another linear layer.',
    'The loss, chapter 3: binary cross-entropy. The "with logits" version folds the sigmoid into the loss, which is numerically safer.',
    'Learning, chapters 2 and 5: the optimizer, here Adam; zero the old gradients, call backward for backprop, and step. Note zero_grad: PyTorch adds gradients up by default, a classic first bug.',
    'And the data: mini-batches from a loader, chapter 2, and a validation loss computed every epoch, this chapter, so you can see overfitting coming. Model, loss, learning: the same three questions as every lecture in this series.',
  ],
  html: () => `${hdr('Chapter 6', 'The whole lecture in twelve lines')}
    ${at(96, 260, `<div class="code" id="tch" style="font-size:24px;line-height:1.55;padding:24px 32px">${TORCH.map(([gp, l]) => `<div class="ln" data-g="${gp}">${l || ' '}</div>`).join('')}</div>`, 0, 'rise')}
    ${at(1390, 300, '<div class="lbl c-model">model</div><div class="lede" style="margin-top:6px">chapter 4</div>', 0, 'rise')}
    ${at(1390, 430, '<div class="lbl c-loss">loss</div><div class="lede" style="margin-top:6px">chapters 1 and 3</div>', 1, 'rise')}
    ${at(1390, 560, '<div class="lbl c-grad">learning</div><div class="lede" style="margin-top:6px">chapters 2 and 5</div>', 2, 'rise')}
    ${at(1390, 690, '<div class="lbl c-data">data and validation</div><div class="lede" style="margin-top:6px">chapters 2 and 6</div>', 3, 'rise')}
    ${sub('Model: chapter 4.', 0, 1)}
    ${sub('Loss: cross-entropy, sigmoid folded in.', 1, 2)}
    ${sub('Learning: zero, backward, step. <b>Forget zero_grad and gradients pile up.</b>', 2, 3)}
    ${sub('Mini-batches in, validation out. Model, loss, learning.', 3)}`,
  frame(g, S) {
    const box = document.getElementById('tch'); if (!box) return;
    const f = S.p(500), cur = S.beat + 1;
    box.querySelectorAll('.ln').forEach(el => {
      const gp = +el.dataset.g, on = gp === cur, seen = gp > 0 && gp < cur;
      el.style.opacity = on ? 1 : seen ? .75 : lerp(1, .3, f);
      el.style.background = on ? `rgba(${TORCH_COL[gp]},${.14 * f})` : 'transparent';
    });
  },
});

quizScene('quiz-6', 'Chapter 6', [
  ['Training loss keeps falling; validation loss started rising at epoch 40. What do you do?', 'Overfitting. Keep the epoch-40 weights (early stopping), then regularise or get more data.'],
  ['Your network predicts 140% yield for a new feed composition. What went wrong, and how could you fix it?', 'It is extrapolating. Check the input is inside the training range; bound the output or build the known physics into the model.'],
], ['Early stopping.', 'The validation curve tells you when the model started memorising. Keep the weights from its minimum; regularisation and more data push that minimum lower.', 'Extrapolation.', 'Networks are interpolators. Flag inputs outside the training range, bound the output with a sigmoid times 100 percent, or use a hybrid model with mass balances or rate laws built in.']);
