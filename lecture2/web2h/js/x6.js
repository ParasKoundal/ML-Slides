/* x6.js: extended cut, chapter 6. Initialisation (activations through depth); weight decay and dropout;
 * early stopping on a live training run; cross-validation and leakage between batches; ensembles as an
 * uncertainty flag. */
'use strict';

/* ------------------------------------------------------------ initialisation: what the signal looks like at depth */
const INIT = (() => {
  const r = RNG(41), n = 400, w = 64, L = 8, nb = 41;
  const X0 = Array.from({ length: n }, () => Array.from({ length: w }, () => r.n()));
  const run = sd => {
    let A = X0; const layers = [];
    for (let l = 0; l < L; l++) {
      const Wm = Array.from({ length: w }, () => Array.from({ length: w }, () => r.n() * sd));
      A = A.map(x => Wm.map(row => { let z = 0; for (let k = 0; k < w; k++) z += row[k] * x[k]; return Math.tanh(z); }));
      const h = Array(nb).fill(0); let s2 = 0;
      A.forEach(a => a.forEach(v => { h[Math.min(nb - 1, Math.floor((v + 1) / 2 * nb))]++; s2 += v * v; }));
      layers.push({ h, sd: Math.sqrt(s2 / (n * w)) });
    }
    return layers;
  };
  return { L, nb, cases: [run(0.02), run(0.6), run(1.6 / Math.sqrt(w))] };
})();
const INIT_CASES = [['too small', 'σ = 0.02', COL.cls0], ['too large', 'σ = 0.6', COL.loss], ['scaled to fan-in (Xavier)', 'σ = 1.6/√64 = 0.2', COL.good]];
Deck.insert('optimizers', {
  id: 'init', title: 'Initialisation: keep the signal alive', beats: 3, dur: [4800, 4800, 5400], min: 3.5,
  notes: [
    'Before the first gradient step, every weight needs a starting value, and the choice matters more than it looks. Here a stack of eight tanh layers, 64 neurons each, and 400 random inputs. Each row is a histogram of the neuron outputs at one layer, from the first at the top to the eighth at the bottom. Start every weight with a tiny standard deviation, 0.02. Each layer multiplies the signal by about 0.16, so by layer three everything is squeezed onto zero. The forward signal is gone, and so, by the chain rule, is the gradient.',
    'Overcorrect, standard deviation 0.6, and each pre-activation is huge. Every tanh is pinned at plus or minus one: all the histograms pile up at the edges. Those neurons are saturated, their slope is nearly zero, and gradients vanish through them for the opposite reason.',
    'The fix is to scale the random weights to the number of inputs each neuron has, its fan-in: standard deviation one over the square root of fan-in, the Xavier or Glorot rule, times a small gain of about 1.6 for tanh because tanh squashes. Then the variance of the signal stays roughly constant from layer to layer and the histograms keep their shape all the way down. For ReLU networks the He rule uses square root of two over fan-in, because ReLU zeroes half its inputs. PyTorch layers already default to rules like these, which is one reason deep networks became trainable around 2010.',
  ],
  html: () => `${hdr('Chapter 6 · initialisation', 'Start the weights at the right size')}
    ${INIT_CASES.map(([t, s, c], k) => at(1250, 300, `<div class="lbl" style="color:${c}">${t}</div><div class="big mono" style="margin-top:8px">${s}</div>`, k, 'rise', '', k < 2 ? k + 1 : undefined)).join('')}
    ${at(1250, 470, '<div class="lede" style="font-size:27px;width:560px">8 tanh layers × 64 neurons, 400 random inputs. Each row: the outputs of one layer.</div>', 0, 'rise')}
    ${at(1250, 660, `<div class="eq-s">${M(T`\operatorname{sd}(W) = \frac{g}{\sqrt{n_{\mathrm{in}}}}`)}</div><div class="lede" style="font-size:27px;margin-top:12px;width:560px">g ≈ 1.6 for tanh (Xavier), √2 for ReLU (He)</div>`, 2, 'rise', '--d:800ms')}
    ${sub('Too small: each layer shrinks the signal. By layer 3 it is gone.', 0, 1)}
    ${sub('Too large: every tanh saturates at ±1, <b>and the slopes vanish.</b>', 1, 2)}
    ${sub('Scale by fan-in: the signal keeps its shape through every layer.', 2)}`,
  frame(g, S) {
    const cs = INIT.cases[S.beat], col = INIT_CASES[S.beat][2], x0 = 230, w = 820, rowH = 68, y0 = 290, bw = w / INIT.nb;
    const a0 = S.at(S.beat, 500);
    cs.forEach((ly, l) => {
      const a = a0 * clamp(S.p(400, 200 + l * 250) * 1.2), mx = Math.max(...ly.h), base = y0 + l * rowH + rowH - 8;
      D.text('layer ' + (l + 1), x0 - 20, base - 20, { size: 20, align: 'right', a });
      D.seg(x0, base, x0 + w, base, '#3a4352', 1, a);
      ly.h.forEach((v, k) => { const hh = (v / mx) * (rowH - 14); if (hh > .5) D.rect(x0 + k * bw + 1, base - hh, bw - 2, hh, col, .85 * a); });
      D.text('sd ' + ly.sd.toFixed(3), x0 + w + 20, base - 20, { size: 20, a, col: COL.dim });
    });
    [-1, 0, 1].forEach(v => D.text(String(v), x0 + (v + 1) / 2 * w, y0 + 8 * rowH + 22, { size: 20, align: 'center', a: a0 }));
    D.text('neuron output (tanh)', x0 + w, y0 + 8 * rowH + 56, { size: 22, align: 'right', a: a0, font: '"IBM Plex Sans", sans-serif' });
  },
});

/* ------------------------------------------------------------ weight decay and dropout */
const WD = (() => {
  const cheb = (s, d) => { const t = s / 10 - 1, T0 = [1, t]; for (let k = 2; k <= d; k++) T0.push(2 * t * T0[k - 1] - T0[k - 2]); return T0.slice(0, d + 1); };
  const fit = lam => { const c = lstsq(HAL.Str.map(s => cheb(s, 9)), HAL.Ytr, lam); const f = s => cheb(s, 9).reduce((a, v, i) => a + v * c[i], 0); return { f, tr: mean(HAL.Str.map((s, i) => (f(s) - HAL.Ytr[i]) ** 2)), va: mean(HAL.Sva.map((s, i) => (f(s) - HAL.Yva[i]) ** 2)) }; };
  const lams = linspace(-9, 1, 41).map(e => 10 ** e), fits = lams.map(fit);
  const best = fits.reduce((b, f, i) => (f.va < fits[b].va ? i : b), 0);
  return { lams, fits, best };
})();
Deck.insert('extrapolate', {
  id: 'regularise', title: 'Weight decay and dropout', beats: 3, dur: [4600, 6400, 6200], min: 4,
  notes: [
    'The degree-nine polynomial from the overfitting scene: ten knobs, ten points, a curve that threads every point and swings wildly in between. To get those swings, the fit needs some very large coefficients that nearly cancel each other.',
    'Weight decay, which is ridge regression from chapter 1 under another name, adds lambda times the sum of squared weights to the loss. Large coefficients now cost something. Watch lambda grow from nearly zero: the swings flatten, the training error rises a little, and the validation error falls to a minimum, about ' + (WD.fits[0].va / WD.fits[WD.best].va).toFixed(1) + ' times lower; most of what is left is the measurement noise itself. Too much lambda and the curve becomes too stiff again; choose lambda on validation data like any other setting. In PyTorch it is the weight_decay argument of the optimiser, and AdamW is Adam with it done properly.',
    'Dropout is the other standard regulariser for networks. On every training step, switch off a random half of the hidden neurons, a different half each time, as in this picture. No neuron can rely on any particular partner being present, so the network cannot build fragile, memorised combinations; it has to spread the knowledge out. It behaves like training a huge ensemble of thinned networks that share weights. At prediction time all neurons are on, scaled to match. One line in PyTorch: nn.Dropout of p.',
  ],
  html: () => `${hdr('Chapter 6 · regularisation', 'Make big weights expensive')}
    ${at(1240, 290, `<div class="eq-s">${M(T`L = \text{MSE} + \color{#b48cff}{\lambda \textstyle\sum_j w_j^2}`)}</div>`, 0, 'rise')}
    ${at(1240, 400, '<div class="lbl c-grad">λ</div>', 1, 'rise', '', 2)}
    ${at(1240, 432, '<div id="wdl" class="mono c-grad" style="font:600 56px/1 var(--mono)"></div>', 1, 'rise', '', 2)}
    ${at(1240, 520, '<div class="lbl">validation error</div>', 1, 'rise', '', 2)}
    ${at(1240, 552, '<div id="wdv" class="mono" style="font:600 56px/1 var(--mono)"></div>', 1, 'rise', '', 2)}
    ${at(1240, 400, '<div class="lbl c-data">dropout, p = 0.5</div><div class="lede" style="font-size:27px;margin-top:10px;width:560px">every step, a random half of the hidden neurons is switched off</div>', 2, 'rise')}
    ${sub('Degree 9, ten points: huge, cancelling coefficients.', 0, 1)}
    ${sub('Charge for large weights: <b>the swings flatten, validation error drops.</b>', 1, 2)}
    ${sub('Dropout: random neurons off each step. No fragile memorised partnerships.', 2)}`,
  frame(g, S) {
    if (S.beat <= 1) {
      const a0 = S.at(0, 700);
      PO.axes({ xt: [0, 5, 10, 15, 20], yt: [0, 0.1, 0.2, 0.3], xl: 'substrate S  (g/L)', yl: 'growth rate μ  (1/h)', a: a0 });
      HAL.Sva.forEach((s, i) => D.dot(PO.X(s), PO.Y(HAL.Yva[i]), 3, '#55606f', .7 * a0));
      const q = S.beat === 0 ? 0 : Ease.inOut(S.at(1, 4800, 600)) * WD.best, k = Math.min(WD.fits.length - 2, Math.floor(q)), f = q - k;
      const fn = s => lerp(WD.fits[k].f(s), WD.fits[k + 1].f(s), f);
      const g0 = D.g; g0.save(); g0.beginPath(); g0.rect(PO.x, PO.y, PO.w, PO.h); g0.clip();
      D.fn(PO, fn, 0.2, 19.8, mixHex(COL.loss, COL.good, q / WD.best), 4, a0, 400, null, 10);
      g0.restore();
      HAL.Str.forEach((s, i) => D.dot(PO.X(s), PO.Y(HAL.Ytr[i]), 9, COL.data, a0, 8));
      const put = (id, v) => { const el = document.getElementById(id); if (el) el.textContent = v; };
      put('wdl', S.beat === 0 ? '0' : sci(WD.lams[Math.round(q)], 0)); put('wdv', sci(lerp(WD.fits[k].va, WD.fits[k + 1].va, f), 1));
      return;
    }
    // dropout: a 1-8-8-1 network, a new random half switched off every 450 ms
    const a = S.at(2, 600), step = Math.floor(S.bt / 450), r = RNG(1000 + step);
    const col = [{ x: 200, n: 1 }, { x: 480, n: 8 }, { x: 760, n: 8 }, { x: 1040, n: 1 }], Y = (n, i) => 560 + (i - (n - 1) / 2) * 62;
    const on = col.map((c, l) => Array.from({ length: c.n }, () => (l === 0 || l === 3 ? true : r() > .5)));
    for (let l = 0; l < 3; l++) for (let i = 0; i < col[l].n; i++) for (let j = 0; j < col[l + 1].n; j++) {
      const live = on[l][i] && on[l + 1][j];
      D.seg(col[l].x, Y(col[l].n, i), col[l + 1].x, Y(col[l + 1].n, j), live ? COL.model : '#1d2430', live ? 1.6 : 1, a * (live ? .7 : .5));
    }
    col.forEach((c, l) => { for (let i = 0; i < c.n; i++) { const o = on[l][i]; D.dot(c.x, Y(c.n, i), 18, o ? '#0f141c' : '#07090d', a); D.ring(c.x, Y(c.n, i), 18, o ? COL.data : '#2a3240', 2.5, a); if (!o) D.text('×', c.x, Y(c.n, i), { size: 22, align: 'center', col: '#55606f', a }); } });
    D.text('training step ' + (step + 1), 620, 880, { size: 24, align: 'center', col: COL.data, a });
  },
});

/* ------------------------------------------------------------ early stopping on a real training run */
const ES = (() => {
  // eight noisy growth-rate measurements and 200 held-out ones, from the Haldane kinetics of chapter 6
  const r = RNG(104), S = linspace(0.5, 19.5, 8).map(v => v + r.u(-.5, .5)), Y = S.map(v => HAL.mu(v) + 0.012 * r.n());
  const Sv = linspace(0.3, 19.7, 200), Yv = Sv.map(v => HAL.mu(v) + 0.012 * r.n()), sy = 6, sx = v => [v / 10 - 1];
  const net = new MLP([1, 40, 1], 'tanh', 'linear', 4, 1), X = S.map(sx), YY = Y.map(y => y * sy), Xv = Sv.map(sx), YYv = Yv.map(y => y * sy);
  const E = 20000, every = 50, tr = [], va = [], snaps = [];
  for (let e = 0; e <= E; e++) {
    if (e % every === 0) { tr.push(net.loss(X, YY) / sy / sy); va.push(net.loss(Xv, YYv) / sy / sy); snaps.push({ W: net.W.map(L => L.map(rw => rw.slice())), b: net.b.map(q => q.slice()) }); }
    net.step(X, YY, 0.05, 0.9);
  }
  const best = va.reduce((b, v, i) => (v < va[b] ? i : b), 0);
  const f = (v, sn) => { let o = sn.b[1][0]; const x = v / 10 - 1; for (let j = 0; j < 40; j++) o += sn.W[1][0][j] * Math.tanh(sn.W[0][j][0] * x + sn.b[0][j]); return o / sy; };
  return { S, Y, Sv, Yv, E, every, tr, va, best, snaps, f };
})();
Deck.insert('extrapolate', {
  id: 'early-stop', title: 'Early stopping', beats: 3, dur: [8400, 4800, 5400], min: 3.5,
  notes: [
    'Eight noisy growth-rate measurements from the same kinetics, and a network far too big for them: one input, 40 tanh neurons, 121 knobs, trained by gradient descent with momentum for 20,000 epochs. Left, the fit as it trains. Right, against epoch on a log scale: the training loss in pink, and below it the validation loss on the 200 held-out points in grey. For the first few thousand epochs both fall together: the network is learning the shape of the kinetics. Then they part. Training loss keeps falling as the network bends itself through each noisy point; validation loss turns around and climbs, by about ' + Math.round(100 * (ES.va[ES.va.length - 1] / ES.va[ES.best] - 1)) + ' percent, while the training loss keeps falling: the network is now fitting the noise in its eight points. Most of the validation error is measurement noise that no model can remove, which is why the rise looks modest; on the fit it is obvious.',
    'Early stopping simply keeps the weights from the epoch where validation loss was lowest, here epoch ' + ES.best * ES.every + ', the green ring. The fit from that epoch is the smooth one. In practice you evaluate on validation data every epoch or every few hundred steps, save a checkpoint whenever it improves, and stop when it has not improved for a set number of evaluations, the patience.',
    'Early stopping is regularisation by time: a network that has taken fewer steps from a small initialisation cannot have grown the large, finely tuned weights that memorising needs. It is free, it combines with weight decay and dropout, and almost every serious training script does it. The one rule: the data you stop on is now part of model selection, so report the final number on a separate test set.',
  ],
  html: () => `${hdr('Chapter 6 · early stopping', 'Stop when validation stops improving')}
        ${at(1250, 270, '<div class="lbl">epoch</div>', 0, 'rise')}
    ${at(1400, 254, '<div id="ese" class="mono" style="font:600 44px/1 var(--mono)"></div>', 0, 'rise')}
    ${sub('Both losses fall together, then part ways.', 0, 1)}
    ${sub('Keep the weights from the validation minimum: <b>the smooth fit.</b>', 1, 2)}
    ${sub('Checkpoint the best, stop after a patience window. Test on fresh data.', 2)}`,
  frame(g, S) {
    const a0 = S.at(0, 700), P = new Plot(150, 300, 960, 520, [0, 20], [-0.05, 0.35]);
    P.axes({ xt: [0, 5, 10, 15, 20], yt: [0, 0.1, 0.2, 0.3], xl: 'substrate S  (g/L)', yl: 'growth rate μ  (1/h)', a: a0 });
    ES.Sv.forEach((s, i) => D.dot(P.X(s), P.Y(ES.Yv[i]), 3, '#55606f', .7 * a0));
    const N = ES.tr.length - 1, n = S.beat === 0 ? Math.round(Math.pow(N, Ease.inOut(S.at(0, 7400, 500)))) : N;
    const sn = ES.snaps[S.beat === 0 ? n : ES.best];
    const g0 = D.g; g0.save(); g0.beginPath(); g0.rect(P.x, P.y, P.w, P.h); g0.clip();
    if (S.beat >= 1) D.fn(P, s => ES.f(s, ES.snaps[N]), 0.2, 19.8, COL.loss, 2.5, .5, 300);
    D.fn(P, s => ES.f(s, sn), 0.2, 19.8, S.beat >= 1 ? COL.good : COL.model, 4, a0, 300, null, 10);
    g0.restore();
    ES.S.forEach((s, i) => D.dot(P.X(s), P.Y(ES.Y[i]), 9, COL.data, a0, 8));
    const vlo = Math.min(...ES.va) * .97, vhi = Math.max(...ES.va.slice(20)) * 1.03;
    const Q1 = new Plot(1250, 330, 560, 190, [ES.every, ES.E], [Math.min(...ES.tr) * .5, ES.tr[0] * 2], { logy: true, logx: true });
    const Q2 = new Plot(1250, 590, 560, 190, [ES.every, ES.E], [vlo, vhi], { logx: true });
    Q1.axes({ xt: [], yt: [], a: a0, grid: false }); Q2.axes({ xt: [100, 1000, 10000], xf: v => (v >= 1000 ? v / 1000 + 'k' : String(v)), xl: 'epoch (log)', a: a0 });
    D.text('training loss (log)', Q1.x + 10, Q1.y + 10, { size: 20, col: COL.loss, a: a0, font: '"IBM Plex Sans", sans-serif' });
    D.text('validation loss (zoomed)', Q2.x + 10, Q2.y + 10, { size: 20, col: '#aab4c3', a: a0, font: '"IBM Plex Sans", sans-serif' });
    D.line(ES.tr.slice(1, n + 1).map((v, i) => [Q1.X((i + 1) * ES.every), Q1.Y(v)]), COL.loss, 3.5, a0);
    D.line(ES.va.slice(1, n + 1).map((v, i) => [Q2.X((i + 1) * ES.every), Q2.Y(Math.min(v, vhi))]), '#aab4c3', 3.5, a0);
    if (S.beat >= 1) { const b = S.at(1, 600); D.ring(Q2.X(ES.best * ES.every), Q2.Y(ES.va[ES.best]), 16, COL.good, 3, b); D.seg(Q2.X(ES.best * ES.every), Q1.y, Q2.X(ES.best * ES.every), Q2.y + Q2.h, COL.good, 1.5, .6 * b, [6, 6]); }
    const el = document.getElementById('ese'); if (el) el.textContent = String((S.beat === 0 ? n : ES.best) * ES.every);
  },
});

/* ------------------------------------------------------------ cross-validation, and leakage between batches */
const CV = (() => {
  const r = RNG(57), nb = 6, nt = 16;
  const curve = t => 1 / (1 + Math.exp(-(t - 5) / 1.3));
  const mk = () => { const x = r.u(0, 1), off = 0.09 * r.n(); return { x, ys: linspace(0, 12, nt).map(t => curve(t) * (0.6 + 0.8 * x) + off + 0.02 * r.n()) }; };
  const batches = Array.from({ length: nb }, mk), fresh = Array.from({ length: 30 }, mk);
  const ts = linspace(0, 12, nt), feat = (b, i) => [ts[i] / 12, b.x];
  const pts = batches.flatMap((b, j) => b.ys.map((y, i) => ({ f: feat(b, i), y, j, i })));
  const knn = (train, f) => { const d = train.map(p => [(p.f[0] - f[0]) ** 2 + (p.f[1] - f[1]) ** 2, p.y]).sort((a, c) => a[0] - c[0]).slice(0, 3); return mean(d.map(q => q[1])); };
  const rmse = (train, test) => Math.sqrt(mean(test.map(p => (knn(train, p.f) - p.y) ** 2)));
  const perm = pts.map((_, i) => i); for (let i = perm.length - 1; i > 0; i--) { const k = Math.floor(r() * (i + 1)); [perm[i], perm[k]] = [perm[k], perm[i]]; }
  const fold = Array(pts.length); perm.forEach((p, k) => { fold[p] = k % 5; });
  const rand = Math.sqrt(mean([0, 1, 2, 3, 4].map(k => rmse(pts.filter((_, i) => fold[i] !== k), pts.filter((_, i) => fold[i] === k)) ** 2)));
  const group = Math.sqrt(mean(Array.from({ length: nb }, (_, j) => rmse(pts.filter(p => p.j !== j), pts.filter(p => p.j === j)) ** 2)));
  const freshPts = fresh.flatMap(b => b.ys.map((y, i) => ({ f: feat(b, i), y })));
  return { nb, nt, ts, batches, fold, rand, group, newb: rmse(pts, freshPts) };
})();
Deck.insert('extrapolate', {
  id: 'cross-val', title: 'Cross-validation and leakage', beats: 3, dur: [5400, 5000, 7000], min: 4,
  notes: [
    'Six fermentation batches, each sampled 16 times over 12 hours: 96 rows of data. We want a model that predicts titre from time and the batch\'s seed density. With little data, one validation split is noisy, so we use k-fold cross-validation: cut the rows into five folds, train on four, validate on the fifth, rotate, and average. In the grid each row is a batch, each cell a sample, and amber marks the validation cells of one fold. Here the folds were drawn at random over all 96 rows, and the cross-validated error is ' + CV.rand.toFixed(3) + '. Excellent.',
    'Now the honest test: thirty brand-new batches the model has never seen. The error is ' + CV.newb.toFixed(3) + ', about ' + Math.round(CV.newb / CV.rand) + ' times worse. What went wrong? Leakage. With a random split, every validation sample has neighbours from the same batch in the training set, one sample earlier and one later, sharing that batch\'s particular quirks. The model only had to interpolate within a batch it already knew. That is not the question we will ask it in the plant.',
    'Split by batch instead: grouped cross-validation, here leave one batch out. Each fold holds out a whole batch, as the plant will present a whole new batch. Now the cross-validated error is ' + CV.group.toFixed(3) + ', matching the new-batch error. The rule is to split the way the model will be used: by batch, by campaign, by catalyst lot, or forwards in time for anything that drifts. In scikit-learn that is GroupKFold or TimeSeriesSplit. Random splits of time series and repeated measurements are the most common way published models look better than they are.',
  ],
  html: () => `${hdr('Chapter 6 · cross-validation', 'Split the way the model will be used')}
    ${at(1250, 290, '<div class="lbl">random 5-fold CV</div>', 0, 'rise')}
    ${at(1250, 322, `<div class="mono c-good" style="font:600 56px/1 var(--mono)">${CV.rand.toFixed(3)}</div>`, 0, 'rise')}
    ${at(1250, 430, '<div class="lbl">30 brand-new batches</div>', 1, 'rise')}
    ${at(1250, 462, `<div class="mono c-loss" style="font:600 56px/1 var(--mono)">${CV.newb.toFixed(3)}</div>`, 1, 'rise')}
    ${at(1250, 570, '<div class="lbl">leave-one-batch-out CV</div>', 2, 'rise', '--d:1600ms')}
    ${at(1250, 602, `<div class="mono c-data" style="font:600 56px/1 var(--mono)">${CV.group.toFixed(3)}</div>`, 2, 'rise', '--d:1600ms')}
    ${at(1250, 720, '<div class="lede" style="font-size:26px;width:560px">RMS error in titre (g/L). GroupKFold · TimeSeriesSplit</div>', 2, 'rise', '--d:2400ms')}
    ${sub('Random folds over all 96 samples: a superb validation score.', 0, 1)}
    ${sub('On new batches it is several times worse. <b>Neighbours leaked.</b>', 1, 2)}
    ${sub('Hold out whole batches: the CV score now tells the truth.', 2)}`,
  frame(g, S) {
    const a0 = S.at(0, 700), P = new Plot(150, 290, 960, 330, [0, 12], [-0.2, 1.6]);
    P.axes({ xt: [0, 4, 8, 12], yt: [0, 0.5, 1, 1.5], xl: 'time  (h)', yl: 'titre  (g/L)', a: a0 });
    const bcol = j => mixHex(COL.cls0, COL.grad, j / (CV.nb - 1));
    CV.batches.forEach((b, j) => { D.line(b.ys.map((y, i) => [P.X(CV.ts[i]), P.Y(y)]), bcol(j), 2.5, .8 * a0); b.ys.forEach((y, i) => D.dot(P.X(CV.ts[i]), P.Y(y), 4, bcol(j), a0)); });
    const gx = 150 + 60, gy = 720, cw = 52, ch = 26;
    let isVal;
    if (S.beat <= 1) { const fk = Math.floor(S.bt / 1100) % 5; isVal = (j, i) => CV.fold[j * CV.nt + i] === (S.beat === 0 ? fk : 0); }
    else { const fj = Math.floor(S.bt / 1000) % CV.nb; isVal = j => j === fj; }
    for (let j = 0; j < CV.nb; j++) {
      D.text('batch ' + (j + 1), gx - 14, gy + j * ch + ch / 2, { size: 18, align: 'right', col: bcol(j), a: a0 });
      for (let i = 0; i < CV.nt; i++) { const v = isVal(j, i); D.rect(gx + i * cw + 2, gy + j * ch + 2, cw - 4, ch - 4, v ? COL.data : '#1a2130', a0 * (v ? .95 : 1), null, 0, 3); }
    }
    D.text('■ validation  ■ training', gx + CV.nt * cw, gy - 18, { size: 18, align: 'right', col: COL.dim, a: a0 });
  },
});

/* ------------------------------------------------------------ ensembles: disagreement as a trust flag */
const ENS = (() => {
  const idx = HAL.S.map((s, i) => i).filter(i => HAL.S[i] <= 10), X = idx.map(i => [HAL.S[i] / 10 - 1]), Y = idx.map(i => HAL.Y[i] * 6);
  const nets = [11, 12, 13, 14, 15, 16, 17, 18].map(seed => { const n = new MLP([1, 16, 1], 'tanh', 'linear', seed, 1.5); for (let e = 0; e < 2500; e++) n.step(X, Y, 0.05, 0.9); return n; });
  const f = (k, s) => nets[k].predict([s / 10 - 1]) / 6;
  const stats = s => { const v = nets.map((_, k) => f(k, s)), m = mean(v); return [m, Math.sqrt(mean(v.map(q => (q - m) ** 2)))]; };
  return { f, stats, n: nets.length };
})();
Deck.insert('doctor', {
  id: 'ensemble', title: 'Ensembles: when to distrust a network', beats: 3, dur: [4400, 5200, 6000], min: 3.5,
  notes: [
    'We saw that networks extrapolate badly. Can a network at least tell us when it is extrapolating? One practical answer is an ensemble. Here is one network, 16 tanh neurons, trained on growth-rate data that only goes up to 10 grams per litre of substrate. Inside the data it is fine. Beyond 10 it confidently draws a line, and nothing in the single curve warns you.',
    'Train eight copies of the same network on the same data, each from a different random initialisation. Inside the data range they all agree: the data pin them down. Outside it, each one extrapolates in its own way, and they fan out. The disagreement is information: it measures how little the data constrain the answer there.',
    'Plot the ensemble mean and a band of two standard deviations. Narrow where we have data, widening fast beyond it, and the true kinetics, dashed, leaves the narrow part only where the band has already warned us. This is the simplest useful uncertainty estimate for a neural network, deep ensembles, and it is cheap: train five or ten times. Use it as a trust flag in deployment: when the members disagree, do not act on the prediction; collect data there. It is also exactly the signal Bayesian optimisation uses to decide where to experiment next.',
  ],
  html: () => `${hdr('Chapter 6 · uncertainty', 'When the copies disagree, do not trust it')}
    ${at(1250, 300, '<div class="lbl">one network</div><div class="lede" style="font-size:27px;margin-top:10px;width:560px">fine inside the data, silently wrong outside</div>', 0, 'rise', '', 1)}
    ${at(1250, 300, '<div class="lbl c-model">eight networks, eight random starts</div><div class="lede" style="font-size:27px;margin-top:10px;width:560px">agree where data pin them down, fan out where it does not</div>', 1, 'rise')}
    ${at(1250, 520, `<div class="eq-s">${M(T`\hat y \pm 2\,\operatorname{sd}\big(\hat y_1, \dots, \hat y_8\big)`)}</div><div class="lede" style="font-size:27px;margin-top:12px;width:560px">a wide band is a trust flag: do not act, get data</div>`, 2, 'rise')}
    ${sub('One network: confident everywhere, including where it has no data.', 0, 1)}
    ${sub('Eight random starts: <b>they agree inside the data and fan out beyond.</b>', 1, 2)}
    ${sub('The spread is a cheap, useful uncertainty. Wide band: get data.', 2)}`,
  frame(g, S) {
    const a0 = S.at(0, 700), P = new Plot(150, 300, 960, 520, [0, 20], [-0.1, 0.4]);
    P.axes({ xt: [0, 5, 10, 15, 20], yt: [0, 0.1, 0.2, 0.3], xl: 'substrate S  (g/L)', yl: 'growth rate μ  (1/h)', a: a0 });
    D.rect(P.X(10), P.y, P.X(20) - P.X(10), P.h, '#ffffff', .035 * a0);
    D.text('no data here', P.X(15), P.y + 24, { size: 22, align: 'center', col: COL.dim, a: a0 });
    const g0 = D.g; g0.save(); g0.beginPath(); g0.rect(P.x, P.y, P.w, P.h); g0.clip();
    if (S.beat === 2) {
      const a = S.at(2, 900), xs = linspace(0.2, 20, 120), st = xs.map(s => ENS.stats(s));
      D.poly([...xs.map((s, i) => [P.X(s), P.Y(st[i][0] + 2 * st[i][1])]), ...xs.map((s, i) => [P.X(s), P.Y(st[i][0] - 2 * st[i][1])]).reverse()], COL.model, .22 * a);
      D.line(xs.map((s, i) => [P.X(s), P.Y(st[i][0])]), COL.model, 4, a);
      D.fn(P, HAL.mu, 0.2, 20, '#ffffff', 2, .7 * a, 200, [8, 8]);
    } else {
      const m = S.beat === 0 ? 1 : 1 + Math.floor(S.at(1, 3200, 300) * (ENS.n - 1));
      for (let k = 0; k < m; k++) D.fn(P, s => ENS.f(k, s), 0.2, 20, COL.model, S.beat === 0 ? 4 : 2.5, (S.beat === 0 ? 1 : .75) * S.at(S.beat, 600), 200, null, S.beat === 0 ? 10 : 0);
    }
    g0.restore();
    HAL.S.forEach((s, i) => { if (s <= 10) D.dot(P.X(s), P.Y(HAL.Y[i]), 8, COL.data, a0, 8); });
  },
});
