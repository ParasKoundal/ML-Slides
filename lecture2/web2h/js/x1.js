/* x1.js: extended cut, chapter 1. Reading residuals (linear in the knobs, not in x); curvature of the
 * loss = how well the knobs are pinned down; collinear inputs and ridge regression. */
'use strict';

/* ------------------------------------------------------------ residuals tell you the model is wrong */
const CURV = (() => {
  const r = RNG(77), x = linspace(0.25, 4, 24).map(v => v + r.u(-.05, .05)), y = x.map(v => 6.6 * v / (0.9 + v) + 0.22 * r.n());
  const lin = lstsq(x.map(v => [1, v]), y), quad = lstsq(x.map(v => [1, v, v * v]), y);
  const fl = v => lin[0] + lin[1] * v, fq = v => quad[0] + quad[1] * v + quad[2] * v * v;
  const rl = x.map((v, i) => y[i] - fl(v)), rq = x.map((v, i) => y[i] - fq(v));
  return { x, y, fl, fq, rl, rq, mseL: mean(rl.map(e => e * e)), mseQ: mean(rq.map(e => e * e)) };
})();
Deck.insert('arrhenius', {
  id: 'residuals', title: 'Read the residuals', beats: 3, dur: [4400, 5200, 5600], min: 3,
  notes: [
    'A new data set: titre against feed rate over a wider range, 24 runs. The best straight line looks respectable. The mean squared error is small and most points are close. Is the model right?',
    'Plot what is left over: each residual, measured minus predicted, against feed rate. If the model has captured the process, residuals should look like noise: no pattern, centred on zero. Here they are negative at both ends and positive in the middle. That smile is the process telling you it saturates, and a straight line cannot saturate. Always plot residuals; the loss value alone will not show you this.',
    'The fix keeps us inside linear regression. Add x squared as a second input. The model is still linear in the knobs, w0 plus w1 x plus w2 x squared, so the same normal equations and the same gradient descent apply. The residuals are now patternless and the error drops by a factor of three. "Linear" in linear regression means linear in the parameters, not in x, which is why you can fit polynomials, logs and Arrhenius terms with the same machinery. Choosing those features by hand is exactly the job a neural network will learn to do for itself in chapter 4.',
  ],
  html: () => `${hdr('Chapter 1 · diagnostics', 'The residuals know what the loss hides')}
    ${at(1260, 280, `<div class="lbl">mean squared error</div><div class="num c-loss" style="font-size:96px;margin-top:6px">${fmt(CURV.mseL, 3)}</div>`, 0, 'rise', '', 2)}
    ${at(1260, 280, `<div class="lbl">mean squared error</div><div class="num c-good" style="font-size:96px;margin-top:6px">${fmt(CURV.mseQ, 3)}</div>`, 2, 'rise')}
    ${at(1260, 470, `<div class="eq-s">${M(T`\hat y = w_0 + w_1 x + \color{#ffb547}{w_2 x^2}`)}</div><div class="lede" style="margin-top:12px;font-size:27px;width:560px">still linear in the knobs: same normal equations, same gradient descent</div>`, 2, 'rise')}
    ${sub('A straight line, a small error. Is the model right?', 0, 1)}
    ${sub('Residuals with a pattern: <b>the process saturates, the line cannot.</b>', 1, 2)}
    ${sub('Add x² as a feature: linear in the knobs, not in x.', 2)}`,
  frame(g, S) {
    const P = new Plot(150, 255, 960, 335, [0, 4.2], [0, 6]), R = new Plot(150, 665, 960, 150, [0, 4.2], [-0.9, 0.9]);
    const a0 = S.at(0, 700);
    P.axes({ xt: [0, 1, 2, 3, 4], yt: [0, 2, 4, 6], xl: '', yl: 'titre  (g/L)', a: a0 });
    const q = S.beat === 2 ? Ease.inOut(S.at(2, 1600, 300)) : 0;
    const f = v => lerp(CURV.fl(v), CURV.fq(v), q), res = CURV.x.map((v, i) => lerp(CURV.rl[i], CURV.rq[i], q));
    D.fn(P, f, 0.1, 4.15, q > .5 ? COL.good : COL.model, 4, S.at(0, 1200, 300), 200, null, 12);
    CURV.x.forEach((v, i) => D.dot(P.X(v), P.Y(CURV.y[i]), 7, COL.data, a0, 6));
    if (S.beat >= 1) {
      const a = S.at(1, 700);
      R.axes({ xt: [0, 1, 2, 3, 4], yt: [-0.5, 0, 0.5], xl: 'glucose feed rate  (g L⁻¹ h⁻¹)', yl: 'residual', a });
      D.seg(R.x, R.Y(0), R.x + R.w, R.Y(0), '#3a4352', 1.5, a);
      CURV.x.forEach((v, i) => {
        const d = S.beat === 1 ? Ease.out(clamp(S.at(1, 2200, 300) * 1.6 - i * .025)) : 1;
        D.seg(R.X(v), R.Y(0), R.X(v), R.Y(res[i] * d), COL.loss, 3, a);
        D.dot(R.X(v), R.Y(res[i] * d), 5, COL.loss, a);
      });
      if (S.beat === 1) {
        const b = S.at(1, 900, 2600);
        D.fn(R, v => -0.62 + 1.05 * v - 0.25 * v * v, 0.2, 4.1, COL.data, 3, b, 120, [8, 6]);
        D.text('a smile, not noise', R.X(2.1), R.Y(0.75), { size: 24, col: COL.data, align: 'center', a: b });
      }
    }
  },
});

/* ------------------------------------------------------------ the curvature of the bowl is the uncertainty */
const BOOTF = (() => {
  const r = RNG(5), n = FEED.n, fits = [];
  for (let k = 0; k < 80; k++) {
    const idx = Array.from({ length: n }, () => Math.floor(r() * n));
    const c = lstsq(idx.map(i => [FEED.x[i], 1]), idx.map(i => FEED.y[i]));
    fits.push(c);
  }
  return fits;
})();
Deck.insert('gaussian', {
  id: 'curvature', title: 'A sharp bowl pins the knobs down', beats: 3, dur: [4600, 5200, 5400], min: 3,
  notes: [
    'How sure are we of the slope and intercept? Resample the 14 runs with replacement and refit, 80 times. That is the bootstrap, as in the Arrhenius example. Each faint line is one refit; they fan out most far from the middle of the data.',
    'Now plot each refit as a point on the loss surface, slope against intercept. The cloud is not round: it is a long thin ellipse lying exactly along the valley of the bowl. Where the bowl is steep, refits agree; along the flat valley floor, slope and intercept trade off against each other and the data cannot tell them apart.',
    'This is general and important. The curvature of the loss at its minimum, the Hessian, measures how much information the data hold about each combination of knobs. For least squares, the covariance of the estimates is sigma squared times X-transpose X inverse: the inverse of the curvature. Flat directions mean uncertain, correlated parameters. You will meet this again in kinetic parameter estimation, where pre-exponential factor and activation energy are notoriously correlated for exactly this reason.',
  ],
  html: () => `${hdr('Chapter 1 · uncertainty', 'A sharp bowl pins the knobs down')}
    ${at(1180, 700, `<div class="eq-s">${M(T`\mathrm{Cov}(\hat{\mathbf w}) \approx \sigma^2 (X^{\top}X)^{-1}`)}</div><div class="lede" style="margin-top:12px;font-size:27px;width:600px">curvature = information. Flat valley = uncertain, correlated knobs.</div>`, 2, 'rise')}
    ${sub('Bootstrap: resample the runs, refit, 80 times.', 0, 1)}
    ${sub('Each refit is a point on the bowl. <b>The cloud lies along the valley.</b>', 1, 2)}
    ${sub('Uncertainty is the inverse of curvature.', 2)}`,
  enter(st) {
    st.Q = new Plot(1180, 270, 600, 340, [0.6, 2.4], [-0.6, 2.8]);
    st.img = heatImage(st.Q, (w, b) => Math.log(FEED.L(w, b)), { res: 4, lo: Math.log(FEED.Lmin), hi: Math.log(FEED.Lmin) + 3, stops: ['#06131f', '#0b2c45', '#14506a', '#2a7d86', '#5fb3a1'] });
  },
  frame(g, S) {
    const a0 = S.at(0, 700);
    feedAxes(a0);
    const n = S.beat === 0 ? Math.floor(S.at(0, 3200, 300) * BOOTF.length) : BOOTF.length;
    BOOTF.slice(0, n).forEach(([w, b]) => feedLine(w, b, .12, COL.model));
    FEED.x.forEach((x, i) => D.dot(PF.X(x), PF.Y(FEED.y[i]), 9, COL.data, a0, 10));
    feedLine(FEED.w, FEED.b, a0);
    if (S.beat >= 1) {
      const Q = S.st.Q, a = S.at(1, 800);
      drawImg(S.st.img, Q, .95 * a);
      [0.3, 1, 2].forEach(l => drawSegs(contourSegs(Q, (w, b) => Math.log(FEED.L(w, b)) - Math.log(FEED.Lmin), l, 70, 40), '#7fd3c4', 1, .5 * a));
      Q.axes({ xt: [1, 1.5, 2], yt: [0, 1, 2], xl: 'slope w', yl: 'intercept b', a });
      const m = S.beat === 1 ? Math.floor(S.at(1, 3000, 400) * BOOTF.length) : BOOTF.length;
      BOOTF.slice(0, m).forEach(([w, b]) => D.dot(Q.X(w), Q.Y(b), 4, COL.data, .85 * a));
      D.dot(Q.X(FEED.w), Q.Y(FEED.b), 8, COL.fg, a, 10);
    }
  },
});

/* ------------------------------------------------------------ collinear inputs and ridge regression */
const RIDGE = (() => {
  const r = RNG(19), n = 20, X = [], y = [];
  for (let i = 0; i < n; i++) { const a = r.n(), b = a + 0.08 * r.n(); X.push([a, b]); y.push(1.0 * a + 1.0 * b + 0.5 * r.n()); }
  const fit = (idx, lam) => {
    const A = [[0, 0], [0, 0]], v = [0, 0];
    idx.forEach(i => { const [p, q] = X[i]; A[0][0] += p * p; A[0][1] += p * q; A[1][1] += q * q; v[0] += p * y[i]; v[1] += q * y[i]; });
    A[1][0] = A[0][1]; A[0][0] += lam; A[1][1] += lam;
    return solve(A, v);
  };
  const all = X.map((_, i) => i), boots = [];
  for (let k = 0; k < 50; k++) boots.push(Array.from({ length: n }, () => Math.floor(r() * n)));
  const L = (w1, w2, lam) => mean(X.map(([p, q], i) => (y[i] - w1 * p - w2 * q) ** 2)) + lam / n * (w1 * w1 + w2 * w2);
  return { X, y, n, fit, all, boots, L };
})();
Deck.insert('quiz-1', {
  id: 'ridge', title: 'Collinear inputs and ridge regression', beats: 3, dur: [4800, 5600, 5000], min: 3.5,
  notes: [
    'A common situation in plant data: two inputs that move together. Say the glucose feed is logged by two flowmeters, or feed rate and oxygen uptake are almost proportional. Here are the two knobs of a model with such a pair. The loss is not a bowl but a long, flat trough: increase one weight, decrease the other, and the fit hardly changes. Bootstrap the data and the least-squares answer jumps all over the trough, to weights like plus four and minus two. Each fit is fine on the data; the individual weights are meaningless.',
    'Ridge regression adds a small penalty on the size of the weights, lambda times the sum of squares. Geometrically, it adds a round bowl centred on zero to the trough. Watch lambda grow: the contours close up into an ellipse, and the bootstrap answers collapse onto one stable point, about one and one, the physically sensible split.',
    'In equations, X-transpose X gets lambda added to its diagonal, which makes it invertible and well conditioned. You trade a little bias, the weights shrink toward zero, for a large drop in variance. In neural networks the same idea is called weight decay, and it is one line in the optimizer. We will choose lambda the same way we choose everything else: by validation error.',
  ],
  html: () => `${hdr('Chapter 1 · regularisation', 'Collinear inputs: a trough, not a bowl')}
    ${at(1220, 300, '<div class="lbl">penalty λ</div><div id="rgl" class="num c-grad" style="font-size:96px;margin-top:6px"></div>', 1, 'rise')}
    ${at(1220, 560, `<div class="eq-s">${M(T`L = \tfrac1n\lVert y - Xw\rVert^2 + \color{#b48cff}{\tfrac{\lambda}{n}\lVert w\rVert^2}`)}</div><div class="eq-s" style="margin-top:20px">${M(T`w = (X^{\top}X + \color{#b48cff}{\lambda I})^{-1}X^{\top}y`)}</div><div class="lede" style="margin-top:12px;font-size:27px;width:560px">a little bias for a lot less variance. In networks: weight decay.</div>`, 2, 'rise')}
    ${sub('Two nearly identical inputs. Bootstrap fits wander along a flat trough.', 0, 1)}
    ${sub('Add a penalty on the weights: <b>the trough closes, the answers agree.</b>', 1, 2)}
    ${sub('Ridge regression: (XᵀX + λI) is always invertible.', 2)}`,
  frame(g, S) {
    const P = new Plot(150, 260, 900, 560, [-3, 5], [-3, 5]), a0 = S.at(0, 700);
    const lam = S.beat === 0 ? 0 : S.beat === 1 ? 12 * Ease.inOut(S.at(1, 3600, 400)) : 12;
    const Lmin = (() => { const w = RIDGE.fit(RIDGE.all, lam); return RIDGE.L(w[0], w[1], lam); })();
    const img = heatImage(P, (u, v) => Math.log(RIDGE.L(u, v, lam) - Lmin + .02), { res: 8, lo: Math.log(.02), hi: Math.log(.02) + 6.5, stops: ['#06131f', '#0b2c45', '#14506a', '#2a7d86', '#5fb3a1'] });
    drawImg(img, P, .95 * a0);
    [0.05, .2, .6, 1.5].forEach(l => drawSegs(contourSegs(P, (u, v) => RIDGE.L(u, v, lam) - Lmin, l, 60, 45), '#7fd3c4', 1, .45 * a0));
    P.axes({ xt: [-2, 0, 2, 4], yt: [-2, 0, 2, 4], xl: 'weight on flowmeter 1', yl: 'weight on flowmeter 2', a: a0 });
    const nb = S.beat === 0 ? Math.floor(S.at(0, 3000, 600) * RIDGE.boots.length) : RIDGE.boots.length;
    RIDGE.boots.slice(0, nb).forEach(idx => { const w = RIDGE.fit(idx, lam); if (Math.max(Math.abs(w[0] - 1), Math.abs(w[1] - 1)) > 3.9) return; D.dot(P.X(w[0]), P.Y(w[1]), 6, COL.data, .9 * a0, 6); });
    D.dot(P.X(1), P.Y(1), 9, COL.good, a0 * (S.beat >= 1 ? 1 : 0), 12);
    if (S.beat >= 1) D.text('true weights', P.X(1) + 16, P.Y(1) + 30, { size: 22, col: COL.good, a: S.at(1, 600) });
    const el = document.getElementById('rgl'); if (el) el.textContent = fmt(lam, 1);
  },
});
