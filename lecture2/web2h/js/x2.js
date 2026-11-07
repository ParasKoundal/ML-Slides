/* x2.js: extended cut, chapter 2. Newton's method (use the curvature, and when it fails); saddle points and
 * loss plateaus; learning-rate schedules (constant, warm-up + cosine) and the noise floor of SGD. */
'use strict';

/* ------------------------------------------------------------ Newton: a parabola at every step */
const KIN1 = (() => {
  const r = RNG(61), t = [0.5, 1, 1.5, 2, 3, 4, 5, 6, 8, 10], C = t.map(v => Math.exp(-0.35 * v) + 0.015 * r.n());
  const L = k => mean(t.map((v, i) => (C[i] - Math.exp(-k * v)) ** 2));
  const h = 1e-4, d1 = k => (L(k + h) - L(k - h)) / (2 * h), d2 = k => (L(k + h) - 2 * L(k) + L(k - h)) / (h * h);
  const gd = [0.04]; for (let s = 0; s < 50; s++) { const k = gd[gd.length - 1]; gd.push(k - 0.035 * d1(k)); }
  const newton = k0 => { const p = [k0]; for (let s = 0; s < 5; s++) { const k = p[p.length - 1]; p.push(k - d1(k) / d2(k)); } return p; };
  return { t, C, L, d1, d2, gd, nt: newton(0.04), bad: newton(1.15) };
})();
const PNW = new Plot(150, 270, 1000, 540, [0, 1.6], [0, 0.36]);
Deck.insert('conditioning', {
  id: 'newton', title: "Newton's method: use the curvature", beats: 3, dur: [6400, 6600, 6000], min: 3.5,
  notes: [
    'A real one-parameter problem. Ten samples of a reactant decaying by first-order kinetics, and we want the rate constant k that fits them. The loss against k is not a parabola: it is steep for small k and flattens out for large k, because once the reaction is fast every curve looks alike. Gradient descent from k equal to 0.04 has a problem we met in the step-size scene. Down at small k the curve is very steep, about forty times more curved than at the answer, so the step size has to be small to be stable there. Near the answer, where the curve is gentle, that same small step crawls. It takes 44 steps to get within 0.01 of the best k, 0.35 per hour.',
    'Newton\'s method uses one more piece of information: the curvature, the second derivative. At the current point it builds the parabola that has the same height, slope and curvature, the dashed curve, and jumps straight to the bottom of that parabola. For a true parabola that is one step, which is the eta times c equal to one case from before. Here it takes five, with no learning rate to tune, and near the end the number of correct digits roughly doubles every step. That is quadratic convergence.',
    'Two catches. First, start where the loss curves the wrong way, here at k equal to 1.15 beyond the inflection, and the local parabola opens downwards: Newton jumps to its top and walks off to infinity. Practical codes guard against this with a line search or a trust region. Second, in n dimensions the curvature is the n by n Hessian. For the 3 parameters of a kinetic model that is nothing; scipy\'s curve_fit and least_squares use exactly this family of methods, Gauss-Newton and Levenberg-Marquardt. For a network with a billion knobs the Hessian has 10 to the 18 entries, which is why deep learning lives on first-order methods.',
  ],
  html: () => `${hdr('Chapter 2 · second order', "Newton's method: use the curvature")}
    ${at(1250, 290, '<div class="lbl">fitting a first-order rate constant</div><div class="lede" style="font-size:28px;margin-top:10px;width:540px">ten samples of C<sub>A</sub>(t), model e<sup>−kt</sup>, loss = mean squared error</div>', 0)}
    ${at(1250, 470, '<div class="lbl c-model">gradient descent</div><div class="big" style="margin-top:6px">44 steps</div>', 0, 'rise', '--d:2500ms', 2)}
    ${at(1250, 600, '<div class="lbl c-grad">Newton</div><div class="big" style="margin-top:6px">5 steps</div>', 1, 'rise', '--d:4400ms', 2)}
    ${at(1250, 720, `<div class="eq-s">${M(T`k \leftarrow k - \frac{L'(k)}{\color{#b48cff}{L''(k)}}`)}</div>`, 1, 'rise', '--d:600ms')}
    ${at(1250, 470, '<div class="card" style="width:560px"><h4 class="c-loss">curvature the wrong way</h4><p>beyond the inflection the parabola opens downward: Newton climbs to its top and leaves</p></div>', 2, 'rise', '--d:2600ms')}
    ${sub('Gradient descent: the steep start caps the step, so it crawls near the answer.', 0, 1)}
    ${sub('Newton fits a parabola at each point and <b>jumps to its bottom.</b>', 1, 2)}
    ${sub('Negative curvature sends Newton the wrong way. Codes add a line search.', 2)}`,
  frame(g, S) {
    const P = PNW, a0 = S.at(0, 700);
    P.axes({ xt: [0, 0.4, 0.8, 1.2, 1.6], yt: [0, 0.1, 0.2, 0.3], xl: 'rate constant  k  (1/h)', yl: 'loss', grid: true, a: a0, xf: v => v.toFixed(1), yf: v => v.toFixed(1) });
    D.fn(P, KIN1.L, 0.028, 1.6, '#c8d1de', 3, a0);
    const ball = (k, col, hop = 0) => { const x = P.X(k), y = P.Y(KIN1.L(k)) - hop; D.dot(x, y - 12, 13, col, 1, 22); D.dot(x - 4, y - 16, 4, '#fff', .9); };
    if (S.beat === 0) {
      const q = S.at(0, 5400, 600) * 44, n = Math.floor(q);
      for (let j = 0; j < n; j++) D.seg(P.X(KIN1.gd[j]), P.Y(KIN1.L(KIN1.gd[j])), P.X(KIN1.gd[j + 1]), P.Y(KIN1.L(KIN1.gd[j + 1])), COL.model, 2.5, .7);
      const f = Ease.inOut(q - n), k = lerp(KIN1.gd[n], KIN1.gd[n + 1], f);
      ball(k, COL.model, Math.sin(f * Math.PI) * 18);
      return;
    }
    const path = S.beat === 1 ? KIN1.nt : KIN1.bad, per = S.beat === 1 ? 1150 : 1050, q = S.p(per * 5, 300) * 5, n = Math.min(4, Math.floor(q)), f = q - n;
    const k = path[n], L0 = KIN1.L(k), d1 = KIN1.d1(k), d2 = KIN1.d2(k), kn = path[n + 1];
    const quad = x => L0 + d1 * (x - k) + 0.5 * d2 * (x - k) ** 2;
    const span = Math.max(0.12, Math.abs(kn - k) * 1.5);
    const pa = Ease.out(seg(f, 0, .35)), jump = Ease.inOut(seg(f, .45, .9));
    const pts = []; for (let i = 0; i <= 80; i++) { const x = k - span + 2 * span * i / 80; const y = quad(x); if (y > -0.05 && y < 0.36 && x > 0 && x < 1.6) pts.push([P.X(x), P.Y(y)]); }
    D.line(pts, COL.grad, 3, .9 * pa, [10, 8]);
    for (let j = 0; j < n; j++) D.dot(P.X(path[j]), P.Y(KIN1.L(path[j])), 6, COL.grad, .7);
    const kk = lerp(k, kn, jump), off = kk > 1.6;
    if (off) { D.text('→ k = ∞', P.x + P.w - 10, P.Y(KIN1.L(1.6)) - 60, { size: 30, col: COL.loss, weight: 600, align: 'right' }); return; }
    D.seg(P.X(kn), P.Y(0), P.X(kn), P.Y(Math.max(0, quad(kn))), COL.grad, 1.5, .5 * pa, [4, 6]);
    ball(clamp(kk, 0, 1.6), S.beat === 1 ? COL.grad : COL.loss, Math.sin(jump * Math.PI) * 30);
  },
});

/* ------------------------------------------------------------ saddle points: flat, not stuck */
const SAD = (() => {   // a saddle at the origin, two true minima at v = ±0.58
  const f = (u, v) => 0.55 * u * u - 0.6 * v * v + 0.9 * v ** 4;
  const g = (u, v) => [1.1 * u, -1.2 * v + 3.6 * v ** 3];
  const run = v0 => { let p = [0.95, v0]; const path = [p], loss = [f(...p)]; for (let k = 0; k < 220; k++) { const gr = g(...p); p = [p[0] - 0.05 * gr[0], p[1] - 0.05 * gr[1]]; path.push(p); loss.push(f(...p)); } return { path, loss }; };
  const grid = linspace(-1, 1, 41).map(u => linspace(-1, 1, 41).map(v => f(u, v)));
  return { f, grid, A: run(0), B: run(0.0002) };
})();
Deck.insert('mini-batch', {
  id: 'saddle', title: 'Saddle points and plateaus', beats: 3, dur: [5600, 7200, 5400], min: 3.5,
  notes: [
    'A critical point is anywhere the gradient is zero. In one dimension that means a bottom or a top. In two or more there is a third kind: a saddle, a mountain pass, curving up one way and down the other. Start a ball exactly on the ridge line and plain gradient descent rolls down into the pass and stops. The gradient there is zero, so it has no reason to move. The loss curve on the right goes flat.',
    'Now nudge the start sideways by two ten-thousandths. For a long time nothing seems to happen: the ball creeps toward the pass and lingers there, and the loss curve shows a long plateau. Then the tiny sideways offset grows by a few percent every step, and the ball slides off into one of the two real valleys. The loss drops again. If you have seen a training curve sit still for a while and then suddenly fall, this is very often what was happening.',
    'Why this matters. To be a minimum, every one of the n curvatures, the Hessian eigenvalues, must be positive. With millions of knobs, it is overwhelmingly likely that at least one is negative, so most flat spots a network meets are saddles, not traps. Mini-batch noise and momentum are exactly the nudges that get you off them. The practical lesson: a plateau is not convergence. Check the gradient size, and be patient or change the learning rate before you conclude the model cannot learn more.',
  ],
  html: () => `${hdr('Chapter 2 · saddle points', 'A flat spot is not always the bottom')}
    ${at(1250, 640, `<div class="lbl">a minimum needs</div><div class="eq-s" style="margin-top:12px">${M(T`\nabla L = 0 \ \text{and all}\ \lambda_i(H) > 0`)}</div><div class="lede" style="font-size:27px;margin-top:12px;width:560px">with millions of knobs, almost every flat spot is a saddle</div>`, 2, 'rise')}
    ${sub('Start exactly on the ridge: the ball rolls into the pass and stops.', 0, 1)}
    ${sub('A tiny nudge: a long plateau, then it slides off. <b>Plateau ≠ converged.</b>', 1, 2)}
    ${sub('Mini-batch noise and momentum are the nudges that escape saddles.', 2)}`,
  enter(st) { st.cam = new Cam(640, 640, 380, -0.45, 0.62, 0.12); },
  frame(g, S) {
    const cam = S.st.cam; cam.yaw = -0.45 + S.t / 30000;
    drawSurface(cam, SAD.grid, { zscale: .6, zmin: -.12, zmax: .9, a: S.at(0, 800) });
    const z = (u, v) => SAD.f(u, v) * .6 + .015;
    const n = S.beat === 0 ? Math.floor(S.at(0, 4600, 600) * 220) : S.beat === 1 ? Math.floor(S.at(1, 6200, 400) * 220) : 220;
    const runs = S.beat === 0 ? [[SAD.A, COL.model]] : [[SAD.A, '#8792a3'], [SAD.B, COL.model]];
    for (const [R, col] of runs) {
      const m = S.beat === 0 || R === SAD.B ? n : 220;
      D.line(R.path.slice(0, m + 1).map(([u, v]) => cam.p(u, v, z(u, v)).slice(0, 2)), col, 2.5, .8);
      const [u, v] = R.path[m], [x, y] = cam.p(u, v, z(u, v)); D.dot(x, y - 8, 11, col, 1, 18);
    }
    const sp = cam.p(0, 0, .015); D.ring(sp[0], sp[1], 18, COL.data, 2, S.at(0, 600, 3000));
    D.text('saddle', sp[0] + 26, sp[1] - 30, { size: 22, col: COL.data, a: S.at(0, 600, 3000) });
    const P = new Plot(1250, 300, 560, 250, [0, 220], [-0.12, 0.52]), a = S.at(0, 700, 300);
    P.axes({ xt: [0, 100, 200], yt: [0, 0.4], xl: 'step', yl: 'loss', a, yf: v => v.toFixed(1) });
    D.seg(P.x, P.Y(0), P.x + P.w, P.Y(0), '#3a4352', 1, a, [4, 6]);
    for (const [R, col] of runs) { const m = S.beat === 0 || R === SAD.B ? n : 220; D.line(R.loss.slice(0, m + 1).map((L, k) => [P.X(k), P.Y(L)]), col, 3, a); }
    if (S.beat >= 1) D.text('plateau', P.X(80), P.Y(0) - 22, { size: 22, col: COL.data, align: 'center', a: S.at(1, 600, 2400) });
  },
});

/* ------------------------------------------------------------ learning-rate schedules */
const SCHED = (() => {
  const T = 400, eta = 0.06, warm = 25;
  const lrC = () => eta, lrS = k => (k < warm ? eta * (k + 1) / warm : eta * 0.5 * (1 + Math.cos(Math.PI * (k - warm) / (T - warm))));
  const run = (lr, seed) => {
    const r = RNG(seed); let p = [0.7, 3.7]; const ex = [FEED.L(...p) - FEED.Lmin];
    for (let k = 0; k < T; k++) {
      const idx = [Math.floor(r() * FEED.n), Math.floor(r() * FEED.n)];
      let gw = 0, gb = 0; idx.forEach(i => { const e = FEED.y[i] - (p[0] * FEED.x[i] + p[1]); gw += -2 * FEED.x[i] * e / 2; gb += -2 * e / 2; });
      p = [p[0] - lr(k) * gw, p[1] - lr(k) * gb]; ex.push(Math.max(1e-5, FEED.L(...p) - FEED.Lmin));
    }
    return ex;
  };
  // smooth each curve a little (running mean over 9 steps) so the trend is readable
  const smooth = a => a.map((_, i) => mean(a.slice(Math.max(0, i - 4), i + 5)));
  return { T, eta, warm, lrC, lrS, C: smooth(run(lrC, 4)), S: smooth(run(lrS, 4)) };
})();
Deck.insert('epochs', {
  id: 'schedule', title: 'Learning-rate schedules', beats: 3, dur: [5600, 5600, 5400], min: 3,
  notes: [
    'Back to stochastic gradient descent with mini-batches of two. With a constant learning rate, the loss falls quickly and then stops improving: it hovers at a noise floor, and never gets closer. The reason is the noise in each step: near the bottom, the random part of the gradient is as large as the true part, so the knobs jitter around the minimum in a cloud whose size is proportional to the learning rate.',
    'The cure is to shrink the step as training goes on. This is a cosine schedule: start at the same rate, then decay smoothly to zero over the run. Early on it moves as fast as the constant rate; late in training the jitter shrinks with the step, and the loss keeps falling: the cosine run ends more than a hundred times closer to the minimum, at no extra cost. Same data, same number of steps.',
    'Two more pieces you will see in every modern training recipe. Warm-up: the first few percent of steps ramp the learning rate up from almost zero, because at the start the gradients are large and erratic, and with Adam its running statistics are not yet reliable. Then decay, cosine or step-wise. Large language models are trained with exactly this, linear warm-up then cosine decay. For your own small models: if the loss flattens out noisily, try halving the learning rate before you change anything else.',
  ],
  html: () => `${hdr('Chapter 2 · schedules', 'Big steps early, small steps late')}
    ${at(1260, 270, '<div class="lbl">learning-rate schedule</div>', 0)}
    ${at(1260, 640, `<div class="eq-s">${M(T`\eta_t = \tfrac{\eta_{\max}}{2}\Big(1 + \cos\frac{\pi t}{T}\Big)`)}</div><div class="lede" style="font-size:27px;margin-top:12px;width:560px">after a short linear warm-up: the recipe used for LLMs</div>`, 2, 'rise')}
    ${sub('Constant step: fast at first, then stuck at a noise floor.', 0, 1)}
    ${sub('Decay the step: the jitter shrinks and <b>the loss keeps falling.</b>', 1, 2)}
    ${sub('Warm up, then decay. If the loss flattens noisily, halve η first.', 2)}`,
  frame(g, S) {
    const P = new Plot(150, 280, 960, 540, [0, 400], [1e-4, 10], { logy: true }), a0 = S.at(0, 700);
    P.axes({ xt: [0, 100, 200, 300, 400], yt: [1e-4, 1e-2, 1], yf: v => (v === 1 ? '1' : v === 1e-2 ? '0.01' : '10⁻⁴'), xl: 'SGD step  (batch of 2)', yl: 'loss − minimum  (log scale)', a: a0, grid: true });
    const nC = S.beat === 0 ? Math.floor(S.at(0, 4600, 500) * SCHED.T) : SCHED.T;
    D.line(SCHED.C.slice(0, nC + 1).map((v, k) => [P.X(k), P.Y(v)]), S.beat === 0 ? COL.loss : hexA(COL.loss, .55), 3);
    if (S.beat === 0 && nC > 250) D.text('noise floor', P.X(330), P.Y(0.6), { size: 22, col: COL.loss, align: 'center', a: seg(nC, 250, 320) });
    const Q = new Plot(1260, 320, 540, 200, [0, 400], [0, SCHED.eta * 1.15]);
    Q.axes({ xt: [0, 400], yt: [], xl: 'step', a: a0 });
    D.line(linspace(0, 400, 120).map(k => [Q.X(k), Q.Y(SCHED.lrC(k))]), COL.loss, 3, a0);
    if (S.beat >= 1) {
      const nS = S.beat === 1 ? Math.floor(S.at(1, 4600, 500) * SCHED.T) : SCHED.T, a1 = S.at(1, 600);
      D.line(SCHED.S.slice(0, nS + 1).map((v, k) => [P.X(k), P.Y(v)]), COL.good, 3.5, 1, null, 8);
      D.line(linspace(0, 399, 160).map(k => [Q.X(k), Q.Y(SCHED.lrS(k))]), COL.good, 3.5, a1);
      D.text('constant', Q.X(400), Q.Y(SCHED.eta) - 18, { size: 20, col: COL.loss, align: 'right', a: a1 });
      D.text('warm-up + cosine', Q.X(40), Q.Y(SCHED.eta * .22), { size: 20, col: COL.good, a: a1 });
    }
    if (S.beat === 2) { const a = S.at(2, 700, 300); D.rect(Q.X(0), Q.y, Q.X(SCHED.warm) - Q.X(0), Q.h, COL.data, .18 * a); D.text('warm-up', Q.X(SCHED.warm) + 8, Q.y + 16, { size: 18, col: COL.data, a }); }
  },
});
