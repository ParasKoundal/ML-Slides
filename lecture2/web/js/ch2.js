/* ch2.js: Rolling downhill. Gradient descent in 3-D, the gradient as the springs' net pull, the
 * learning rate as a stability limit, conditioning, non-convex landscapes and mini-batch noise. */
'use strict';

Deck.chapter('Rolling downhill', 'gradient descent, step size, conditioning, noise');

const PAR = (() => {          // the 1-D slice through the bowl at the best intercept
  const c = 2 * mean(FEED.x.map(v => v * v)), L = w => FEED.L(w, FEED.b), dL = w => FEED.grad(w, FEED.b)[0];
  return { c, L, dL, etaMax: 2 / c };
})();

chapterCard(2, 'Rolling downhill', 'gradient descent, step size, conditioning, noise',
  'Chapter two. We found the best line by letting springs pull it. A computer with a million knobs cannot see the whole surface. Here is what it does instead.',
  (g, S) => {
    const P = new Plot(1200, 300, 520, 360, [-0.4, 3.4], [0, 14]);
    D.fn(P, w => PAR.L(w), -0.4, 3.4, '#3a4352', 3, clamp(S.t / 500));
    let w = -0.3; const pts = [[w, PAR.L(w)]];
    for (let k = 0; k < Math.floor(S.p(2800, 400) * 8); k++) { w -= 0.055 * PAR.dL(w); pts.push([w, PAR.L(w)]); }
    D.line(pts.map(([a, b]) => [P.X(a), P.Y(b)]), COL.model, 2, .6, [6, 6]);
    D.dot(P.X(w), P.Y(PAR.L(w)) - 10, 11, COL.model, 1, 18);
  });

/* ------------------------------------------------------------ 1. feeling the slope in the fog */
const FOG_PATH = gdPath((a, c) => FEED.grad(a, c), [3.1, -1.2], 0.085, 60);
Deck.add({
  id: 'fog', title: 'Feeling the slope in the fog', beats: 3, dur: [4600, 7600, 4000], min: 3,
  notes: [
    'We drew the whole surface in chapter one, but that cost us 1,600 loss evaluations for two knobs. With a million knobs you can never draw it. The computer is like a hiker in thick fog: it cannot see the valley, only the ground right under its feet. What it can measure there is the slope: which way is uphill and how steep. That violet arrow is the gradient, the direction of steepest ascent.',
    'So it steps the other way, downhill, by a small amount. Then it feels the slope again at the new spot and repeats. Watch the loss fall with each step. Notice the steps shrink on their own as the ground flattens, because the slope itself shrinks.',
    'That is the entire learning rule. Theta is every knob stacked into one vector. Subtract the learning rate eta times the gradient. Every model in this lecture series, including GPT, is trained with this line or a close cousin of it.',
  ],
  html: () => `${hdr('Chapter 2', 'In the fog, all you can feel is the slope')}
    ${at(1460, 250, '<div class="lbl">step</div><div id="fk" class="mono" style="font-size:54px;margin:6px 0 18px">0</div><div class="lbl">loss</div><div id="fl" class="mono c-loss" style="font-size:54px;margin-top:6px"></div>', 0, 'rise')}
    ${at(1240, 700, `<div class="eq">${M(T`\color{#3fd0ff}{\theta} \leftarrow \color{#3fd0ff}{\theta} - \eta\,\color{#b48cff}{\nabla L(\theta)}`)}</div>`, 2, 'pop')}
    ${sub('The <b style="color:var(--grad)">gradient</b> points uphill. Step the other way.', 0, 1)}
    ${sub('Step, feel the slope again, step. The steps shrink as the ground flattens.', 1, 2)}
    ${sub('That one line trains every model in this series.', 2)}`,
  enter(st) { st.cam = new Cam(800, 700, 320, -0.75, 0.62, 0.14); },
  frame(g, S) {
    const st = S.st, cam = st.cam; cam.yaw = -0.75 + 0.12 * Math.sin(S.t / 5000);
    let k = 0, f = 0;
    if (S.beat === 1) { const q = S.p(7200, 300) * 24; k = Math.floor(q); f = Ease.inOut(clamp((q - k) * 1.6)); }
    else if (S.beat === 2) { k = 24; }
    const a = FOG_PATH[k], bnext = FOG_PATH[k + 1];
    const w = lerp(a[0], bnext[0], f), b = lerp(a[1], bnext[1], f);
    const u = SURF.uOf(w), v = SURF.vOf(b);
    drawSurface(cam, SURF.grid, { zscale: 1, zmin: 0, zmax: 1.1, spot: [u, v, S.beat === 2 ? 2 : .32, .06] });
    const P3 = (ww, bb, dz = .02) => { const q = cam.p(SURF.uOf(ww), SURF.vOf(bb), SURF.zOf(FEED.L(ww, bb)) + dz); return [q[0], q[1]]; };
    if (k > 0) D.line(FOG_PATH.slice(0, k + 1).map(([ww, bb]) => P3(ww, bb)).concat([P3(w, b)]), COL.model, 3, .8, [2, 7]);
    FOG_PATH.slice(1, k + 1).forEach(([ww, bb]) => D.dot(...P3(ww, bb), 4, COL.model, .7));
    // gradient arrow: uphill in violet, a short dashed downhill hint in cyan
    const [gw, gb] = FEED.grad(w, b), gu = gw * 2, gv = gb * 3.5, gn = Math.hypot(gu, gv) || 1;
    const len = .28 * clamp(gn / 6, .35, 1), pa = S.beat === 0 ? S.at(0, 900, 900) : 1 - f * .8;
    const tipU = u + gu / gn * len, tipV = v + gv / gn * len;
    const z0 = SURF.zOf(FEED.L(w, b)), zt = z0 + (SURF.zOf(FEED.L(SURF.wOf(tipU), SURF.bOf(tipV))) - z0);
    const p0 = cam.p(u, v, z0 + .03), p1 = cam.p(tipU, tipV, zt + .03);
    D.arrow(p0[0], p0[1], p1[0], p1[1], COL.grad, 5, 22, pa, 12);
    const p2 = cam.p(u - gu / gn * len, v - gv / gn * len, SURF.zOf(FEED.L(SURF.wOf(u - gu / gn * len), SURF.bOf(v - gv / gn * len))) + .03);
    D.line([[p0[0], p0[1]], [p2[0], p2[1]]], COL.model, 3, pa * .8, [8, 8]);
    const [x, y] = P3(w, b, .04 + Math.sin(f * Math.PI) * .12);
    D.dot(x, y, 13, COL.model, 1, 24); D.dot(x - 3, y - 4, 4, '#fff', .9);
    const ek = document.getElementById('fk'), el = document.getElementById('fl');
    if (ek) ek.textContent = String(k); if (el) el.textContent = fmt(FEED.L(w, b), 3);
  },
});

/* ------------------------------------------------------------ 2. the gradient is the springs' net pull */
const PULL_PATH = gdPath((a, c) => FEED.grad(a, c), [0.25, 4.6], 0.12, 60);
Deck.add({
  id: 'pull', title: 'The gradient is a vote of the misses', beats: 3, dur: [4000, 4000, 6000], min: 2.5,
  notes: [
    'What is that slope, concretely? Back to the springs. Each batch pulls on the line: up if the line is below it, down if above, harder the further away it is. These violet arrows are those pulls.',
    'Add them up and you get the gradient. The derivative with respect to b is minus two over n times the sum of residuals: the net vertical force. The derivative with respect to w weights each residual by its x: the net torque. Batches far to the right have more leverage on the slope. So the gradient is literally a vote of the misses, weighted by the inputs.',
    'Gradient descent moves the line a little in the direction of the vote, then everyone votes again. At the best line the votes cancel exactly, the gradient is zero, and the line stops moving.',
  ],
  html: () => `${hdr('Chapter 2', 'The gradient is a vote of the misses')}
    ${at(1250, 290, `<div class="eq-s">${M(T`\frac{\partial L}{\partial \color{#3fd0ff}{b}} = -\frac{2}{n}\sum_i \color{#ff4d7d}{r_i}`)}</div><div class="lbl" style="margin:8px 0 26px">net vertical pull</div><div class="eq-s">${M(T`\frac{\partial L}{\partial \color{#3fd0ff}{w}} = -\frac{2}{n}\sum_i x_i\,\color{#ff4d7d}{r_i}`)}</div><div class="lbl" style="margin-top:8px">net torque: far batches have leverage</div>`, 1, 'rise')}
    ${at(1250, 690, '<div class="lbl">gradient now</div><div id="gv" class="mono c-grad" style="font-size:40px;margin-top:10px"></div>', 1, 'rise', '--d:400ms')}
    ${sub('Each batch pulls the line toward itself.', 0, 1)}
    ${sub('Add the pulls: that sum is the gradient.', 1, 2)}
    ${sub('Step with the vote, vote again. At the best line the votes cancel.', 2)}`,
  frame(g, S) {
    const k = S.beat === 2 ? Math.floor(Ease.inOut(S.p(5400, 300)) * 60) : 0;
    const [w, b] = PULL_PATH[k];
    feedAxes(); feedPoints();
    const pa = S.at(0, 1200, 300);
    FEED.x.forEach((x, i) => {
      const yl = w * x + b, r = FEED.y[i] - yl, q = Ease.out(clamp(pa * 1.5 - i * .03));
      D.arrow(PF.X(x), PF.Y(yl), PF.X(x), PF.Y(yl + r * .55 * q), COL.grad, 4, 16, q > 0 ? 1 : 0, 8);
    });
    feedLine(w, b);
    const [gw, gb] = FEED.grad(w, b), el = document.getElementById('gv');
    if (el) el.textContent = `∂L/∂w = ${fmt(gw)}   ∂L/∂b = ${fmt(gb)}`;
  },
});

/* ------------------------------------------------------------ 3. step size is a stability limit */
const ETAS = [[0.1, 'crawls', COL.dim], [1.0, 'one jump', COL.good], [1.8, 'rings', COL.data], [2.1, 'blows up', COL.loss]];
function etaPath(r, n = 12) { let w = -0.25; const p = [w]; for (let k = 0; k < n; k++) { w -= r * PAR.etaMax / 2 * PAR.dL(w); p.push(w); } return p; }
Deck.add({
  id: 'step-size', title: 'The learning rate is a stability limit', beats: 4, dur: [4400, 4400, 4400, 4800], min: 3,
  notes: [
    'The learning rate eta is the dial you will touch most. Slice the bowl along w so we can see it in one dimension: a parabola with curvature c, here about ' + PAR.c.toFixed(1) + '. With a small step, eta times c equal to 0.1, the ball crawls. Safe, slow.',
    'Make eta times c equal to one and it lands at the bottom in a single jump. For a parabola that is the perfect step.',
    'Push to 1.8 and every step overshoots to the other wall, but by a little less each time. It rings, like an underdamped loop, and still converges.',
    'Past 2 it blows up: each overshoot is bigger than the last. Here is why. Near the minimum the error obeys e new equals one minus eta c times e old. That is a discrete-time system with gain one minus eta c, and it is stable only if that gain is less than one in magnitude: eta less than two over c. Process control people: this is exactly a proportional controller with too much gain. In many dimensions c is the largest eigenvalue of the Hessian, so the steepest direction sets the speed limit for every direction.',
  ],
  html: () => `${hdr('Chapter 2', 'Step size is a stability limit')}
    ${ETAS.map(([r, lbl, c], k) => at(1240, 280, `<div class="lbl"><span style="text-transform:none">η</span> × curvature</div><div class="num" style="color:${c};font-size:120px;margin-top:8px">${r.toFixed(1)}</div><div class="h3" style="color:${c};margin-top:6px">${lbl}</div>`, k, 'pop', '', k < 3 ? k + 1 : undefined)).join('')}
    ${at(1240, 640, `<div class="eq-s">${M(T`e_{k+1} = (1 - \eta\,c)\,e_k`)}</div><div class="eq-s" style="margin-top:20px">${M(T`\text{stable} \iff 0 < \eta < \frac{2}{\lambda_{\max}}`)}</div>`, 3, 'rise', '--d:600ms')}
    ${sub('Too small: safe and slow.', 0, 1)}${sub('Just right: one jump to the bottom.', 1, 2)}${sub('Too big: it rings, like an under-damped loop.', 2, 3)}${sub('Past the limit, every step overshoots further. <b>Too much gain.</b>', 3)}`,
  frame(g, S) {
    const P = new Plot(150, 270, 920, 560, [-0.6, 3.6], [0, 14]);
    P.axes({ xt: [0, 1, 2, 3], yt: [0, 7, 14], xl: 'slope of the line  w', yl: 'loss', grid: true });
    D.fn(P, PAR.L, -0.6, 3.6, '#c8d1de', 3);
    D.seg(P.X(FEED.w), P.y, P.X(FEED.w), P.y + P.h, COL.good, 1.5, .4, [6, 6]);
    const [r, , c] = ETAS[S.beat], path = etaPath(r);
    const q = S.p(4000, 200) * 12, k = Math.floor(q), f = Ease.inOut(clamp((q - k) * 1.4));
    for (let j = 0; j < Math.min(k, 12); j++) {
      const a = path[j], b = path[j + 1];
      D.line([[P.X(a), P.Y(PAR.L(a))], [P.X(b), P.Y(PAR.L(b))]].map(([x, y]) => [clamp(x, P.x - 40, P.x + P.w + 40), clamp(y, P.y - 10, P.y + P.h)]), c, 2.5, .8);
      D.dot(P.X(a), P.Y(PAR.L(a)), 5, c, .8);
    }
    const a = path[Math.min(k, 12)], b = path[Math.min(k + 1, 12)], wnow = lerp(a, b, k >= 12 ? 0 : f);
    const hop = k >= 12 ? 0 : Math.sin(f * Math.PI) * 50;
    const x = clamp(P.X(wnow), P.x - 30, P.x + P.w + 30), y = clamp(P.Y(PAR.L(wnow)) - hop, P.y - 10, P.y + P.h);
    if (P.Y(PAR.L(wnow)) < P.y - 10) D.text('→ ∞', x + 26, y - 12, { size: 34, col: COL.loss, weight: 600 });
    D.dot(x, y - 12, 14, COL.model, 1, 24); D.dot(x - 4, y - 16, 4, '#fff', .9);
  },
});

/* ------------------------------------------------------------ 4. conditioning: raw units vs standardised */
function condRun(C, eta, start = [0.92, 0.85], tol = 1e-4, max = 3000) {
  let p = start.slice(); const path = [p.slice()];
  for (let k = 0; k < max; k++) { const gr = C.G(...p); p = [p[0] - eta * gr[0], p[1] - eta * gr[1]]; path.push(p.slice()); if (C.L(...p) < tol) break; }
  return path;
}
const COND_RUNS = { raw: condRun(COND.raw, 1.6 / COND.raw.lam[0]), std: condRun(COND.std, 1.6 / COND.std.lam[0]) };
const PC1 = new Plot(150, 300, 760, 480, [-1, 1], [-1, 1]), PC2 = new Plot(1010, 300, 760, 480, [-1, 1], [-1, 1]);
Deck.add({
  id: 'conditioning', title: 'Units matter: conditioning', beats: 3, dur: [4400, 6200, 4600], min: 3,
  notes: [
    'Now the most common reason gradient descent is slow in engineering data. Suppose the inputs are temperature in degrees, with a spread of about 2.5, and pH, with a spread of 0.2. The loss bowl inherits those scales: it is about 156 times steeper along the temperature weight than along the pH weight. On the left is that bowl seen from above, a long narrow trench. On the right, the same model after standardising both inputs to mean zero and unit spread: nearly round.',
    'Run gradient descent on both with the largest safe step. The steep direction forces a tiny step, so along the flat direction it crawls: hundreds of steps. On the round bowl it is done in a handful.',
    'The rule of thumb: the number of steps grows with the condition number kappa, the ratio of the largest to the smallest curvature. Standardising inputs is the cheapest speed-up in machine learning. It also makes the weights comparable. Lecture 3 adds momentum and Adam, which partly fix this automatically.',
  ],
  html: () => `${hdr('Chapter 2 · conditioning', 'Mixed units stretch the bowl')}
    ${at(150, 250, '<div class="lbl">raw units · T in °C, pH</div>', 0)}
    ${at(1010, 250, '<div class="lbl">standardised · (x − mean) / sd</div>', 0, 'rise', '--d:200ms')}
    ${at(150, 800, `<div class="big">κ = <span class="c-loss mono">${COND.raw.kappa.toFixed(0)}</span> &nbsp; <span class="c-dim">steps:</span> <span id="n1" class="mono"></span></div>`, 0, 'rise')}
    ${at(1010, 800, `<div class="big">κ = <span class="c-good mono">${COND.std.kappa.toFixed(2)}</span> &nbsp; <span class="c-dim">steps:</span> <span id="n2" class="mono"></span></div>`, 0, 'rise', '--d:200ms')}
    ${sub('Same data, same model. One bowl is 156× steeper one way.', 0, 1)}
    ${sub('The steep wall caps the step, so the flat direction crawls.', 1, 2)}
    ${sub('Steps grow with κ. <b>Standardise your inputs.</b>', 2)}`,
  enter(st) {
    const map = L => Math.log10(L + 1e-4) / 1.2 + 1.1;
    st.h1 = heatImage(PC1, COND.raw.L, { res: 4, map: v => clamp(map(v) / 2.2), bands: 14, stops: ['#081426', '#0e2c48', '#13506b', '#1f7f86', '#47a99a', '#9fd3a4'] });
    st.h2 = heatImage(PC2, COND.std.L, { res: 4, map: v => clamp(map(v) / 2.2), bands: 14, stops: ['#081426', '#0e2c48', '#13506b', '#1f7f86', '#47a99a', '#9fd3a4'] });
  },
  frame(g, S) {
    const st = S.st;
    drawImg(st.h1, PC1, S.at(0, 800)); drawImg(st.h2, PC2, S.at(0, 800, 200));
    const run = (P, path, id) => {
      const n = S.beat === 0 ? 0 : S.beat === 1 ? Math.floor(Math.min(1, S.p(5600, 200)) ** 1.6 * (path.length - 1)) : path.length - 1;
      const pts = path.slice(0, n + 1).map(([a, b]) => [P.X(a), P.Y(b)]);
      D.line(pts, '#ffffff', 1.6, .85);
      const [x, y] = pts[pts.length - 1]; D.dot(x, y, 10, COL.model, 1, 18);
      D.dot(P.X(0), P.Y(0), 7, COL.good, .9);
      const el = document.getElementById(id); if (el) el.textContent = String(n);
    };
    run(PC1, COND_RUNS.raw, 'n1'); run(PC2, COND_RUNS.std, 'n2');
    D.text('weight on temperature', PC1.x + PC1.w / 2, PC1.y + PC1.h + 22, { size: 20, align: 'center' });
    D.text('weight on temperature', PC2.x + PC2.w / 2, PC2.y + PC2.h + 22, { size: 20, align: 'center' });
    D.text('weight on pH', PC1.x - 16, PC1.y + PC1.h / 2, { size: 20, align: 'center', rot: -Math.PI / 2 });
    D.text('weight on pH', PC2.x - 16, PC2.y + PC2.h / 2, { size: 20, align: 'center', rot: -Math.PI / 2 });
  },
});

/* ------------------------------------------------------------ 5. non-convex landscapes */
Deck.add({
  id: 'non-convex', title: 'One valley, or many?', beats: 2, dur: [7000, 4200], min: 2,
  notes: [
    'Linear and logistic regression have bowl-shaped losses: one bottom, and downhill always finds it. Neural networks do not. Here is a bumpy landscape. Drop balls from a grid of starting points and let each one descend. They end up in different valleys depending on where they started. Downhill is a local rule.',
    'Is that a disaster? Mostly not, for large networks. In very high dimensions most critical points are saddles rather than bad local minima, and the minima that training finds tend to have similar loss. What matters more in practice is the step size, the noise from mini-batches, and where you start. Initialisation is part of Lecture 3.',
  ],
  html: () => `${hdr('Chapter 2', 'Networks have many valleys')}
    ${sub('Different starts, different valleys. Downhill is local.', 0, 1)}
    ${sub('In high dimensions: mostly saddles, and many equally good valleys.', 1)}`,
  enter(st) {
    st.cam = new Cam(960, 700, 410, -0.6, 0.66, 0.12);
    st.balls = []; for (let i = 0; i < 5; i++) for (let j = 0; j < 5; j++) st.balls.push({ u: -0.85 + i * .425, v: -0.85 + j * .425, trail: [] });
  },
  frame(g, S) {
    const st = S.st, cam = st.cam; cam.yaw = -0.6 + S.t / 14000;
    drawSurface(cam, BUMPY_GRID, { zscale: .5, zmin: -.35, zmax: 1.25 });
    const go = S.t > 900;
    for (const b of st.balls) {
      if (go && b.trail.length < 600) { for (let k = 0; k < 2; k++) { const [gu, gv] = BUMPY.g(b.u, b.v); b.u -= .012 * gu; b.v -= .012 * gv; } b.trail.push([b.u, b.v]); }
      const pts = b.trail.map(([u, v]) => { const q = cam.p(u, v, BUMPY.f(u, v) * .5 + .01); return [q[0], q[1]]; });
      D.line(pts, COL.model, 2, .5);
      const [x, y] = cam.p(b.u, b.v, BUMPY.f(b.u, b.v) * .5 + .03);
      D.dot(x, y, 8, COL.model, 1, 14);
    }
  },
});

/* ------------------------------------------------------------ 6. mini-batches: noisy, unbiased, cheap */
const MB = (() => {
  const r = RNG(77), n = FEED.n, theta = [0.7, 3.7];
  const gradOn = (idx, w, b) => { let gw = 0, gb = 0; idx.forEach(i => { const rr = FEED.y[i] - (w * FEED.x[i] + b); gw += -2 * FEED.x[i] * rr; gb += -2 * rr; }); return [gw / idx.length, gb / idx.length]; };
  const sample = B => { const p = [...Array(n).keys()]; for (let i = n - 1; i > 0; i--) { const j = Math.floor(r() * (i + 1)); [p[i], p[j]] = [p[j], p[i]]; } return p.slice(0, B); };
  const clouds = {}; for (const B of [2, 7]) clouds[B] = Array.from({ length: 50 }, () => gradOn(sample(B), ...theta));
  const sgd = [theta.slice()]; let p = theta.slice();
  for (let k = 0; k < 160; k++) { const g = gradOn(sample(2), ...p); p = [p[0] - .05 * g[0], p[1] - .05 * g[1]]; sgd.push(p.slice()); }
  const full = gdPath((a, c) => FEED.grad(a, c), theta, .05, 160);
  return { theta, clouds, sgd, full, full0: FEED.grad(...theta) };
})();
const PMB = new Plot(150, 270, 900, 580, [-0.4, 2.6], [-0.5, 5.5]);
Deck.add({
  id: 'mini-batch', title: 'Mini-batches: noisy but cheap', beats: 3, dur: [4600, 3600, 6400], min: 3,
  notes: [
    'In real problems n is not 14 but millions, and computing the exact gradient means a pass over all of them for every single step. Instead, take a random mini-batch, here just 2 of the 14 batches, and compute the gradient on those. Each thin violet arrow is one such estimate. They scatter, but they scatter around the true gradient, the thick arrow. The estimate is unbiased.',
    'Use 7 batches instead of 2 and the cloud tightens. The spread of the estimate shrinks like one over the square root of the batch size: four times the batch, half the noise.',
    'Now run it. Full-batch gradient descent, dashed, walks smoothly down. Stochastic gradient descent, the bright jittery path, staggers but heads the same way, and each of its steps cost one seventh as much. In practice that noise is even helpful: it shakes the model out of sharp, narrow valleys.',
  ],
  html: () => `${hdr('Chapter 2 · stochastic gradient descent', 'A few batches give a noisy, honest slope')}
    ${at(1180, 300, '<div class="lbl">mini-batch size</div><div id="bs" class="num c-grad" style="margin-top:10px">2</div><div class="lede" style="font-size:27px;margin-top:10px">of 14 batches per step</div>', 0, 'rise')}
    ${at(1180, 640, `<div class="eq-s">${M(T`\mathbb{E}\big[\color{#b48cff}{\hat g_B}\big] = \color{#b48cff}{\nabla L},\qquad \operatorname{sd} \propto \tfrac{1}{\sqrt{B}}`)}</div>`, 1, 'rise')}
    ${sub('Each thin arrow: the slope from 2 random batches. They centre on the truth.', 0, 1)}
    ${sub('Bigger batch, tighter cloud: noise falls like 1/√B.', 1, 2)}
    ${sub('SGD staggers, but each step is 7× cheaper. Same valley.', 2)}`,
  enter(st) { st.heat = heatImage(PMB, (w, b) => FEED.L(w, b), { res: 4, map: v => clamp(Math.log10(v / FEED.Lmin) / 2.2), bands: 12, stops: ['#081426', '#0e2c48', '#13506b', '#1f7f86', '#47a99a', '#9fd3a4'] }); },
  frame(g, S) {
    drawImg(S.st.heat, PMB, .9);
    D.text('slope  w', PMB.x + PMB.w, PMB.y + PMB.h + 24, { size: 20, align: 'right' }); D.text('intercept  b', PMB.x, PMB.y - 22, { size: 20 });
    const [tw, tb] = MB.theta, sc = .14, X0 = PMB.X(tw), Y0 = PMB.Y(tb);
    const tip = ([gw, gb]) => [PMB.X(tw - sc * gw), PMB.Y(tb - sc * gb)];
    const el = document.getElementById('bs');
    if (S.beat <= 1) {
      const B = S.beat === 0 ? 2 : 7, cl = MB.clouds[B], n = Math.floor(S.at(S.beat, 2400, 400) * cl.length);
      if (el) el.textContent = String(B);
      cl.slice(0, n).forEach(gr => D.arrow(X0, Y0, ...tip(gr), COL.grad, 1.6, 10, .55));
      D.arrow(X0, Y0, ...tip(MB.full0), '#ffffff', 5, 22, S.at(0, 600), 10);
      D.dot(X0, Y0, 10, COL.model, 1, 16);
    } else {
      if (el) el.textContent = '2';
      const n = Math.floor(S.p(6000, 300) * 160);
      D.line(MB.full.slice(0, n + 1).map(([a, b]) => [PMB.X(a), PMB.Y(b)]), '#ffffff', 2.5, .8, [8, 8]);
      D.line(MB.sgd.slice(0, n + 1).map(([a, b]) => [PMB.X(a), PMB.Y(b)]), COL.model, 2.5, 1);
      const [a, b] = MB.sgd[n]; D.dot(PMB.X(a), PMB.Y(b), 10, COL.model, 1, 18);
      D.dot(PMB.X(FEED.w), PMB.Y(FEED.b), 8, COL.good, 1, 10);
    }
  },
});

/* ------------------------------------------------------------ 7. epochs, batches, steps */
const EP_PERM = (() => { const r = RNG(5), p = [...Array(10000).keys()]; for (let i = p.length - 1; i > 0; i--) { const j = Math.floor(r() * (i + 1)); [p[i], p[j]] = [p[j], p[i]]; } return p; })();
Deck.add({
  id: 'epochs', title: 'Epochs, batches and steps', beats: 1, dur: 7600, min: 1.5,
  notes: 'Vocabulary you will see in every training log. Suppose a historian database of 10,000 batch records. Shuffle them, cut them into mini-batches of 32, and take one gradient step per mini-batch. One pass through all the data is an epoch: 10,000 over 32, so 313 steps. Training for 50 epochs means about 15,650 steps, and each record was seen 50 times. Watch the records light up 32 at a time.',
  html: () => `${hdr('Chapter 2 · vocabulary', 'One epoch = every record seen once')}
    ${at(1150, 300, '<div class="lbl">records</div><div class="mono" style="font-size:54px;margin:6px 0 22px">10,000</div><div class="lbl">mini-batch size</div><div class="mono c-grad" style="font-size:54px;margin:6px 0 22px">32</div><div class="lbl">step</div><div id="es" class="mono c-model" style="font-size:54px;margin:6px 0 22px"></div><div class="lbl">50 epochs</div><div class="mono" style="font-size:40px;margin-top:6px">≈ 15,650 steps</div>', 0, 'rise')}
    ${sub('Shuffle, slice into batches of 32, one step per batch: 313 steps per epoch.', 0)}`,
  frame(g, S) {
    const x0 = 180, y0 = 270, pitch = 6.2, steps = Math.floor(Ease.inOut(S.p(6800, 400)) * 313);
    g.save();
    g.fillStyle = '#1b2230'; for (let i = 0; i < 10000; i++) g.fillRect(x0 + (i % 100) * pitch, y0 + Math.floor(i / 100) * pitch, 4.4, 4.4);
    g.fillStyle = hexA(COL.model, .45);
    for (let s = 0; s < steps; s++) for (let k = 0; k < 32; k++) { const i = EP_PERM[s * 32 + k]; if (i === undefined) continue; g.fillRect(x0 + (i % 100) * pitch, y0 + Math.floor(i / 100) * pitch, 4.4, 4.4); }
    g.fillStyle = COL.data; g.shadowColor = COL.data; g.shadowBlur = 8;
    for (let k = 0; k < 32; k++) { const i = EP_PERM[Math.min(312, steps) * 32 + k]; if (i === undefined) continue; g.fillRect(x0 + (i % 100) * pitch - 1, y0 + Math.floor(i / 100) * pitch - 1, 6.4, 6.4); }
    g.restore();
    const el = document.getElementById('es'); if (el) el.textContent = `${Math.min(313, steps + 1)} / 313`;
  },
});

quizScene('quiz-2', 'Chapter 2', [
  ['Your training loss jumps around, then becomes NaN. What do you change first?', 'Lower the learning rate. Once η·λmax > 2, every step overshoots further than the last.'],
  ['Inputs: temperature in kelvin (~300) and pressure in bar (~1). Why does plain gradient descent crawl?', 'The bowl is badly conditioned: the step must suit the steep direction, so the flat one barely moves. Standardise first.'],
], ['Loss exploding to NaN.', 'Lower the learning rate first. Exploding loss is almost always eta times the largest curvature above 2. Also check the inputs are scaled.', 'Kelvin and bar.', 'Raw kelvin values around 300 also make the bias and temperature weights nearly collinear, so kappa is enormous. Subtract the mean and divide by the spread.']);
