/* ch1.js: A line that learns. Linear regression seen as springs, squares, a surface and a projection;
 * then Arrhenius kinetics, where the loss comes from, and robustness to bad data. */
'use strict';

Deck.chapter('A line that learns', 'linear regression, springs and the loss surface');

/* reusable: a plot of the 14 fed-batch runs */
const PF = new Plot(150, 270, 1000, 580, [0, 4], [0, 8]);
function feedAxes(a = 1) { PF.axes({ xt: [0, 1, 2, 3, 4], yt: [0, 2, 4, 6, 8], xl: 'glucose feed rate  (g L⁻¹ h⁻¹)', yl: 'titre  (g/L)', a, grid: true }); }
function feedPoints(a = 1, drop = 1) {
  FEED.x.forEach((x, i) => {
    const q = clamp(drop * 1.6 - i * 0.045), e = Ease.back(q);
    D.dot(PF.X(x), lerp(PF.y - 60, PF.Y(FEED.y[i]), e), 10, COL.data, a * clamp(q * 3), 12);
  });
}
function feedLine(w, b, a = 1, col = COL.model) { D.line([[PF.X(0), PF.Y(b)], [PF.X(4), PF.Y(w * 4 + b)]], col, 5, a, null, 14); }
const FEED_GD = gdPath((w, b) => FEED.grad(w, b), FEED.start, 0.12, 300);

chapterCard(1, 'A line that learns', 'linear regression, springs and the loss surface',
  'Chapter one. The simplest model there is, and the place where the idea of a loss becomes something you can see and touch.',
  (g, S) => { const P = new Plot(1180, 300, 560, 380, [0, 4], [0, 8]); const p = S.p(2500, 300); FEED.x.forEach((x, i) => D.dot(P.X(x), P.Y(FEED.y[i]), 7, COL.data, .7 * clamp(p * 3 - i * .1))); const k = Math.floor(Ease.inOut(p) * 200); const [w, b] = FEED_GD[k]; D.line([[P.X(0), P.Y(b)], [P.X(4), P.Y(4 * w + b)]], COL.model, 4, .8 * clamp(p * 2), null, 10); });

/* ------------------------------------------------------------ 1. the data and the model family */
Deck.add({
  id: 'data', title: 'Fourteen batches and a line', beats: 3, dur: [4200, 3600, 5600], min: 1.5,
  notes: [
    'Fourteen fed-batch runs, simulated so the numbers are clean. On the x-axis is how fast we fed glucose; on the y-axis is the titre we got. More feed gives more product, roughly, with scatter. We want a rule that predicts titre from feed rate.',
    'Our model family is every straight line: y-hat equals w times x plus b. Two knobs: w is the slope, b is the intercept. This is a bad first guess.',
    'Training means nothing more than choosing those two numbers. Every setting of the knobs is a different line, so we need a way to score them.',
  ],
  html: () => `${hdr('Chapter 1', 'Fourteen batches, two knobs')}
    ${at(1270, 300, `<div class="eq">${M(T`\hat{y} = \color{#3fd0ff}{w}\,x + \color{#3fd0ff}{b}`)}</div>`, 1, 'pop')}
    ${at(1275, 400, '<div class="lede" style="font-size:30px;width:520px"><span class="c-model">w</span> is the slope, <span class="c-model">b</span> the intercept. Every pair is one line.</div>', 1, 'rise', '--d:300ms')}
    ${at(1275, 560, '<div class="lbl">knobs</div><div id="wb" class="mono c-model" style="font-size:44px;margin-top:10px"></div>', 1, 'rise', '--d:500ms')}
    ${sub('14 runs (simulated). More feed, more product, roughly.', 0, 1)}
    ${sub('Training = choosing <b>w</b> and <b>b</b>. Nothing else.', 2)}`,
  frame(g, S) {
    feedAxes(); feedPoints(1, S.at(0, 2200, 200));
    if (S.beat >= 1) {
      let [w, b] = FEED.start;
      if (S.beat === 2) {
        const tour = [[-0.2, 5.2], [0.4, 1.0], [2.2, -0.5], [0.9, 3.6], [1.2, 1.6]], p = S.p(5000) * (tour.length - 1), k = Math.min(tour.length - 2, Math.floor(p)), f = Ease.inOut(p - k);
        w = lerp(tour[k][0], tour[k + 1][0], f); b = lerp(tour[k][1], tour[k + 1][1], f);
      }
      feedLine(w, b, S.at(1, 600));
      const el = document.getElementById('wb'); if (el) el.textContent = `w = ${fmt(w)}   b = ${fmt(b)}`;
    }
  },
});

/* ------------------------------------------------------------ 2. residuals as springs */
Deck.add({
  id: 'springs', title: 'Residuals as springs', beats: 3, dur: [3800, 7000, 5000], min: 2.5,
  notes: [
    'Here is a physical way to think about it. Hook every batch to the line with a vertical spring. A spring stretched by r stores energy one half k r squared. A bad line stretches the springs a lot.',
    'Now let go. The springs pull the line until it comes to rest. Watch the energy readout fall. Where it settles is exactly the least-squares line: the line with the least total squared stretch.',
    'At rest, two things must hold. The vertical forces cancel: the residuals sum to zero. And the torques cancel: the residuals weighted by x sum to zero. Those two balance equations are the normal equations of least squares. Engineers have been solving them since Gauss.',
  ],
  html: () => `${hdr('Chapter 1', 'Hang every batch on a spring')}
    ${at(1290, 300, '<div class="lbl">energy in the springs</div><div id="en" class="num c-loss" style="margin-top:14px"></div><div class="lede" style="font-size:26px;margin-top:10px">mean of r², the squared stretches</div>', 0, 'rise')}
    ${at(1290, 640, `<div class="eq-s">${M(T`\textstyle\sum_i \color{#ff4d7d}{r_i} = 0`)}</div><div class="lbl" style="margin:6px 0 22px">forces balance</div><div class="eq-s">${M(T`\textstyle\sum_i x_i\,\color{#ff4d7d}{r_i} = 0`)}</div><div class="lbl" style="margin-top:6px">torques balance</div>`, 2, 'rise')}
    ${sub('A bad line stretches the springs.', 0, 1)}
    ${sub('Let go: the line settles where the stored energy is lowest.', 1, 2)}
    ${sub('At rest, forces and torques balance. <b>That is least squares.</b>', 2)}`,
  enter(st) { st.w = FEED.start[0]; st.b = FEED.start[1]; st.vw = 0; st.vb = 0; },
  frame(g, S) {
    const st = S.st;
    if (S.beat === 1) {
      for (let k = 0; k < 6; k++) {
        const h = S.dt / 1000 / 6, [gw, gb] = FEED.grad(st.w, st.b);
        st.vw += h * (-7.5 * gw - 1.9 * st.vw); st.vb += h * (-7.5 * gb - 1.9 * st.vb);
        st.w += h * st.vw; st.b += h * st.vb;
      }
    } else if (S.beat === 2) { st.w = lerp(st.w, FEED.w, .08); st.b = lerp(st.b, FEED.b, .08); }
    else { st.w = FEED.start[0]; st.b = FEED.start[1]; st.vw = st.vb = 0; }
    feedAxes(); feedPoints();
    const pa = S.at(0, 1200, 300);
    FEED.x.forEach((x, i) => {
      const yl = st.w * x + st.b, y = FEED.y[i];
      const len = Math.abs(PF.Y(y) - PF.Y(yl)), coils = clamp(Math.round(len / 14), 3, 14);
      D.spring(PF.X(x), PF.Y(y), PF.X(x), lerp(PF.Y(y), PF.Y(yl), Ease.out(clamp(pa * 1.5 - i * .03))), COL.loss, coils, 8, 2.5, .95);
    });
    feedLine(st.w, st.b);
    const el = document.getElementById('en'); if (el) el.textContent = fmt(FEED.L(st.w, st.b), 2);
  },
});

/* ------------------------------------------------------------ 3. squares: the loss */
const SQ_PATH = gdPath((a, c) => FEED.grad(a, c), [0.55, 2.7], 0.12, 250);
Deck.add({
  id: 'squares', title: 'Squares: the loss', beats: 3, dur: [3600, 5200, 4800], min: 2,
  notes: [
    'Same idea, drawn as area. Square each miss. A miss of 1 gives area 1; a miss of 2 gives area 4. The average area is the loss, the mean squared error.',
    'Move the line toward the data and watch the squares shrink. The number on the right is the only thing training looks at.',
    'Here it is written out, symbol by symbol. Amber is the data, fixed. Cyan are the knobs. The pink bracket is the residual, the miss. The data never change during training, so the loss is a function of the knobs only: L of w and b. Notice what squaring does: one batch that misses by twice as much counts four times as much. Squared error is very sensitive to outliers; we come back to that in a few minutes.',
  ],
  html: () => `${hdr('Chapter 1', 'Square the misses, average them')}
    ${at(1290, 300, '<div class="lbl">loss · mean squared error</div><div id="ls" class="num c-loss" style="margin-top:14px"></div>', 0, 'rise')}
    ${at(1250, 560, `<div class="eq-s">${M(T`L(\color{#3fd0ff}{w},\color{#3fd0ff}{b}) = \frac{1}{n}\sum_{i=1}^{n}\color{#ff4d7d}{\Big(}\color{#ffb547}{y_i} \color{#ff4d7d}{-} (\color{#3fd0ff}{w}\,x_i + \color{#3fd0ff}{b})\color{#ff4d7d}{\Big)^2}`)}</div>`, 2, 'pop')}
    ${at(1255, 700, '<div class="lede" style="font-size:27px;width:600px">Data fixed. Knobs free. <span class="c-loss">Twice the miss, four times the cost.</span></div>', 2, 'rise', '--d:400ms')}
    ${sub('Square each miss. The average area is the loss.', 0, 1)}
    ${sub('Training only ever looks at this one number.', 1)}`,
  frame(g, S) {
    let w = 0.55, b = 2.7;
    if (S.beat >= 1) { const k = Math.floor(Ease.inOut(S.at(1, 4600, 300)) * 250); const q = SQ_PATH[k]; w = q[0]; b = q[1]; }
    feedAxes(); feedPoints();
    const pa = S.at(0, 1200, 300);
    FEED.x.forEach((x, i) => {
      const yl = w * x + b, y = FEED.y[i], s = Math.abs(PF.Y(y) - PF.Y(yl)) * Ease.out(clamp(pa * 1.4 - i * .03));
      const top = Math.min(PF.Y(y), PF.Y(yl)), left = PF.X(x) + s > PF.x + PF.w ? PF.X(x) - s : PF.X(x);
      D.rect(left, top, s, s, hexA(COL.loss, .16), 1, hexA(COL.loss, .85), 2);
    });
    feedLine(w, b);
    const el = document.getElementById('ls'); if (el) el.textContent = fmt(FEED.L(w, b), 2);
  },
});

/* ------------------------------------------------------------ 4. the loss surface */
const SURF = (() => {
  const n = 40, wOf = u => 1.5 + 2 * u, bOf = v => 1.5 + 3.5 * v, zOf = L => Math.min(1.15, Math.sqrt(L) / 6);
  const grid = linspace(-1, 1, n + 1).map(u => linspace(-1, 1, n + 1).map(v => zOf(FEED.L(wOf(u), bOf(v)))));
  return { grid, uOf: w => (w - 1.5) / 2, vOf: b => (b - 1.5) / 3.5, zOf, wOf, bOf };
})();
Deck.add({
  id: 'surface', title: 'Every line is a point on a surface', beats: 3, dur: [6400, 5200, 5000], min: 2.5,
  notes: [
    'Now the key picture of the whole lecture. On the left, a line. On the right, the loss for every possible pair of knobs, drawn as a surface: w along one axis, b along the other, and height is the loss. As I move the line, its point moves on the surface.',
    'The best line is the lowest point. Finding it is the whole job of training.',
    'For a straight line with squared error the surface is a quadratic bowl. Its curvature is the Hessian, which here is two over n times X-transpose X. Its eigenvalues are about 11.6 and 0.31: the bowl is 38 times steeper one way than the other. That ratio, the condition number, will decide how hard gradient descent has to work. Hold on to it for chapter 2.',
  ],
  html: () => `${hdr('Chapter 1', 'Every line is one point on a surface')}
    ${at(1540, 830, '<div class="lbl">height = loss</div>', 0, 'fade', '--d:600ms')}
    ${at(130, 830, `<div class="eq-xs">${M(T`H = \tfrac{2}{n}X^{\top}X,\quad \lambda = 11.6,\ 0.31`)}</div><div class="lede" style="font-size:26px;margin-top:6px">a long tilted valley · condition number <span class="c-data">κ ≈ 38</span></div>`, 2, 'rise')}
    ${sub('Move the line, and its point moves on the loss surface.', 0, 1)}
    ${sub('The best line is the lowest point. Training = finding it.', 1, 2)}
    ${sub('A quadratic bowl, but a stretched one. Remember κ.', 2)}`,
  enter(st) { st.cam = new Cam(1380, 690, 290, -0.75, 0.6, 0.14); st.path = gdPath((a, c) => FEED.grad(a, c), [3.1, -1.2], 0.085, 160); },
  frame(g, S) {
    const st = S.st, cam = st.cam;
    cam.yaw = -0.75 + 0.18 * Math.sin(S.t / 5200);
    const P = new Plot(130, 300, 620, 420, [0, 4], [0, 8]);
    P.axes({ xt: [0, 2, 4], yt: [0, 4, 8], xl: 'feed rate', yl: 'titre' });
    FEED.x.forEach((x, i) => D.dot(P.X(x), P.Y(FEED.y[i]), 7, COL.data));
    let w, b, trail = [];
    if (S.beat === 0) { const th = S.p(6000) * TAU * .95; w = FEED.w + 1.25 * Math.cos(th); b = FEED.b + 2.4 * Math.sin(th); }
    else { const k = S.beat === 1 ? Math.floor(Ease.out(S.p(4600)) * (st.path.length - 1)) : st.path.length - 1; [w, b] = st.path[k]; trail = st.path.slice(0, k + 1); }
    D.line([[P.X(0), P.Y(b)], [P.X(4), P.Y(4 * w + b)]], COL.model, 4, 1, null, 10);
    drawSurface(cam, SURF.grid, { zscale: 1, zmin: 0, zmax: 1.1, a: S.at(0, 900) });
    // axis labels on the floor
    const lab = (u, v, s, c) => { const [x, y] = cam.p(u, v, 0); D.text(s, x, y, { size: 22, col: c, align: 'center' }); };
    lab(0, -1.25, 'slope  w', COL.model); lab(1.25, 0, 'intercept  b', COL.model);
    if (trail.length > 1) D.line(trail.map(([a, c]) => { const q = cam.p(SURF.uOf(a), SURF.vOf(c), SURF.zOf(FEED.L(a, c)) + .02); return [q[0], q[1]]; }), COL.model, 3, .9);
    const [x, y] = cam.p(SURF.uOf(w), SURF.vOf(b), SURF.zOf(FEED.L(w, b)) + .03);
    D.dot(x, y, 12, COL.model, 1, 22); D.dot(x - 3, y - 3, 4, '#fff', .9);
    if (S.beat >= 1) { const [mx, my] = cam.p(SURF.uOf(FEED.w), SURF.vOf(FEED.b), SURF.zOf(FEED.Lmin)); D.ring(mx, my, 18 + 4 * Math.sin(S.t / 200), COL.good, 3, S.at(1, 600, 3500)); }
  },
});

/* ------------------------------------------------------------ 5. least squares as a projection */
const PROJ = (() => {
  const x = [1, 2, 3], one = [1, 1, 1], y = [2.6, 2.2, 5.6];
  const A = [[3, 6], [6, 14]], c = [sum(y), sum(x.map((v, i) => v * y[i]))];
  const [b, w] = solve(A, c); const yh = x.map(v => w * v + b);
  return { x, one, y, w, b, yh };
})();
Deck.add({
  id: 'projection', title: 'Least squares is a projection', beats: 3, dur: [4800, 5200, 4800], min: 2.5,
  notes: [
    'One more view, for the linear algebra fans. Stack the measured titres into one vector y. With n batches it lives in n-dimensional space; I have drawn three batches so we can see it. The model can only produce vectors of the form w times x plus b times the ones vector. Those fill a plane, the column space of X. Our data vector sticks out of that plane because of noise.',
    'Least squares asks: which point on the plane is closest to y? Watch the candidate slide across the plane while the pink distance shrinks.',
    'The closest point is the foot of the perpendicular. So the residual is perpendicular to every column of X: X-transpose times the residual is zero. Those are the same balance equations as the springs, now as geometry, and solving them gives w equals X-transpose X inverse X-transpose y. If two columns of X point the same way, for example feed rate in grams and in moles, the plane collapses and X-transpose X cannot be inverted.',
  ],
  html: () => `${hdr('Chapter 1', 'Least squares picks the nearest point')}
    ${at(1260, 640, `<div class="eq-s">${M(T`X^{\top}(\color{#ffb547}{y} - X\color{#3fd0ff}{w}) = 0`)}</div><div class="eq-s" style="margin-top:22px">${M(T`\color{#3fd0ff}{w} = (X^{\top}X)^{-1}X^{\top}\color{#ffb547}{y}`)}</div>`, 2, 'rise')}
    ${at(1260, 300, '<div class="lbl c-data">y · your measurements</div><div class="lbl c-model" style="margin-top:14px">plane · everything a line can predict</div><div class="lbl c-loss" style="margin-top:14px">residual · what is left over</div>', 0, 'rise', '--d:400ms')}
    ${sub('Drawn for 3 batches. Your data are one point; a line can only reach the plane.', 0, 1)}
    ${sub('Least squares slides to the nearest point on the plane.', 1, 2)}
    ${sub('The residual is perpendicular to every feature: the normal equations.', 2)}`,
  enter(st) { st.cam = new Cam(700, 600, 96, -0.9, 0.42, 0.04); },
  frame(g, S) {
    const st = S.st, cam = st.cam; cam.yaw = -0.9 + 0.25 * Math.sin(S.t / 4000);
    const { x, one, y, w, b, yh } = PROJ;
    const pt = (v, z = 0) => { const q = cam.p(v[0] - 2.5, v[1] - 2.5, v[2] - 1.6 + z); return [q[0], q[1]]; };
    // coordinate axes
    [[6, 0, 0], [0, 6, 0], [0, 0, 6]].forEach((e, k) => { D.line([pt([0, 0, 0]), pt(e)], '#2c3646', 2); D.text(`batch ${k + 1}`, ...pt(e.map(v => v * 1.08)), { size: 20 }); });
    // the plane spanned by 1 and x
    const corner = (s, t) => pt(one.map((o, i) => s * o + t * x[i]));
    const quad = [corner(-0.5, 0), corner(-1.4, 2.4), corner(1.6, 2.4), corner(3.2, 0)];
    D.poly(quad, hexA(COL.model, .14), S.at(0, 800), hexA(COL.model, .5), 1.5);
    for (let s = -1; s <= 3; s += .5) D.line([corner(s, 0), corner(s - .9 * 0, 2.4)].map((p, k) => (k ? corner(s - 0.9 + .9, 2.4) : p)), hexA(COL.model, .12), 1);
    // the data vector
    D.arrow(...pt([0, 0, 0]), ...pt(y), COL.data, 5, 22, S.at(0, 800, 600), 12);
    D.text('y', ...pt(y.map(v => v * 1.06)), { size: 30, col: COL.data, font: '"IBM Plex Sans"', weight: 600, a: S.at(0, 800, 600) });
    // a candidate sliding on the plane
    if (S.beat >= 1) {
      const p = Ease.inOut(S.at(1, 4400, 200));
      const path = [[0.4, 0.2], [2.4, 0.9], [-.2, 1.6], [b, w]];
      const k = Math.min(path.length - 2, Math.floor(p * 3)), f = Ease.inOut(p * 3 - k);
      const s = lerp(path[k][0], path[k + 1][0], f), t = lerp(path[k][1], path[k + 1][1], f);
      const cand = one.map((o, i) => s * o + t * x[i]);
      D.line([pt(cand), pt(y)], COL.loss, 3, 1, [10, 8]);
      D.dot(...pt(cand), 11, COL.model, 1, 18);
      const dist = Math.hypot(...cand.map((c, i) => c - y[i]));
      D.text(`distance ${fmt(dist)}`, ...pt(cand.map((c, i) => (c + y[i]) / 2)).map((v, i) => v + (i ? -14 : 24)), { size: 24, col: COL.loss });
    }
    if (S.beat >= 2) {
      const a = S.at(2, 700);
      D.arrow(...pt([0, 0, 0]), ...pt(yh), COL.model, 5, 20, a, 10);
      D.line([pt(yh), pt(y)], COL.loss, 5, a, null, 10);
      // right-angle mark
      const u = yh.map((v, i) => (y[i] - v)), un = Math.hypot(...u), e = yh.map(v => v), en = Math.hypot(...e), s = .35;
      const p1 = yh.map((v, i) => v + u[i] / un * s), p2 = yh.map((v, i) => v + u[i] / un * s - e[i] / en * s), p3 = yh.map((v, i) => v - e[i] / en * s);
      D.line([pt(p1), pt(p2), pt(p3)], COL.fg, 2, a);
    }
  },
});

/* ------------------------------------------------------------ 6. Arrhenius: linear in the parameters */
const PA_raw = new Plot(150, 250, 960, 600, [295, 375], [0, 0.2]), PA_lin = new Plot(150, 250, 960, 600, [2.65, 3.4], [-6.5, -1.5]);
Deck.add({
  id: 'arrhenius', title: 'Arrhenius: a curve hiding a line', beats: 4, dur: [4000, 4600, 4600, 5200], min: 3,
  notes: [
    'A chemical engineering example you have all done by hand. Eight rate constants measured between 300 and 370 kelvin, simulated with a true activation energy of 55 kilojoules per mole and 6 percent noise. Plotted raw, it curves upward. No straight line in T fits this.',
    'But take logs. ln k equals ln A minus Ea over R times one over T. Plot ln k against one over T and the same eight points fall on a line. This is exactly our model: y equals w x plus b, with y = ln k and x = 1000 over T.',
    'Fit it with least squares. The slope is minus Ea over R, which gives Ea of about 56 kilojoules per mole; the intercept gives A. That is linear regression. The word linear means linear in the knobs, not in your measurements: any model that is a weighted sum of fixed functions of the inputs is fair game.',
    'Researchers should never report one number. Resample the eight runs with replacement, refit, and repeat sixty times: the fan of lines is your uncertainty, and the spread of slopes gives a 95 percent interval for Ea of roughly 54.5 to 58.5 kilojoules per mole. That is the bootstrap, and it works for any model in this lecture.',
  ],
  html: () => `${hdr('Chapter 1 · kinetics', 'Arrhenius: a curve hiding a line')}
    ${at(1250, 290, `<div class="eq-s">${M(T`k = A\,e^{-E_a/RT}`)}</div>`, 0, 'rise', '', 1)}
    ${at(1250, 290, `<div class="eq-s">${M(T`\ln k = \ln A - \frac{E_a}{R}\cdot\frac{1}{T}`)}</div><div class="eq-s" style="margin-top:18px">${M(T`\color{#ffb547}{y} = \color{#3fd0ff}{b} + \color{#3fd0ff}{w}\,x`)}</div>`, 1, 'rise')}
    ${at(1250, 530, '<div class="lbl">fitted</div><div class="big" style="margin-top:10px"><span class="mono c-model">E<sub>a</sub> = ' + ARR.Ea_fit.toFixed(1) + ' kJ/mol</span></div><div class="big" style="margin-top:6px"><span class="mono c-model">A = ' + ARR.A_fit.toExponential(1).replace('e+', ' × 10^') + ' s⁻¹</span></div><div class="lede" style="font-size:26px;margin-top:8px">true: 55 kJ/mol, 1.0 × 10^7 s⁻¹</div>', 2, 'rise')}
    ${at(1250, 780, `<div class="lbl">bootstrap, 60 resamples</div><div class="big c-data" style="margin-top:10px">95%: ${ARR.ci[0].toFixed(1)} to ${ARR.ci[1].toFixed(1)} kJ/mol</div>`, 3, 'rise')}
    ${sub('8 rate constants (simulated). They curve: no straight line in T fits.', 0, 1)}
    ${sub('Take logs: ln k against 1/T is a straight line.', 1, 2)}
    ${sub('"Linear" means linear in the knobs, not in your measurements.', 2, 3)}
    ${sub('Resample the runs: the fan of lines is your uncertainty on E<sub>a</sub>.', 3)}`,
  frame(g, S) {
    const f = Ease.inOut(S.at(1, 2600, 300));
    const aRaw = 1 - f, aLin = f;
    if (aRaw > 0.01) PA_raw.axes({ xt: [300, 320, 340, 360], yt: [0, 0.1, 0.2], xl: 'temperature T  (K)', yl: 'rate constant k  (s⁻¹)', a: aRaw });
    if (aLin > 0.01) PA_lin.axes({ xt: [2.7, 2.9, 3.1, 3.3], yt: [-6, -4, -2], xl: '1000 / T  (K⁻¹)', yl: 'ln k', a: aLin, xf: v => v.toFixed(1) });
    ARR.T.forEach((Tk, i) => {
      const k = ARR.k[i], q = Ease.back(clamp(S.at(0, 1800, 200) * 1.6 - i * .08));
      const x0 = PA_raw.X(Tk), y0 = lerp(PA_raw.y, PA_raw.Y(k), q), x1 = PA_lin.X(1000 / Tk), y1 = PA_lin.Y(Math.log(k));
      // points travel on a slight arc so the reordering is visible
      const xx = lerp(x0, x1, f), yy = lerp(y0, y1, f) - Math.sin(Math.PI * f) * 80;
      D.dot(xx, yy, 11, COL.data, clamp(q * 3), 14);
    });
    if (S.beat === 0) D.fn(PA_raw, t => ARR.A * Math.exp(-ARR.Ea / (ARR.R * t)), 300, 372, hexA(COL.data, .5), 2, S.at(0, 800, 1800), 120, [6, 8]);
    if (S.beat >= 3) {
      const n = Math.floor(S.at(3, 3000) * ARR.boots.length);
      ARR.boots.slice(0, n).forEach(bt => D.line([[PA_lin.X(2.65), PA_lin.Y(bt.b + bt.w * 2.65)], [PA_lin.X(3.4), PA_lin.Y(bt.b + bt.w * 3.4)]], COL.model, 1.5, .18));
    }
    if (S.beat >= 2) { const p = Ease.inOut(S.at(2, 1200)); const { w, b } = ARR.fit; D.line([[PA_lin.X(2.65), PA_lin.Y(b + w * 2.65)], [PA_lin.X(lerp(2.65, 3.4, p)), PA_lin.Y(b + w * lerp(2.65, 3.4, p))]], COL.model, 5, 1, null, 14); }
  },
});

/* ------------------------------------------------------------ 7. why squares? Gaussian noise; and robustness */
const ROB = (() => {
  const x = [...FEED.x, 3.05], y = [...FEED.y, 0.9], n = x.length;
  const ls = lstsq(x.map(v => [v, 1]), y);
  let wb = ls.slice();
  for (let it = 0; it < 60; it++) {            // iteratively re-weighted least squares for the absolute loss
    const wts = x.map((v, i) => 1 / Math.max(1e-3, Math.abs(y[i] - (wb[0] * v + wb[1]))));
    const A = [[0, 0], [0, 0]], c = [0, 0];
    x.forEach((v, i) => { const r = [v, 1]; for (let j = 0; j < 2; j++) { c[j] += wts[i] * r[j] * y[i]; for (let k = 0; k < 2; k++) A[j][k] += wts[i] * r[j] * r[k]; } });
    wb = solve(A, c);
  }
  return { x, y, ls, l1: wb, sigma: Math.sqrt(FEED.Lmin) };
})();
Deck.add({
  id: 'gaussian', title: 'Why squares? Gaussian noise', beats: 3, dur: [4400, 4200, 5200], min: 3,
  notes: [
    'Why squares, and not absolute values or fourth powers? Here is the honest answer. Assume each batch misses the true line by random Gaussian noise with spread sigma. Then each point has a probability under the bell curve centred on the line. A good line puts every point near the peak of its bell.',
    'Multiply the probabilities of all batches to get the likelihood of the data. Take minus the log and the exponentials turn into a sum of squared residuals over two sigma squared, plus a constant. So minimising squared error is the same as maximum likelihood, if the noise is Gaussian. The loss is a statement about your noise.',
    'And if it is not Gaussian? Suppose one batch had a failed titre assay and reads 0.9. Squared error panics: that one point pulls the whole line down, because its miss is squared. The absolute-value loss, which corresponds to heavier-tailed Laplace noise, barely moves. Choosing a loss is a modelling decision, the same way choosing a kinetic rate law is.',
  ],
  html: () => `${hdr('Chapter 1 · where the loss comes from', 'Squared error assumes Gaussian noise')}
    ${at(1200, 300, `<div class="eq-xs">${M(T`p(\color{#ffb547}{y_i}) \propto \exp\!\Big(-\frac{\color{#ff4d7d}{r_i}^2}{2\sigma^2}\Big)`)}</div><div class="eq-xs" style="margin-top:24px">${M(T`-\ln\prod_i p(\color{#ffb547}{y_i}) = \frac{1}{2\sigma^2}\sum_i \color{#ff4d7d}{r_i}^2 + \text{const}`)}</div>`, 1, 'rise')}
    ${at(1250, 620, '<div class="lbl">one failed assay</div><div class="big" style="margin-top:10px"><span class="c-model">squared error</span> is dragged down</div><div class="big" style="margin-top:6px"><span class="c-good">absolute error</span> barely moves</div>', 2, 'rise')}
    ${sub('Assume each batch misses the line by Gaussian noise.', 0, 1)}
    ${sub('Least squares = maximum likelihood, if the noise is Gaussian.', 1, 2)}
    ${sub('The loss is a claim about your noise. Choose it like a rate law.', 2)}`,
  frame(g, S) {
    feedAxes();
    const out = S.at(2, 900, 200), drag = Ease.inOut(S.at(2, 1600, 1100));
    const w = lerp(FEED.w, ROB.ls[0], drag), b = lerp(FEED.b, ROB.ls[1], drag);
    // bells
    if (S.beat <= 1) {
      const a = S.at(0, 1000, 600) * (S.beat === 2 ? 0 : 1), sg = ROB.sigma;
      FEED.x.forEach((x, i) => {
        const yc = FEED.w * x + FEED.b, pts = [];
        for (let k = -40; k <= 40; k++) { const dy = 3 * sg * k / 40; pts.push([PF.X(x) + 55 * Math.exp(-dy * dy / (2 * sg * sg)), PF.Y(yc + dy)]); }
        D.line(pts, COL.model, 2, .55 * a);
        D.seg(PF.X(x), PF.Y(yc - 3 * sg), PF.X(x), PF.Y(yc + 3 * sg), COL.model, 1, .25 * a);
        const r = FEED.y[i] - yc; D.seg(PF.X(x), PF.Y(FEED.y[i]), PF.X(x) + 55 * Math.exp(-r * r / (2 * sg * sg)), PF.Y(FEED.y[i]), COL.data, 2, a);
      });
    }
    feedPoints();
    if (S.beat >= 2) {
      D.dot(PF.X(3.05), PF.Y(0.9), 11, COL.data, out, 12); D.ring(PF.X(3.05), PF.Y(0.9), 22, COL.loss, 3, out);
      D.text('failed assay', PF.X(3.05) + 32, PF.Y(0.9), { size: 22, col: COL.loss, a: out });
      D.line([[PF.X(0), PF.Y(ROB.l1[1])], [PF.X(4), PF.Y(4 * ROB.l1[0] + ROB.l1[1])]], COL.good, 4, Ease.out(S.at(2, 800, 1200)), [16, 10]);
    }
    feedLine(w, b);
  },
});

/* ------------------------------------------------------------ 8. more inputs: a plane */
const PLANE = (() => {
  const r = RNG(17), n = 40, X = [], y = [];
  for (let i = 0; i < n; i++) { const f = r.u(0.5, 3.5), t = r.u(30, 38); X.push([f, t]); y.push(0.9 + 1.2 * f + 0.25 * (t - 34) + 0.35 * r.n()); }
  const c = lstsq(X.map(([f, t]) => [f, t - 34, 1]), y);
  return { X, y, c };
})();
Deck.add({
  id: 'plane', title: 'More inputs: a plane', beats: 2, dur: [5000, 4200], min: 1.5,
  notes: [
    'Add a second input, temperature, and the line becomes a plane in three dimensions. Same loss, the vertical misses squared and averaged; same normal equations; now three knobs.',
    'With d inputs it is a hyperplane in d plus one dimensions, written compactly as w-transpose x plus b. We cannot draw twenty inputs, but the maths does not care: everything in this lecture works the same in any dimension.',
  ],
  html: () => `${hdr('Chapter 1', 'Two inputs: the line becomes a plane')}
    ${at(1300, 330, `<div class="eq-s">${M(T`\hat{y} = \color{#3fd0ff}{w_1}\,x_{\text{feed}} + \color{#3fd0ff}{w_2}\,x_{T} + \color{#3fd0ff}{b}`)}</div>`, 0, 'rise', '--d:600ms')}
    ${at(1300, 460, `<div class="eq-s">${M(T`\hat{y} = \color{#3fd0ff}{\mathbf{w}}^{\top}\mathbf{x} + \color{#3fd0ff}{b}`)}</div><div class="lede" style="font-size:27px;margin-top:12px">d inputs: a hyperplane. Same loss, same algebra.</div>`, 1, 'rise')}
    ${sub('Same loss, same normal equations. Now three knobs.', 0)}`,
  enter(st) { st.cam = new Cam(700, 560, 270, -0.6, 0.45, 0.1); },
  frame(g, S) {
    const cam = S.st.cam; cam.yaw = -0.6 + S.t / 9000;
    const P = ([f, t], y) => { const q = cam.p((f - 2) / 1.5, (t - 34) / 4, (y - 4) / 3); return [q[0], q[1]]; };
    const c = PLANE.c, a = S.at(0, 1000);
    // floor grid
    for (let f = 0.5; f <= 3.5; f += .5) D.line([P([f, 30], 0.5), P([f, 38], 0.5)], '#1d2430', 1, a);
    for (let t = 30; t <= 38; t += 1) D.line([P([0.5, t], 0.5), P([3.5, t], 0.5)], '#1d2430', 1, a);
    // plane mesh
    const pl = (f, t) => P([f, t], c[0] * f + c[1] * (t - 34) + c[2]);
    const pa = Ease.out(S.at(0, 1400, 900));
    D.poly([pl(0.5, 30), pl(3.5, 30), pl(3.5, 38), pl(0.5, 38)], hexA(COL.model, .16 * pa), 1, hexA(COL.model, .7 * pa), 2);
    for (let f = 0.5; f <= 3.5; f += .5) D.line([pl(f, 30), pl(f, 38)], hexA(COL.model, .25 * pa), 1);
    PLANE.X.forEach((x, i) => {
      const yhat = c[0] * x[0] + c[1] * (x[1] - 34) + c[2];
      D.line([P(x, PLANE.y[i]), P(x, yhat)], COL.loss, 2, .7 * pa);
      D.dot(...P(x, PLANE.y[i]), 6, COL.data, a, 8);
    });
    D.text('feed rate', ...P([3.9, 30], 0.5), { size: 22 }); D.text('temperature', ...P([3.6, 38.6], 0.5), { size: 22 });
  },
});

/* ------------------------------------------------------------ pause and think */
function quizScene(id, chap, qs, notes) {
  const beats = qs.length * 2;
  Deck.add({
    id, title: 'Pause and think', beats, dur: Array(beats).fill(2400), min: 2.5, notes,
    html: () => `${hdr(chap, 'Pause and think')}` + qs.map(([q, a], k) => think(110 + (k % 2) * 870, 280 + Math.floor(k / 2) * 330, q, a, 2 * k, 2 * k + 1)).join(''),
  });
}
quizScene('quiz-1', 'Chapter 1', [
  ['You fit Arrhenius by least squares on ln k. Is that the same fit as least squares on k itself?', 'No. Logs assume multiplicative noise and weight every run alike. Fitting k directly lets the hot, fast runs dominate.'],
  ['When does XᵀX fail to be invertible, and what does that mean for the bowl?', 'When features are collinear, like feed in g/h and in mol/h. The bowl gets a flat valley floor: infinitely many best fits.'],
], ['Give them a minute. The first one is the classic trap with linearised kinetics.', 'Logs weight relative errors equally; fitting k directly weights absolute errors, so the hot fast runs dominate. Neither is wrong; they assume different noise.', 'Second question: collinearity.', 'Collinear features make XᵀX singular. Geometrically the plane collapses to a line; on the surface, the bottom of the bowl becomes a flat trough. Ridge regression, adding a little of the identity, is the standard fix, and is what we just saw in the ridge scene.']);
