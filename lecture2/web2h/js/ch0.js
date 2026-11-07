/* ch0.js: the opening. Cold open, colour key, the three questions, the bridge from Lecture 1, today's route. */
'use strict';

/* a chapter title card, shared by every chapter */
function chapterCard(num, title, sub, notes, draw) {
  Deck.add({
    id: 'ch' + num, title: `Chapter ${num}: ${title}`, beats: 1, dur: 3600, min: 0.3, notes,
    html: () => `
      ${at(150, 300, `<div class="mono c-loss" style="font:600 220px/1 var(--mono);letter-spacing:-.04em">${String(num).padStart(2, '0')}</div>`, 0, 'slam')}
      ${at(150, 560, words(title, 'h1', 0, `style="--d:250ms;width:1700px;white-space:nowrap;${title.length > 18 ? 'font-size:96px' : ''}"`), 0, 'fade')}
      ${at(156, 720, `<div class="lede" style="font-style:italic">${sub}</div>`, 0, 'rise', '--d:700ms')}`,
    frame(g, S) { if (draw) draw(g, S); },
  });
}

Deck.chapter('Opening');

/* ------------------------------------------------------------ cold open: a landscape full of balls rolling downhill */
const BUMPY_GRID = (() => { const n = 46; return linspace(-1, 1, n + 1).map(u => linspace(-1, 1, n + 1).map(v => BUMPY.f(u, v))); })();
Deck.add({
  id: 'cold-open', title: 'Cold open', beats: 4, dur: [5200, 3600, 3800, 5000], chrome: false, min: 1,
  notes: [
    'Let the picture run for a moment before you speak. Every one of these balls is a model: a set of knobs. The height under it is how wrong that model is.',
    'Linear regression, logistic regression, the network in your phone and ChatGPT all learn this way.',
    'Each ball only feels the slope under its feet and takes a small step downhill, again and again. That is gradient descent, and it is today\'s topic.',
    'Lecture 2 of 5. Last time we built models out of decision trees. Today we build them out of weighted sums, and we train them by following the gradient.',
  ],
  html: () => `
    <div class="abs" style="left:0;top:110px;width:1920px;text-align:center">
      ${words('Every model you have heard of,', 'h2', 0, 'data-x="3"')}
      ${words('from a fitted line to ChatGPT,', 'h2 c-dim', 1, 'data-x="3" style="margin-top:12px"')}
      ${words('learns by rolling downhill.', 'h2 c-data', 2, 'data-x="3" style="margin-top:12px"')}
    </div>
    <div class="abs" style="left:0;top:330px;width:1920px;text-align:center">
      <div class="h1 slam" data-b="3" style="font-size:150px;letter-spacing:-.03em">LEARNING BY<br>GRADIENTS</div>
      <div class="eyebrow rise" data-b="3" style="--d:500ms;margin-top:46px;color:var(--fg)">Lecture 2 of 5 &nbsp;·&nbsp; From decision trees to LLMs</div>
      <div class="lede rise" data-b="3" style="--d:800ms;margin:22px auto 0;font-size:30px">Paras Koundal</div>
    </div>`,
  enter(st) {
    const r = RNG(4);
    st.balls = Array.from({ length: 40 }, (_, i) => ({ u: r.u(-.92, .92), v: r.u(-.92, .92), delay: i * 45, trail: [] }));
    st.cam = new Cam(960, 690, 500, -0.6, 0.62, 0.12); st.r = r;
  },
  frame(g, S) {
    const st = S.st, cam = st.cam;
    cam.yaw = -0.6 + S.t * 0.00005;
    const dim = S.beat === 3 ? 1 - 0.65 * S.p(900) : 1;
    cam.cy = 690 + (S.beat === 3 ? 120 * Ease.inOut(S.p(1200)) : 0);
    drawSurface(cam, BUMPY_GRID, { zscale: .5, zmin: -.35, zmax: 1.25, a: 0.95 * dim });
    const lr = 0.02 * S.dt / 16.7;
    for (const b of st.balls) {
      const life = S.t - b.delay; if (life < 0) continue;
      if (life > 900) {
        const [gu, gv] = BUMPY.g(b.u, b.v); b.u -= lr * gu; b.v -= lr * gv;
        b.still = Math.hypot(gu, gv) < 0.02 ? (b.still || 0) + S.dt : 0;
        if (b.still > 1200) { b.u = st.r.u(-.92, .92); b.v = st.r.u(-.92, .92); b.delay = S.t; b.trail = []; b.still = 0; continue; }
      }
      b.trail.push([b.u, b.v]); if (b.trail.length > 70) b.trail.shift();
      const drop = 1 - Ease.out(clamp(life / 900));
      const pts = b.trail.map(([u, v]) => cam.p(u, v, BUMPY.f(u, v) * .5 + .01));
      if (life > 900) D.line(pts.map(p => [p[0], p[1]]), COL.model, 2.5, .45 * dim);
      const [x, y] = cam.p(b.u, b.v, BUMPY.f(b.u, b.v) * .5 + .03 + drop * 1.4);
      D.dot(x, y, 9, COL.model, clamp(life / 300) * dim, 18);
      D.dot(x - 2.5, y - 2.5, 3, '#ffffff', .8 * clamp(life / 300) * dim);
    }
    // a soft vignette keeps the type readable
    const vg = g.createLinearGradient(0, 0, 0, 520); vg.addColorStop(0, 'rgba(7,9,13,.92)'); vg.addColorStop(1, 'rgba(7,9,13,0)');
    g.fillStyle = vg; g.fillRect(0, 0, W, 520);
  },
});

/* ------------------------------------------------------------ the colour key */
Deck.add({
  id: 'colour-key', title: 'How to read the pictures', beats: 1, dur: 5200, min: 0.5,
  notes: 'One minute on how to read today\'s pictures, because colour carries meaning everywhere. Amber is always measured data. Cyan is always the model, its prediction or its knobs. Pink is always error. Violet is always a gradient, the direction of steepest increase. Green means it worked.',
  html: () => `${hdr('Before we start', 'Every colour means one thing')}
    ${[['Measured data', 'a batch you actually ran', 'c-data'], ['The model', 'its prediction, its knobs', 'c-model'], ['Error', 'how far off the model is', 'c-loss'], ['Gradient', 'which way is uphill', 'c-grad'], ['Success', 'it learned', 'c-good']]
      .map(([a, b, c], i) => at(560, 330 + i * 116, `<div class="h3 ${c}" style="font-size:42px">${a}</div><div class="lede" style="font-size:28px;margin-top:4px">${b}</div>`, 0, 'rise', `--d:${200 + i * 380}ms`)).join('')}`,
  frame(g, S) {
    const y0 = 372, dy = 116, x = 330;
    const p = i => Ease.out(clamp((S.bt - 200 - i * 380) / 700)) * (S.beat > 0 ? 1 : 1);
    // data: three dots
    [0, 1, 2].forEach(k => D.dot(x - 60 + k * 60, y0 + (k === 1 ? -14 : 8), 13 * p(0), COL.data, 1, 14));
    // model: a line drawing itself
    D.line([[x - 100, y0 + dy + 22], [lerp(x - 100, x + 100, p(1)), lerp(y0 + dy + 22, y0 + dy - 22, p(1))]], COL.model, 6, 1, null, 12);
    // error: a spring
    D.spring(x, y0 + 2 * dy - 34, x, y0 + 2 * dy - 34 + 70 * p(2), COL.loss, 6, 12, 3);
    D.dot(x, y0 + 2 * dy - 34, 8 * p(2), COL.loss);
    // gradient: an arrow
    D.arrow(x - 90, y0 + 3 * dy + 10, lerp(x - 90, x + 90, p(3)), y0 + 3 * dy + 10 - 30 * p(3), COL.grad, 6, 26, p(3) > 0 ? 1 : 0, 10);
    // success: a tick
    const q = p(4); D.line([[x - 50, y0 + 4 * dy], [x - 15, y0 + 4 * dy + 32 * clamp(q * 2)], [x - 15 + 80 * clamp(q * 2 - 1), y0 + 4 * dy + 32 - 70 * clamp(q * 2 - 1)]].slice(0, q > .5 ? 3 : 2), COL.good, 8, q > 0 ? 1 : 0, null, 14);
  },
});

/* ------------------------------------------------------------ the three questions */
Deck.add({
  id: 'three-questions', title: 'Model, loss, learning', beats: 3, dur: [4200, 4200, 4500], min: 1.5,
  notes: [
    'Every lecture in this series answers the same three questions. Which family of functions are we choosing from? How do we score a choice? How do we move to a better one?',
    'Lecture 1 answered them with trees: boxes in feature space, impurity or log-loss, and greedy splitting plus averaging and boosting.',
    'Today every answer changes. The model becomes a weighted sum with a bend. The loss comes from probability: squared error and cross-entropy. And the learning rule is the gradient, which is the engine of every lecture from here to LLMs.',
  ],
  html: () => {
    const cols = [['Model', 'Which family of functions?', 'boxes from trees', 'weighted sums, then a bend', 'c-model'],
      ['Loss', 'How wrong is a choice?', 'impurity, log-loss', 'squared error, cross-entropy', 'c-loss'],
      ['Learning', 'How do we get better?', 'greedy splits, boosting', 'gradient descent + backprop', 'c-grad']];
    return `${hdr('The three questions', 'Same questions, new answers')}` + cols.map(([a, q, old, nw, c], i) => `
      ${at(120 + i * 580, 340, `<div class="mono c-loss" style="font:600 30px/1 var(--mono)">${i + 1}</div><div class="h2" style="margin-top:14px">${a}</div><div class="lede" style="font-size:30px;margin-top:10px">${q}</div>`, 0, 'rise', `--d:${i * 200}ms`)}
      ${at(120 + i * 580, 610, `<div class="lbl">Lecture 1</div><div class="big c-dim" style="margin-top:10px">${old}</div>`, 1, 'rise', `--d:${i * 200}ms`, 2)}
      ${at(120 + i * 580, 610, `<div class="lbl" style="color:var(--fg)">Today</div><div class="big ${c}" style="margin-top:10px;font-weight:500">${nw}</div>`, 2, 'rise', `--d:${300 + i * 250}ms`)}`).join('');
  },
});

/* ------------------------------------------------------------ the bridge from boosting */
const BOOST = (() => {
  const { x, y } = FEED, n = x.length, M = 40, nu = 0.3, F = Array(n).fill(mean(y)), stages = [], grid = linspace(0, 4, 161);
  let Fg = grid.map(() => mean(y)); stages.push(Fg.slice());
  for (let m = 0; m < M; m++) {
    const r = y.map((v, i) => v - F[i]);
    let best = null;
    for (const t of x) {
      const L = x.map((v, i) => [v, r[i]]).filter(([v]) => v <= t), R = x.map((v, i) => [v, r[i]]).filter(([v]) => v > t);
      if (!L.length || !R.length) continue;
      const ml = mean(L.map(a => a[1])), mr = mean(R.map(a => a[1]));
      const sse = sum(L.map(a => (a[1] - ml) ** 2)) + sum(R.map(a => (a[1] - mr) ** 2));
      if (!best || sse < best.sse) best = { t, ml, mr, sse };
    }
    x.forEach((v, i) => { F[i] += nu * (v <= best.t ? best.ml : best.mr); });
    Fg = Fg.map((v, k) => v + nu * (grid[k] <= best.t ? best.ml : best.mr)); stages.push(Fg.slice());
  }
  return { grid, stages };
})();
const GD_FEED = gdPath((w, b) => FEED.grad(w, b), FEED.start, 0.12, 400);
Deck.add({
  id: 'bridge', title: 'From boosting to gradients', beats: 2, dur: [6000, 3500], min: 2,
  notes: [
    'Remember the end of Lecture 1. Gradient boosting starts from a flat guess and adds small trees, each fitted to the residuals. We showed that fitting the residuals is a step downhill on the loss, where the thing being moved is the prediction itself. Watch the staircase creep onto the data. At the same time, on the right, a straight line does the same job by moving just two numbers, its slope and intercept.',
    'Same loss, same downhill step. What changes is what moves. Boosting moves the predictions F(x), one tree at a time. Today the model has knobs, the weights w, and we move those. That one change is what makes neural networks, and eventually LLMs, trainable.',
  ],
  html: () => `${hdr('From Lecture 1', 'Boosting moved predictions. Today we move weights.')}
    ${at(150, 860, M(T`F \leftarrow F - \eta\,\frac{\partial L}{\partial F}`), 1, 'rise', 'font-size:50px')}
    ${at(1110, 860, M(T`\color{#3fd0ff}{w} \leftarrow \color{#3fd0ff}{w} - \eta\,\frac{\partial L}{\partial \color{#3fd0ff}{w}}`), 1, 'rise', 'font-size:50px;--d:250ms')}
    ${at(150, 262, '<div class="lbl">Lecture 1 · boosting</div>', 0)}
    ${at(1110, 262, '<div class="lbl">Today · gradient descent on w, b</div>', 0, 'rise', '--d:200ms')}`,
  frame(g, S) {
    const P1 = new Plot(150, 380, 680, 400, [0, 4], [0, 8]), P2 = new Plot(1110, 380, 680, 400, [0, 4], [0, 8]);
    const p = S.at(0, 5200, 300);
    for (const P of [P1, P2]) {
      P.axes({ xt: [0, 1, 2, 3, 4], yt: [0, 4, 8], xl: 'feed rate', yl: 'titre' });
      FEED.x.forEach((x, i) => D.dot(P.X(x), P.Y(FEED.y[i]), 8, COL.data, 1, 8));
    }
    const m = Math.min(BOOST.stages.length - 1, Math.floor(p * (BOOST.stages.length - 1)));
    const st = BOOST.stages[m];
    D.line(BOOST.grid.map((x, k) => [P1.X(x), P1.Y(st[k])]), COL.model, 4, 1, null, 10);
    D.text(`tree ${m}`, P1.x + P1.w, P1.y - 20, { size: 22, align: 'right', col: COL.model });
    const k = Math.min(GD_FEED.length - 1, Math.floor(Math.pow(p, 2) * 120));
    const [w, b] = GD_FEED[k];
    D.line([[P2.X(0), P2.Y(b)], [P2.X(4), P2.Y(w * 4 + b)]], COL.model, 4, 1, null, 10);
    D.text(`step ${k}   w = ${fmt(w)}   b = ${fmt(b)}`, P2.x + P2.w, P2.y - 20, { size: 22, align: 'right', col: COL.model });
  },
});

/* ------------------------------------------------------------ today's route */
Deck.add({
  id: 'route', title: 'Today\'s route', beats: 1, dur: 6500, min: 1,
  notes: 'Here is the route for today. We start with the simplest model, a straight line, and use it to meet the loss. Then the learning rule, gradient descent, and what goes wrong with it. Then yes/no answers and where the loss functions come from. Then we stack neurons into a network, learn how backpropagation computes every gradient at once, watch a network learn an operating window live, and finish with what gradients can do for chemical engineers beyond neural networks: fitting kinetic models, optimising processes through surrogate models, and building physics into the loss. Two hours, one idea: feel the slope, step downhill, repeat.',
  html: () => hdr('Today', 'Seven stops, one idea'),
  frame(g, S) {
    const stops = [['01', 'A line that learns', 'regression · springs · the loss surface'], ['02', 'Rolling downhill', 'gradient descent · step size · conditioning · noise'], ['03', 'Yes or no?', 'logistic regression · likelihood · cross-entropy'], ['04', 'Stacking neurons', 'networks · approximation · bending space'], ['05', 'Blame flows backwards', 'backpropagation, by hand'], ['06', 'Watch it learn', 'training live · overfitting · uncertainty'], ['07', 'Gradients for engineers', 'kinetics · surrogates · physics']];
    const hill = x => 420 + 300 * (x / 1640) + 26 * Math.sin(x / 140);
    const X0 = 140, X1 = 1780, p = Ease.inOut(S.p(5600, 400));
    void X1;
    const pts = linspace(X0, X1, 200).map(x => [x, hill(x - X0)]);
    D.line(pts, '#2c3646', 3, clamp(S.bt / 400));
    stops.forEach(([n, t, s], i) => {
      const x = X0 + 40 + i * 255, y = hill(x - X0), reached = p * 6 >= i - 0.01;
      const a = clamp((S.bt - 300 - i * 280) / 500);
      D.dot(x, y, reached ? 11 : 8, reached ? COL.loss : '#3a4352', a);
      const up = i % 2 === 0, words_ = s.split(' · '), y0 = up ? y - 70 - 32 * words_.length - 80 : y + 60;
      D.text(n, x - 10, y0, { size: 28, weight: 600, col: COL.loss, a });
      D.text(t, x - 10, y0 + 38, { size: 30, weight: 600, font: '"IBM Plex Sans", sans-serif', col: COL.fg, a });
      words_.forEach((w, k) => D.text(w, x - 10, y0 + 80 + k * 32, { size: 22, font: '"IBM Plex Sans", sans-serif', col: COL.dim, a }));
      D.seg(x, y + (up ? -14 : 14), x, up ? y0 + 80 + 32 * words_.length - 10 : y0 - 22, '#2c3646', 1.5, a);
    });
    const bx = X0 + 40 + p * 1530, by = hill(bx - X0);
    D.dot(bx, by - 14, 14, COL.model, 1, 22); D.dot(bx - 4, by - 18, 4, '#fff', .9);
  },
});
