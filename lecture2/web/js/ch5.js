/* ch5.js: Blame flows backwards. Gains in series (the chain rule as a process engineer knows it);
 * backprop worked by hand on a two-knob network; why one backward pass is enough; gradient checking;
 * vanishing gradients; the whole thing in numpy. */
'use strict';

Deck.chapter('Blame flows backwards', 'backpropagation, by hand and in code');

chapterCard(5, 'Blame flows backwards', 'backpropagation, by hand and in code',
  'Chapter five. We have a network with 65 knobs and a loss. Gradient descent needs the slope of the loss with respect to every one of them. This chapter is how we get all of them at once.',
  (g, S) => {
    const xs = [1300, 1440, 1580, 1720], y = 400, p = S.p(1400, 200);
    xs.forEach((x, k) => { D.ring(x, y, 22, COL.model, 3, clamp(p * 4 - k)); if (k) D.seg(xs[k - 1] + 24, y, x - 24, y, COL.faint, 2, p); });
    const q = (S.t / 1800) % 1, x = lerp(xs[3], xs[0], Ease.inOut(q));
    D.dot(x, y, 9, COL.grad, p, 18);
  });

/* ------------------------------------------------------------ 1. gains in series: the chain rule */
const GAINS = [2, 0.5, 3];
Deck.add({
  id: 'gains', title: 'Gains in series multiply', beats: 3, dur: [4200, 3600, 5200], min: 2.5,
  notes: [
    'Start with something every chemical engineer already knows. Three units in series, a sensor, a controller, a valve, each with a steady-state gain. A small wiggle goes in on the left. The first unit doubles it, the second halves it, the third triples it.',
    'So the overall gain is the product: 2 times 0.5 times 3, which is 3. In calculus language, the derivative of the output with respect to the input is the product of the local derivatives. That is the chain rule, and it is the entire mathematical content of backpropagation.',
    'Now read it backwards, from the output. How much does y move per unit nudge just before the last block? 3. Just before the middle block? 0.5 times 3, so 1.5. At the input? 2 times 1.5, so 3. Each step back multiplies by one more local gain, and reuses the product it already has. That reuse is the trick: walking backwards, you get the sensitivity of the output to every point in the chain in a single pass.',
  ],
  html: () => `${hdr('Chapter 5', 'Gains in series multiply')}
    ${at(560, 720, `<div class="eq">${M(T`\frac{dy}{dx} = K_3\,K_2\,K_1 = 3 \times 0.5 \times 2 = 3`)}</div>`, 1, 'rise')}
    ${at(560, 840, '<div class="lede c-grad">walk back from the output: multiply one gain at a time</div>', 2, 'rise')}
    ${sub('A wiggle goes in. Each unit scales it by its gain.', 0, 1)}
    ${sub('Overall gain = product of local gains. <b>That is the chain rule.</b>', 1, 2)}
    ${sub('Read it backwards and every sensitivity comes out in one pass.', 2)}`,
  frame(g, S) {
    const bx = [560, 960, 1360], by = 440, bw = 200, bh = 150, amp0 = 26;
    const a0 = S.at(0, 1200, 100);
    const wave = (x0, x1, amp, a, ph) => {
      const pts = linspace(x0, x1, 120).map(x => [x, by + amp * Math.sin((x - S.t * .25) / 26 + ph)]);
      D.line(pts, COL.data, 4, a, null, 10);
    };
    const edges = [[150, bx[0] - bw / 2], [bx[0] + bw / 2, bx[1] - bw / 2], [bx[1] + bw / 2, bx[2] - bw / 2], [bx[2] + bw / 2, 1780]];
    let amp = amp0;
    edges.forEach(([x0, x1], k) => {
      const ak = clamp(a0 * 4 - k);
      wave(x0, x1, amp, ak, k);
      if (k < 3) amp *= GAINS[k];
    });
    bx.forEach((x, k) => {
      const ak = clamp(a0 * 4 - k);
      D.rect(x - bw / 2, by - bh / 2, bw, bh, '#0f141c', ak, COL.model, 3, 16);
      D.text('K' + (k + 1), x, by - 22, { size: 26, align: 'center', col: COL.dim, a: ak });
      D.text('× ' + GAINS[k], x, by + 22, { size: 44, align: 'center', col: COL.model, weight: 600, a: ak });
    });
    D.text('x', 150, by - 90, { size: 32, col: COL.data, a: a0 }); D.text('y', 1760, by - 120, { size: 32, col: COL.data, a: a0 });
    if (S.beat >= 2) {
      const sens = [3, 1.5, 3, 1], lx = [(edges[0][0] + edges[0][1]) / 2, (edges[1][0] + edges[1][1]) / 2, (edges[2][0] + edges[2][1]) / 2, (edges[3][0] + edges[3][1]) / 2];
      const q = S.at(2, 3600, 200), pos = lerp(1780, 150, q);
      D.dot(pos, by + 150, 10, COL.grad, q < 1 ? 1 : 0, 20);
      D.seg(Math.max(pos, 150), by + 150, 1780, by + 150, COL.grad, 2, .5);
      [3, 2, 1, 0].forEach(k => {
        const shown = pos <= lx[k] + 1 ? 1 : 0;
        D.text(k === 3 ? '∂y/∂y = 1' : '' + sens[k], lx[k], by + 200, { size: 40, align: 'center', col: COL.grad, weight: 600, a: shown });
      });
    }
  },
});

/* ------------------------------------------------------------ 2. backprop by hand on a two-knob network */
const BPN = {   // node layout for the computational graph
  x: [200, 360], z: [600, 360], h: [960, 360], yh: [1320, 360], L: [1680, 360],
  w1: [400, 560], w2: [1140, 560], y: [1500, 560],
};
Deck.add({
  id: 'by-hand', title: 'Backprop by hand', beats: 6, dur: [4200, 4200, 3800, 4600, 4600, 4800], min: 6,
  notes: [
    'Here is the smallest network that still has everything: one input, one ReLU neuron, one output, two knobs. A batch arrives with x equal to 2, and the true answer is y equal to 3. The knobs start at w1 equal to 0.5 and w2 equal to 1.5. Every circle is a number; every edge is one simple operation. This picture is called a computational graph.',
    'Forward pass. z is w1 times x: 1. ReLU of 1 is 1. Times w2 gives a prediction of 1.5. Squared error against 3: 2.25. Keep every one of these numbers; the backward pass needs them.',
    'Now the backward pass. Start at the loss and ask how it changes with the prediction. The derivative of (y-hat minus y) squared is 2 times (y-hat minus y): 2 times minus 1.5, minus 3. Negative: increasing the prediction lowers the loss.',
    'One step back, through the multiplication by w2. Every edge only needs its local slope. The slope of y-hat with respect to w2 is h, which is 1, so dL/dw2 is minus 3 times 1: minus 3. The slope with respect to h is w2, 1.5, so dL/dh is minus 4.5. Notice we used h, a number we stored during the forward pass.',
    'Through the ReLU. Its slope is 1 because z was positive; had z been negative, the slope would be zero and the blame would stop here. So dL/dz is minus 4.5. Then through the multiplication by w1: the local slope is x, which is 2. dL/dw1 is minus 9. Every knob now has its gradient, and the rule was the same at every node: upstream gradient times local slope.',
    'Take one gradient-descent step with learning rate 0.05. w1 goes from 0.5 up to 0.95, w2 from 1.5 to 1.65. Run forward again: the prediction is 3.14 and the loss has fallen from 2.25 to 0.018. That is one complete training step, done by hand. A real network repeats exactly this, with matrices instead of single numbers.',
  ],
  html: () => `${hdr('Chapter 5', 'Backprop by hand: two knobs, one batch')}
    ${at(150, 720, `<div class="eq-s">${M(T`\frac{\partial L}{\partial \hat y} = 2(\hat y - y) = 2(1.5 - 3) = -3`)}</div>`, 2, 'rise', '', 3)}
    ${at(150, 700, `<div class="eq-s">${M(T`\frac{\partial L}{\partial w_2} = \frac{\partial L}{\partial \hat y}\,h = -3`)}</div><div class="eq-s" style="margin-top:18px">${M(T`\frac{\partial L}{\partial h} = \frac{\partial L}{\partial \hat y}\,w_2 = -4.5`)}</div>`, 3, 'rise', '', 4)}
    ${at(150, 700, `<div class="eq-s">${M(T`\frac{\partial L}{\partial z} = \frac{\partial L}{\partial h}\cdot \mathbb{1}[z>0] = -4.5`)}</div><div class="eq-s" style="margin-top:18px">${M(T`\frac{\partial L}{\partial w_1} = \frac{\partial L}{\partial z}\,x = -9`)}</div>`, 4, 'rise', '', 5)}
    ${at(150, 700, `<div class="eq-s">${M(T`w \leftarrow w - \eta\,\nabla L,\quad \eta = 0.05`)}</div>
      <table class="table" style="margin-top:22px"><tr><th></th><th>before</th><th>after</th></tr>
      <tr><td>${M(T`w_1`)}</td><td class="mono">0.50</td><td class="mono c-model">0.95</td></tr>
      <tr><td>${M(T`w_2`)}</td><td class="mono">1.50</td><td class="mono c-model">1.65</td></tr>
      <tr><td>loss</td><td class="mono c-loss">2.25</td><td class="mono c-good">0.018</td></tr></table>`, 5, 'rise')}
    ${at(1180, 720, '<div class="card" style="width:600px"><h4 class="c-grad">the only rule</h4><p>gradient into a node × the local slope of its operation = gradient out to each input</p></div>', 3, 'rise')}
    ${sub('A computational graph: every circle is a number, every edge one operation.', 0, 1)}
    ${sub('Forward: compute and <b>store</b> every intermediate value.', 1, 2)}
    ${sub('Backward: start at the loss. ∂L/∂ŷ = −3.', 2, 3)}
    ${sub('Upstream gradient × local slope. Uses the stored h.', 3, 4)}
    ${sub('ReLU passes the blame (z > 0). Every knob has its gradient.', 4, 5)}
    ${sub('One step: loss 2.25 → 0.018.', 5)}`,
  frame(g, S) {
    const B = BP, N = BPN, a0 = S.at(0, 1400, 100);
    const after = S.beat >= 5 ? Ease.inOut(S.at(5, 1400, 300)) : 0;
    const val = { x: B.x, z: lerp(B.z, B.w1n * B.x, after), h: lerp(B.h, B.hn, after), yh: lerp(B.yh, B.yhn, after), L: lerp(B.L, B.Ln, after), w1: lerp(B.w1, B.w1n, after), w2: lerp(B.w2, B.w2n, after), y: B.y };
    const grad = { L: 1, yh: B.dL_dyh, h: B.dL_dh, z: B.dL_dz, w2: B.dL_dw2, w1: B.dL_dw1 };
    const gb = { L: 2, yh: 2, w2: 3, h: 3, z: 4, w1: 4 };   // beat each gradient appears
    const fwdOrder = ['x', 'z', 'h', 'yh', 'L'];
    // edges: [from, to, op label]
    const E_ = [['x', 'z', '× w₁'], ['z', 'h', 'ReLU'], ['h', 'yh', '× w₂'], ['yh', 'L', '(ŷ − y)²'], ['w1', 'z', ''], ['w2', 'yh', ''], ['y', 'L', '']];
    const R = 46;
    E_.forEach(([f, t, lab], k) => {
      const [x1, y1] = N[f], [x2, y2] = N[t], L = Math.hypot(x2 - x1, y2 - y1), ux = (x2 - x1) / L, uy = (y2 - y1) / L;
      const r1 = f.startsWith('w') || f === 'y' ? 34 : R;
      D.arrow(x1 + ux * r1, y1 + uy * r1, x2 - ux * (R + 4), y2 - uy * (R + 4), '#3a4658', 2.5, 14, a0);
      if (lab) D.text(lab, (x1 + x2) / 2, y1 - 34, { size: 26, align: 'center', col: COL.dim, font: '"IBM Plex Sans"', a: a0 });
    });
    // local slopes on edges, from beat 3
    const slopes = [['h', 'yh', 'w₂ = 1.5', 3], ['w2', 'yh', 'h = 1', 3], ['z', 'h', '1  (z > 0)', 4], ['w1', 'z', 'x = 2', 4]];
    slopes.forEach(([f, t, s, b]) => {
      if (S.beat < b) return;
      const [x1, y1] = N[f], [x2, y2] = N[t];
      D.text(s, (x1 + x2) / 2 + (f.startsWith('w') ? -70 : 0), (y1 + y2) / 2 + (f.startsWith('w') ? 14 : 34), { size: 22, align: 'center', col: COL.grad, a: S.at(b, 600, 600) * (S.beat >= 5 ? .35 : 1) });
    });
    // forward pulse
    if (S.beat === 1) {
      const q = S.at(1, 2600, 200) * 4, k = Math.floor(q), f = q - k;
      if (k < 4) { const [x1, y1] = N[fwdOrder[k]], [x2, y2] = N[fwdOrder[k + 1]]; D.dot(lerp(x1, x2, f), lerp(y1, y2, f), 9, COL.data, 1, 16); }
    }
    // backward pulses
    const bwd = [[2, 'L', 'yh'], [3, 'yh', 'h'], [3, 'yh', 'w2'], [4, 'h', 'z'], [4, 'z', 'w1']];
    bwd.forEach(([b, f, t]) => {
      if (S.beat !== b) return;
      const q = S.at(b, 900, 100), [x1, y1] = N[f], [x2, y2] = N[t];
      if (q < 1) D.dot(lerp(x1, x2, q), lerp(y1, y2, q), 9, COL.grad, 1, 18);
    });
    // nodes
    Object.entries(N).forEach(([k, [x, y]]) => {
      const knob = k.startsWith('w'), r = knob || k === 'y' ? 34 : R;
      const c = knob ? COL.model : k === 'L' ? COL.loss : k === 'y' ? COL.dim : COL.fg;
      D.dot(x, y, r, '#0f141c', a0); D.ring(x, y, r, c, 3, a0);
      const name = { x: 'x', z: 'z', h: 'h', yh: 'ŷ', L: 'L', w1: 'w₁', w2: 'w₂', y: 'y' }[k];
      D.text(name, x, y, { size: 30, align: 'center', col: c, font: '"IBM Plex Sans"', weight: 600, a: a0 });
      // forward value (amber, above for chain nodes, beside for knobs)
      const shownF = knob || k === 'x' || k === 'y' ? a0 : (S.beat >= 1 ? clamp((S.at(1, 2600, 200) * 4 - fwdOrder.indexOf(k) + 1)) : 0);
      const v = val[k], vs = Math.abs(v - Math.round(v)) < 1e-9 ? String(v) : fmt(v, k === 'L' && v < .1 ? 3 : 2);
      if (knob || k === 'y') D.text(vs, x - 54, y, { size: 28, align: 'right', col: knob ? COL.model : COL.dim, a: shownF });
      else D.text(vs, x, y - 82, { size: 32, align: 'center', col: k === 'L' ? COL.loss : COL.data, weight: 600, a: shownF });
      // gradient (violet, below)
      if (grad[k] !== undefined && S.beat >= gb[k]) {
        const ga = S.at(gb[k], 500, 500) * (S.beat >= 5 && !knob ? .35 : 1);
        const gy = knob ? y + 64 : y + 84;
        D.text((grad[k] < 0 ? '−' : '') + Math.abs(grad[k]), x, gy, { size: 32, align: 'center', col: COL.grad, weight: 600, a: ga });
      }
    });
    if (S.beat >= 2) D.text('∂L/∂(·)', 120, 444, { size: 22, col: COL.grad, a: S.at(2, 600) });
  },
});

/* ------------------------------------------------------------ 3. why not just nudge each knob? */
Deck.add({
  id: 'cost', title: 'One backward pass buys every gradient', beats: 3, dur: [5200, 4200, 4600], min: 2.5,
  notes: [
    'Why go to this trouble? The obvious alternative is to nudge each knob a little and re-run the model to see how the loss changes. That is finite differences. For our 65-knob network you need two runs per knob: 130 forward passes for one gradient. Watch the counter.',
    'Backpropagation does one forward pass, storing the intermediate values, and one backward pass that hands every knob its gradient at once. The cost is roughly two to three forward passes, whatever the number of knobs. This is reverse-mode automatic differentiation; it was known to control engineers and numerical analysts long before Rumelhart, Hinton and Williams made it famous for neural networks in 1986.',
    'At scale the difference is everything. GPT-3 has 175 billion knobs. Finite differences would need 350 billion forward passes for every single step. Backprop needs about three. Without reverse mode there is no deep learning. The price is memory: every stored activation has to be kept until the backward pass uses it.',
  ],
  html: () => `${hdr('Chapter 5', 'One backward pass buys every gradient')}
    ${at(150, 250, '<div class="lbl c-loss">nudge one knob at a time</div>', 0, 'rise')}
    ${at(1010, 250, '<div class="lbl c-grad">backpropagation</div>', 1, 'rise')}
    ${at(150, 640, '<div class="lbl">forward passes per gradient</div><div id="fdc" class="num c-loss" style="margin-top:8px"></div>', 0, 'rise')}
    ${at(1010, 640, '<div class="lbl">forward passes per gradient</div><div id="bpc" class="num c-grad" style="margin-top:8px"></div>', 1, 'rise')}
    ${at(150, 830, '<div class="big">GPT-3, 175 billion knobs: <span class="mono c-loss">350,000,000,000</span> passes &nbsp;vs&nbsp; <span class="mono c-grad">≈ 3</span></div>', 2, 'pop')}
    ${sub('65 knobs, two runs each: 130 forward passes per gradient.', 0, 1)}
    ${sub('One forward, one backward: every gradient, ≈ 3 passes of work.', 1, 2)}
    ${sub('Reverse mode is why networks with billions of knobs can train at all.', 2)}`,
  frame(g, S) {
    const cols = 13, rows = 5, cs = 46, gap = 8;
    const grid = (x0, y0, lit, col) => {
      for (let i = 0; i < cols * rows; i++) {
        const cx = x0 + (i % cols) * (cs + gap), cy = y0 + Math.floor(i / cols) * (cs + gap), l = lit(i);
        D.rect(cx, cy, cs, cs, l > 0 ? mixHex('#141b26', col, l) : '#141b26', S.at(0, 800), l > 0 ? col : '#222b38', 1.5, 6);
      }
    };
    // finite differences: one knob at a time, twice
    const q = S.beat === 0 ? S.at(0, 4200, 500) : 1, n = Math.floor(q * 65);
    grid(150, 310, i => (i < n ? .35 : i === n && q < 1 ? 1 : 0), COL.loss);
    const fd = document.getElementById('fdc'); if (fd) fd.textContent = String(Math.min(130, Math.round(q * 130)));
    // backprop: one forward sweep, one backward sweep
    if (S.beat >= 1) {
      const r = S.beat === 1 ? S.at(1, 2400, 300) : 1;
      const fw = clamp(r * 2), bw = clamp(r * 2 - 1);
      grid(1010, 310, i => { const c = i % cols, f = c <= fw * cols ? 1 : 0, b = (cols - 1 - c) < bw * cols ? 1 : 0; return b ? .8 : f ? .25 : 0; }, COL.grad);
      const bp = document.getElementById('bpc'); if (bp) bp.textContent = r < 1 ? fmt(1 + 2 * r, 1) : '≈ 3';
      if (r < 1) { const x = fw < 1 ? 1010 + fw * cols * (cs + gap) : 1010 + (1 - bw) * cols * (cs + gap); D.seg(x, 296, x, 310 + rows * (cs + gap) + 6, fw < 1 ? COL.data : COL.grad, 4, 1); }
    }
  },
});

/* ------------------------------------------------------------ 4. gradient checking */
const GC = { L: w => (3 * Math.max(0, w) - 3) ** 2, w: 0.5, g: -9 };
Deck.add({
  id: 'grad-check', title: 'Check backprop with finite differences', beats: 3, dur: [3800, 5600, 4600], min: 2.5,
  notes: [
    'Finite differences are too slow to train with, but they are the perfect referee. Here is the loss as a function of w1 alone, with w2 held at 1.5. Backprop said the slope at w1 equals 0.5 is minus 9: the violet tangent.',
    'The forward difference: nudge by epsilon, take rise over run. With a big epsilon the pink secant is visibly off; as epsilon shrinks it swings toward the tangent. The error falls in proportion to epsilon. Here it is exactly 9 epsilon.',
    'The central difference nudges both ways and divides by two epsilon. Its error falls like epsilon squared; for this quadratic loss it is exact at any epsilon. In practice: when you write your own backward pass, check a handful of random knobs this way in double precision, and look for a relative error around ten to the minus six or smaller. Larger means a bug in the backward code.',
  ],
  html: () => `${hdr('Chapter 5', 'Trust backprop, but check it')}
    ${at(1180, 300, `<div class="lbl c-loss">forward difference</div><div class="eq-xs" style="margin-top:10px">${M(T`\frac{L(w+\varepsilon) - L(w)}{\varepsilon}\;\;\text{error} \propto \varepsilon`)}</div><div id="gfd" class="big mono c-loss" style="margin-top:10px"></div>`, 1, 'rise')}
    ${at(1180, 560, `<div class="lbl c-good">central difference</div><div class="eq-xs" style="margin-top:10px">${M(T`\frac{L(w+\varepsilon) - L(w-\varepsilon)}{2\varepsilon}\;\;\text{error} \propto \varepsilon^2`)}</div><div id="gcd" class="big mono c-good" style="margin-top:10px"></div>`, 2, 'rise')}
    ${at(1180, 300, '<div class="lbl c-grad">backprop</div><div class="num c-grad" style="margin-top:8px">−9</div>', 0, 'rise', '', 1)}
    ${sub('Loss against w₁ alone. Backprop says the slope is −9.', 0, 1)}
    ${sub('Nudge by ε: the secant swings onto the tangent as ε shrinks.', 1, 2)}
    ${sub('Nudge both ways: far more accurate. Check a few random knobs.', 2)}`,
  frame(g, S) {
    const P = new Plot(150, 270, 900, 600, [0, 1.25], [0, 9]);
    P.axes({ xt: [0, 0.25, 0.5, 0.75, 1, 1.25], yt: [0, 3, 6, 9], xl: 'w₁', yl: 'loss L', a: S.at(0, 800) });
    D.fn(P, GC.L, 0, lerp(0, 1.25, Ease.inOut(S.at(0, 1400))), COL.loss, 4, 1, 200, null, 10);
    const w = GC.w, L0 = GC.L(w), ta = S.at(0, 800, 1000);
    const tan = (s, x0, col, wdt, a) => D.seg(P.X(x0 - .45), P.Y(L0 + s * (x0 - .45 - w) + (GC.L(x0) - L0 - s * (x0 - w))), P.X(x0 + .45), P.Y(L0 + s * (x0 + .45 - w) + (GC.L(x0) - L0 - s * (x0 - w))), col, wdt, a);
    D.seg(P.X(w - .4), P.Y(L0 + .4 * 9), P.X(w + .4), P.Y(L0 - .4 * 9), COL.grad, 3, ta);
    D.dot(P.X(w), P.Y(L0), 9, COL.fg, ta);
    let eps = .4;
    if (S.beat >= 1) eps = lerp(.4, .02, Ease.inOut(S.beat === 1 ? S.at(1, 4200, 400) : 1));
    if (S.beat >= 1) {
      const s = (GC.L(w + eps) - L0) / eps;
      const a = S.beat === 1 ? 1 : .35;
      D.seg(P.X(w - .3), P.Y(L0 - .3 * s), P.X(w + eps + .3), P.Y(L0 + (eps + .3) * s), COL.loss, 3, a);
      D.dot(P.X(w + eps), P.Y(GC.L(w + eps)), 7, COL.loss, a);
      D.text('ε = ' + fmt(eps, 3), P.X(w + eps) + 14, P.Y(GC.L(w + eps)) - 22, { size: 22, col: COL.loss, a });
      const el = document.getElementById('gfd'); if (el) el.textContent = 'ε = ' + fmt(eps, 3) + '  →  ' + fmt(s, 3);
    }
    if (S.beat >= 2) {
      const e2 = lerp(.4, .02, Ease.inOut(S.at(2, 3000, 400)));
      const s = (GC.L(w + e2) - GC.L(w - e2)) / (2 * e2);
      D.seg(P.X(w - e2 - .15), P.Y(GC.L(w - e2) - .15 * s), P.X(w + e2 + .15), P.Y(GC.L(w + e2) + .15 * s), COL.good, 3, 1);
      D.dot(P.X(w - e2), P.Y(GC.L(w - e2)), 7, COL.good); D.dot(P.X(w + e2), P.Y(GC.L(w + e2)), 7, COL.good);
      const el = document.getElementById('gcd'); if (el) el.textContent = 'ε = ' + fmt(e2, 3) + '  →  ' + fmt(s, 3);
    }
  },
});

/* ------------------------------------------------------------ 5. vanishing gradients */
Deck.add({
  id: 'vanishing', title: 'Gradients vanish through sigmoids', beats: 3, dur: [4200, 5200, 4600], min: 3,
  notes: [
    'The chain rule has a dark side. Stack ten sigmoid layers. On the way back, every layer multiplies the gradient by the sigmoid\'s slope, and the sigmoid\'s slope is never more than one quarter.',
    'So the gradient shrinks by at least a factor of four per layer. After ten layers, 0.25 to the tenth is about ten to the minus six. The layers near the input receive essentially no signal: they barely learn. This is the vanishing-gradient problem, and it is why deep networks were considered untrainable through the 1990s.',
    'The fixes are now standard. ReLU has slope exactly one wherever it is active, so the gradient passes through undamped. Residual connections add a shortcut around each layer, which gives the gradient a path with slope one. Careful initialisation and normalisation layers keep signals in a healthy range. Every modern deep network, transformers included, uses these.',
  ],
  html: () => `${hdr('Chapter 5', 'Ten sigmoids swallow the gradient')}
    ${at(1180, 250, `<div class="eq-s">${M(T`\sigma'(z) = \sigma(z)\,(1-\sigma(z)) \le \tfrac14`)}</div>`, 0, 'rise', '', 2)}
    ${at(150, 250, `<div class="eq-s">${M(T`\frac{\partial L}{\partial a_1} = \prod_{k=1}^{10} \sigma'(z_k)\,w_k`)}</div>`, 0, 'rise')}
    ${at(1180, 250, '<div class="lede c-model" style="width:640px">ReLU has slope 1 where active.<br>Residual shortcuts and normalisation help too.</div>', 2, 'rise')}
    ${sub('Every layer multiplies the backward signal by σ′ ≤ ¼.', 0, 1)}
    ${sub('Ten layers: 0.25¹⁰ ≈ 10⁻⁶. <b>The first layers barely learn.</b>', 1, 2)}
    ${sub('ReLU and shortcut connections keep the gradient alive.', 2)}`,
  frame(g, S) {
    const n = 10, x0 = 230, dx = 160, ny = 440, base = 900, top = 560;
    const a0 = S.at(0, 1000);
    for (let k = 0; k < n; k++) {
      const x = x0 + k * dx;
      if (k) D.seg(x - dx + 26, ny, x - 26, ny, COL.faint, 2, a0);
      D.ring(x, ny, 24, S.beat >= 2 ? COL.model : COL.loss, 3, clamp(a0 * 3 - k * .2));
      D.text('σ', x, ny, { size: 26, align: 'center', col: COL.fg, font: '"IBM Plex Sans"', a: S.beat >= 2 ? 0 : a0 });
      D.text('R', x, ny, { size: 24, align: 'center', col: COL.fg, font: '"IBM Plex Sans"', a: S.beat >= 2 ? S.at(2, 600) : 0 });
    }
    D.text('input', x0, ny - 54, { size: 22, align: 'center', a: a0 }); D.text('loss', x0 + (n - 1) * dx, ny - 54, { size: 22, align: 'center', a: a0 });
    // log-scale bar axis: 1 at top, 1e-7 at base
    const Y = v => lerp(base, top, (Math.log10(v) + 7) / 7);
    if (S.beat >= 1) {
      const aa = S.at(1, 600);
      [1, 1e-2, 1e-4, 1e-6].forEach(v => { D.seg(x0 - 70, Y(v), x0 + (n - 1) * dx + 40, Y(v), COL.grid, 1, aa); D.text(v === 1 ? '1' : '10' + ['⁻²', '⁻⁴', '⁻⁶'][[1e-2, 1e-4, 1e-6].indexOf(v)], x0 - 84, Y(v), { size: 20, align: 'right', a: aa }); });
      D.text('gradient size reaching each layer (log scale)', x0 - 84, top - 62, { size: 20, a: aa });
      const q = S.beat === 1 ? S.at(1, 3600, 300) : 1, front = (n - 1) - q * (n - 1);
      for (let k = n - 1; k >= 0; k--) {
        if (k < front - 1e-6) continue;
        const gS = Math.pow(0.25, n - k), gR = 1;
        const x = x0 + k * dx;
        const mix = S.beat >= 2 ? Ease.inOut(S.at(2, 1400, 300)) : 0;
        const v = Math.pow(10, lerp(Math.log10(gS), Math.log10(gR), mix));
        D.rect(x - 30, Y(v), 60, base - Y(v), mixHex(COL.loss, COL.model, mix), .85);
        if (k === 0 || k === n - 1 || k === 5) D.text(v > .99 ? '1' : sci(v), x, Y(v) - 22, { size: 22, align: 'center', col: COL.fg, a: .9 });
      }
      if (q < 1) D.dot(x0 + front * dx, ny, 10, COL.grad, 1, 18);
    }
  },
});

/* ------------------------------------------------------------ 6. the whole thing in numpy */
const NP_CODE = [
  [0, '<span class="k">import</span> numpy <span class="k">as</span> np'],
  [0, 'W1 = rng.normal(<span class="n">0</span>, <span class="n">.5</span>, (<span class="n">2</span>, <span class="n">16</span>));  b1 = np.zeros(<span class="n">16</span>)'],
  [0, 'w2 = rng.normal(<span class="n">0</span>, <span class="n">.5</span>, <span class="n">16</span>);       b2 = <span class="n">0.0</span>'],
  [0, '<span class="k">for</span> epoch <span class="k">in</span> range(<span class="n">2000</span>):'],
  [1, '    <span class="c"># forward: keep every intermediate</span>'],
  [1, '    z1 = X @ W1 + b1'],
  [1, '    h  = np.maximum(<span class="n">0</span>, z1)'],
  [1, '    p  = <span class="n">1</span> / (<span class="n">1</span> + np.exp(-(h @ w2 + b2)))'],
  [2, '    <span class="c"># backward: same lines, reversed</span>'],
  [2, '    dz2 = (p - y) / len(y)'],
  [2, '    dw2 = h.T @ dz2;      db2 = dz2.sum()'],
  [2, '    dh  = np.outer(dz2, w2)'],
  [2, '    dz1 = dh * (z1 > <span class="n">0</span>)'],
  [2, '    dW1 = X.T @ dz1;      db1 = dz1.sum(<span class="n">0</span>)'],
  [3, '    <span class="c"># update</span>'],
  [3, '    W1 -= lr * dW1;  b1 -= lr * db1'],
  [3, '    w2 -= lr * dw2;  b2 -= lr * db2'],
];
Deck.add({
  id: 'numpy', title: 'Backprop in fifteen lines of numpy', beats: 4, dur: [3600, 4200, 3600, 4200], min: 3.5,
  notes: [
    'Here is the 2-16-1 network trained on the operating-window data, in plain numpy, no framework. The forward pass is three lines: weighted sum, ReLU, sigmoid. We keep z1, h and p because the backward pass will need them.',
    'The backward pass mirrors the forward pass line by line, in reverse. dz2 is p minus y: that is the gift from chapter 3, sigmoid plus cross-entropy gives the simplest possible gradient. dw2 uses the stored h. dh sends the blame back through w2. Multiplying by (z1 greater than 0) is the ReLU slope, zero or one. dW1 uses the inputs. Exactly the by-hand example, with matrices.',
    'The update is gradient descent, the same line as chapter 2. That is a complete neural-network trainer. You will run this in the exercises.',
    'In practice nobody writes the backward pass by hand. PyTorch, JAX and TensorFlow record the forward operations and run reverse mode for you: loss.backward() is these six lines, generated automatically for any network you build. But now you know exactly what it does.',
  ],
  html: () => `${hdr('Chapter 5', 'The whole thing in numpy')}
    ${at(96, 250, `<div class="code" id="npc" style="font-size:22px;line-height:1.5;padding:22px 30px">${NP_CODE.map(([gp, l]) => `<div class="ln" data-g="${gp}">${l || ' '}</div>`).join('')}</div>`, 0, 'rise')}
    ${at(1190, 300, '<div class="lbl c-data">forward</div><div class="lede" style="margin-top:10px;width:620px">weighted sum, ReLU, sigmoid. Store z1, h, p.</div>', 0, 'rise', '', 1)}
    ${at(1190, 300, `<div class="lbl c-grad">backward</div><div class="lede" style="margin-top:10px;width:620px"><span class="mono">p − y</span>: sigmoid + cross-entropy, from chapter 3.<br><br><span class="mono">(z1 &gt; 0)</span>: the ReLU slope, 0 or 1.<br><br>each line mirrors a forward line</div>`, 1, 'rise', '', 2)}
    ${at(1190, 300, '<div class="lbl c-model">update</div><div class="lede" style="margin-top:10px;width:620px">plain gradient descent, chapter 2.</div>', 2, 'rise', '', 3)}
    ${at(1190, 300, `<div class="lbl c-good">in PyTorch</div><div class="code" style="font-size:22px;margin-top:14px">loss = bce(model(X), y)
loss.<span class="f">backward</span>()   <span class="c"># all of it</span>
opt.<span class="f">step</span>()</div><div class="lede" style="margin-top:20px;width:620px">autograd records the forward pass and runs reverse mode for you.</div>`, 3, 'rise')}
    ${sub('Forward: three lines. Keep the intermediates.', 0, 1)}
    ${sub('Backward: each line mirrors a forward line, in reverse.', 1, 2)}
    ${sub('Update: the gradient-descent step from chapter 2.', 2, 3)}
    ${sub('<b>loss.backward()</b> writes the backward pass for you.', 3)}`,
  frame(g, S) {
    const box = document.getElementById('npc'); if (!box) return;
    const active = S.beat + 1, f = S.p(500);
    box.querySelectorAll('.ln').forEach(el => {
      const gp = +el.dataset.g, on = S.beat === 3 ? false : gp === active;
      el.style.opacity = on ? 1 : S.beat === 3 ? .55 : lerp(1, .32, f);
      el.style.background = on ? `rgba(180,140,255,${.12 * f})` : 'transparent';
    });
  },
});

quizScene('quiz-5', 'Chapter 5', [
  ['Why does training a network need much more memory than using it?', 'The backward pass multiplies by the stored forward values (like h). Every activation must be kept until it is used.'],
  ['The first layers of your 30-layer sigmoid network never change. Why, and what would you try?', 'Vanishing gradients. Switch to ReLU, add residual connections or normalisation.'],
], ['Memory.', 'Look back at the by-hand example: dL/dw2 needed h. Training keeps every intermediate in memory until the backward pass is done; inference can throw them away immediately.', 'Second question.', 'Each sigmoid multiplies the gradient by at most one quarter; after thirty layers nothing reaches the input side. ReLU, residual connections and normalisation are the standard cures.']);
