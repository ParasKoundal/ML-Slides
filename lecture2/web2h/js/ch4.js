/* ch4.js: Stacking neurons. One line cannot draw a window; a neuron; why the bend matters;
 * ReLU hinges approximate Haldane kinetics; a 2-16-1 network; hidden layers bend space. */
'use strict';

Deck.chapter('Stacking neurons', 'networks, approximation and bending space');

const PW = new Plot(150, 270, 960, 580, [30, 38], [6.6, 7.4]);
const winPoints = (P = PW, a = 1, r = 7) => WIN.X.forEach(([T_, pH], i) => D.dot(P.X(T_), P.Y(pH), r, WIN.y[i] ? COL.cls1 : COL.cls0, a, 5));
const winAxes = (P = PW, a = 1) => P.axes({ xt: [30, 32, 34, 36, 38], yt: [6.6, 7.0, 7.4], xl: 'temperature  (°C)', yl: 'pH', a, yf: v => v.toFixed(1) });
const FIELD_STOPS = ['#0b2347', '#12345a', '#2a2f45', '#5a3b2c', '#8a5a1f'];

/* logistic regression on the window data, for the "one line fails" moment */
const WIN_LOG = (() => {
  let w = [0, 0], b = 0;
  for (let k = 0; k < 3000; k++) { let g0 = 0, g1 = 0, gb = 0; WIN.Z.forEach(([a, c], i) => { const d = sigmoid(w[0] * a + w[1] * c + b) - WIN.y[i]; g0 += d * a; g1 += d * c; gb += d; }); w[0] -= .5 * g0 / WIN.n; w[1] -= .5 * g1 / WIN.n; b -= .5 * gb / WIN.n; }
  const p = (T_, pH) => { const [a, c] = WIN.z([T_, pH]); return sigmoid(w[0] * a + w[1] * c + b); };
  const acc = mean(WIN.X.map((x, i) => ((p(...x) > .5 ? 1 : 0) === WIN.y[i] ? 1 : 0)));
  return { w, b, p, acc, base: 1 - mean(WIN.y) };
})();
/* the trained 2-16-1 network used for the diagram */
const NET16 = (() => { const net = new MLP([2, 16, 1], 'relu', 'sigmoid', 4); for (let e = 0; e < 600; e++) net.step(WIN.Z, WIN.y, 0.5, 0.9); return net; })();

chapterCard(4, 'Stacking neurons', 'networks, approximation and bending space',
  'Chapter four. What happens when one straight boundary is not enough.',
  (g, S) => {
    const xs = [1360, 1560, 1760], ys = [[300, 420], [240, 320, 400, 480], [360]];
    const p = S.p(2600, 300);
    ys[0].forEach(y0 => ys[1].forEach(y1 => D.seg(xs[0], y0, xs[1], y1, COL.model, 1.5, .35 * p)));
    ys[1].forEach(y1 => D.seg(xs[1], y1, xs[2], ys[2][0], COL.model, 1.5, .35 * p));
    ys.forEach((col, k) => col.forEach(y => D.ring(xs[k], y, 20, k === 1 ? COL.data : COL.model, 3, p)));
  });

/* ------------------------------------------------------------ 1. the operating window: one line fails */
Deck.add({
  id: 'window', title: 'One line cannot draw a window', beats: 3, dur: [3800, 4200, 4800], min: 2.5,
  notes: [
    'A new data set, and the one we will use for the rest of the lecture. 240 simulated batches at different temperatures and pH. The amber ones worked, and they sit in an operating window: too cold, too hot, too acidic or too basic and the batch fails. Every process engineer has drawn a picture like this.',
    'Logistic regression does its best with one straight boundary. It reports about ' + Math.round(100 * WIN_LOG.acc) + ' percent accuracy. That sounds respectable.',
    'It is not. Only ' + Math.round(100 * mean(WIN.y)) + ' percent of batches are good, so a model that calls every batch bad scores ' + Math.round(100 * WIN_LOG.base) + ' percent too. That is what it has done: one straight line cannot enclose a window. Two lessons: accuracy can lie on imbalanced data, so look at the boundary and at other metrics; and we need a model that can bend.',
  ],
  html: () => `${hdr('Chapter 4', 'One straight line cannot draw a window')}
    ${at(1250, 330, `<div class="lbl">logistic regression</div><div class="num c-model" style="margin-top:12px;font-size:120px">${Math.round(100 * WIN_LOG.acc)}%</div>`, 1, 'rise')}
    ${at(1250, 590, `<div class="lbl">"every batch is bad"</div><div class="num c-loss" style="margin-top:12px;font-size:120px">${Math.round(100 * WIN_LOG.base)}%</div>`, 2, 'pop')}
    ${sub('240 batches (simulated). Good ones (amber) sit in an operating window.', 0, 1)}
    ${sub('The best single straight boundary claims ' + Math.round(100 * WIN_LOG.acc) + '% accuracy.', 1, 2)}
    ${sub('So does saying "bad" every time. <b>One line cannot enclose a window.</b>', 2)}`,
  enter(st) { st.img = heatImage(PW, WIN_LOG.p, { res: 6, lo: 0, hi: 1, stops: FIELD_STOPS }); },
  frame(g, S) {
    if (S.beat >= 1) { drawImg(S.st.img, PW, .95 * S.at(1, 800)); drawSegs(contourSegs(PW, WIN_LOG.p, .5), COL.data, 4, S.at(1, 800, 400), 12); }
    winAxes(); winPoints(PW, clamp(S.at(0, 1500) * 2));
  },
});

/* ------------------------------------------------------------ 2. one neuron, and its bend */
const ACTS = [['sigmoid', z => sigmoid(z), COL.loss, M(T`\sigma(z) = \frac{1}{1+e^{-z}}`)], ['tanh', z => Math.tanh(z), COL.grad, M(T`\tanh(z) = \frac{e^{z}-e^{-z}}{e^{z}+e^{-z}}`)], ['ReLU', z => Math.max(0, z), COL.data, M(T`\mathrm{ReLU}(z) = \max(0, z)`)]];
Deck.add({
  id: 'neuron', title: 'A neuron: weigh, add, bend', beats: 3, dur: [4400, 3600, 4200], min: 2.5,
  notes: [
    'Meet the neuron. It takes its inputs, multiplies each by a weight, adds them up with a bias, and then bends the result with a fixed function g, the activation. That is all. Logistic regression is exactly one neuron, with the sigmoid as its bend.',
    'There are several common bends. The sigmoid squashes to 0 to 1; tanh squashes to minus 1 to 1 and is centred on zero.',
    'And ReLU, the rectified linear unit: zero for negative inputs, the input itself for positive ones. It is the default inside modern networks because it is cheap, and because its slope is exactly 1 wherever it is switched on, which keeps gradients alive through many layers. More on that in Lecture 3.',
  ],
  html: () => `${hdr('Chapter 4', 'A neuron: weigh, add, bend')}
    ${at(170, 760, `<div class="eq-s">${M(T`a = g\big(\color{#3fd0ff}{w_1}x_1 + \color{#3fd0ff}{w_2}x_2 + \color{#3fd0ff}{b}\big)`)}</div>`, 0, 'rise', '--d:500ms')}
    ${ACTS.map(([n, , c, tex], k) => at(1290, 720, `<div class="h3" style="color:${c}">${n}</div><div class="eq-xs" style="margin-top:8px">${tex}</div>`, k, 'rise', '', k < 2 ? k + 1 : undefined)).join('')}
    ${sub('Weigh the inputs, add, bend. Logistic regression is one neuron.', 0, 1)}
    ${sub('The bend is called the activation.', 1, 2)}
    ${sub('ReLU: max(0, z). The default inside modern networks.', 2)}`,
  frame(g, S) {
    const nodes = { x1: [260, 380], x2: [260, 600], s: [640, 490], o: [960, 490] };
    const p = S.at(0, 1200, 200);
    [['x1', 1.4], ['x2', -0.9]].forEach(([k, w]) => {
      const [x, y] = nodes[k];
      D.line([[x + 44, y], [lerp(x + 44, nodes.s[0] - 60, p), lerp(y, nodes.s[1], p)]], w > 0 ? COL.model : COL.loss, 2 + 2 * Math.abs(w), .8);
      for (let j = 0; j < 3; j++) { const q = ((S.t / 1300 + j / 3) % 1); D.dot(lerp(x + 44, nodes.s[0] - 60, q), lerp(y, nodes.s[1], q), 5, COL.data, p * Math.sin(Math.PI * q)); }
    });
    D.ring(nodes.x1[0], nodes.x1[1], 44, COL.model, 3, p); D.ring(nodes.x2[0], nodes.x2[1], 44, COL.model, 3, p);
    D.text('T', nodes.x1[0], nodes.x1[1], { size: 34, align: 'center', col: COL.fg, font: '"IBM Plex Sans"', weight: 600, a: p });
    D.text('pH', nodes.x2[0], nodes.x2[1], { size: 30, align: 'center', col: COL.fg, font: '"IBM Plex Sans"', weight: 600, a: p });
    D.ring(nodes.s[0], nodes.s[1], 62, COL.data, 4, p); D.text('Σ', nodes.s[0] - 18, nodes.s[1], { size: 50, align: 'center', col: COL.fg, a: p });
    // a tiny glyph of the current bend inside the neuron
    const k = S.beat, f = Ease.inOut(S.p(1200)), fn = z => (k === 0 ? ACTS[0][1](z) : lerp(ACTS[k - 1][1](z), ACTS[k][1](z), f));
    D.line(linspace(-3, 3, 40).map(z => [nodes.s[0] + 14 + z * 7, nodes.s[1] + 6 - clamp(fn(z), -1.2, 3) * 14]), COL.fg, 2.5, p);
    D.line([[nodes.s[0] + 62, nodes.s[1]], [nodes.o[0] - 44, nodes.o[1]]], COL.model, 4, p);
    D.ring(nodes.o[0], nodes.o[1], 44, COL.model, 3, p); D.text('a', nodes.o[0], nodes.o[1], { size: 34, align: 'center', col: COL.fg, font: '"IBM Plex Sans"', weight: 600, a: p });
    // the bend, large
    const P = new Plot(1290, 290, 480, 360, [-4, 4], [-1.3, 3.2]);
    P.axes({ xt: [-4, 0, 4], yt: [-1, 0, 1, 2, 3], xl: 'z', yl: 'g(z)', grid: true });
    D.fn(P, fn, -4, 4, ACTS[k][2], 5, 1, 160, null, 12);
  },
});

/* ------------------------------------------------------------ 3. why the bend matters */
const LN = [[1.2, -0.2, 0.9], [-0.8, 0.3, 1.1], [0.6, -0.5, -1.4]];  // slope, intercept, output weight
Deck.add({
  id: 'collapse', title: 'Without the bend, layers collapse', beats: 2, dur: [4600, 4600], min: 2,
  notes: [
    'Why bother with the bend? Take three neurons with no activation. Each is a straight line in its input. The next layer adds them up with weights: the result, the thick line, is another straight line. Stack ten layers like that and you still have one straight line, because the product of matrices is just another matrix.',
    'Now put a ReLU on each. Each line becomes a hinge: flat, then rising, at its own place. Add the hinges and you get a curve with corners. More neurons, more corners. That is the entire secret of why networks can fit shapes that a linear model cannot.',
  ],
  html: () => `${hdr('Chapter 4', 'Without the bend, layers collapse')}
    ${at(150, 255, '<div class="lbl">three neurons, no bend</div>', 0)}
    ${at(1010, 255, '<div class="lbl c-data">the same three, with ReLU</div>', 1)}
    ${at(150, 790, `<div class="eq-s">${M(T`W_2(W_1x) = (W_2W_1)\,x`)}</div><div class="lede" style="font-size:26px;margin-top:8px">still one straight line</div>`, 0, 'rise', '--d:2400ms')}
    ${at(1010, 790, `<div class="eq-s">${M(T`\textstyle\sum_k v_k\,\mathrm{ReLU}(a_kx + c_k)`)}</div><div class="lede" style="font-size:26px;margin-top:8px">corners: any shape you like</div>`, 1, 'rise', '--d:2400ms')}
    ${sub('Lines added to lines stay a line, however many layers.', 0, 1)}
    ${sub('Bend each one first, and the sum can take any shape.', 1)}`,
  frame(g, S) {
    const cols = [COL.model, COL.grad, COL.good];
    const draw = (P, bent, p) => {
      P.axes({ xt: [], yt: [], a: .6 }); D.seg(P.x, P.Y(0), P.x + P.w, P.Y(0), '#2a3240', 1);
      const term = (k, x) => { const [a, c, v] = LN[k]; const z = a * x + c; return v * (bent ? Math.max(0, z) : z); };
      LN.forEach((_, k) => D.fn(P, x => term(k, x), -2, 2, cols[k], 2.5, .6 * clamp(p * 3 - k * .4)));
      const q = clamp(p * 2 - 1.1);
      if (q > 0) D.fn(P, x => term(0, x) + term(1, x) + term(2, x), -2, lerp(-2, 2, Ease.inOut(q)), bent ? COL.data : COL.fg, 6, 1, 200, null, 14);
    };
    draw(new Plot(150, 300, 780, 440, [-2, 2], [-3, 3]), false, S.at(0, 2600, 200));
    if (S.beat >= 1) draw(new Plot(1010, 300, 780, 440, [-2, 2], [-3, 3]), true, S.at(1, 2600, 200));
  },
});

/* ------------------------------------------------------------ 4. hinges approximate Haldane kinetics */
const PH = new Plot(150, 270, 1000, 560, [0, 20], [-0.02, 0.32]);
const HINGE_K = [1, 3, 6, 16];
Deck.add({
  id: 'hinges', title: 'Enough hinges fit any curve', beats: 4, dur: [4200, 3400, 3400, 4200], min: 3,
  notes: [
    'Let us fit something with real curvature: substrate-inhibition kinetics. The growth rate rises with substrate, peaks, then falls as substrate becomes inhibitory: the Haldane law. 26 noisy simulated measurements; the network never sees the equation. One ReLU neuron gives one hinge: a straight line with one corner.',
    'Three neurons, three hinges. The violet ticks mark where each corner sits.',
    'Six neurons: the peak and the inhibition are already there.',
    'Sixteen: it follows the curve closely. This is the universal approximation theorem in pictures: with enough neurons, one hidden layer can approximate any continuous function on a bounded range as closely as you like. A detail for the experts: here I froze the corner positions and solved only the output weights, which is just linear regression on hinge features. Full training moves the corners too, by gradient descent. Note what the theorem does not promise: anything about outside the range, or how much data you need. Chapter 6.',
  ],
  html: () => `${hdr('Chapter 4 · kinetics', 'Each neuron adds one corner')}
    ${at(1260, 290, `<div class="eq-xs">${M(T`\mu = \frac{\mu_{\max}S}{K_s + S + S^2/K_i}`)}</div><div class="lede" style="font-size:25px;margin-top:6px">true law (dashed) · never shown to the network</div>`, 0, 'rise')}
    ${at(1260, 470, `<div class="eq-xs">${M(T`\hat\mu(S) = b + \sum_{k=1}^{K} \color{#3fd0ff}{c_k}\,\mathrm{ReLU}(S - s_k)`)}</div>`, 0, 'rise', '--d:300ms')}
    ${at(1260, 600, '<div class="lbl">neurons</div><div id="hk" class="num c-data" style="margin-top:8px;font-size:120px"></div><div class="lbl" style="margin-top:12px">mean squared error</div><div id="hm" class="mono c-loss" style="font-size:40px;margin-top:6px"></div>', 0, 'rise', '--d:500ms')}
    ${sub('Substrate inhibition, 26 noisy points (simulated). One neuron: one corner.', 0, 1)}
    ${sub('Three neurons, three corners.', 1, 2)}${sub('Six: the peak and the inhibition appear.', 2, 3)}
    ${sub('Sixteen: enough corners fit any smooth curve on this range.', 3)}`,
  frame(g, S) {
    PH.axes({ xt: [0, 5, 10, 15, 20], yt: [0, 0.1, 0.2, 0.3], xl: 'substrate S  (g/L)', yl: 'specific growth rate μ  (1/h)', grid: true });
    D.fn(PH, HAL.mu, 0, 20, COL.dim, 2, .8, 200, [8, 8]);
    HAL.S.forEach((s, i) => D.dot(PH.X(s), PH.Y(HAL.Y[i]), 8, COL.data, 1, 8));
    const k = HINGE_K[S.beat], fit = HAL.fitRelu(k), prev = S.beat ? HAL.fitRelu(HINGE_K[S.beat - 1]) : null;
    const f = Ease.inOut(S.p(1400, 200)), fn = s => (prev ? lerp(prev.f(s), fit.f(s), f) : fit.f(s) * Ease.out(S.at(0, 1200, 400)));
    D.fn(PH, fn, 0, 20, COL.model, 5, 1, 300, null, 14);
    fit.knots.forEach(c => D.seg(PH.X(c), PH.Y(fn(c)) - 14, PH.X(c), PH.Y(fn(c)) + 14, COL.grad, 3, f));
    const mse = mean(HAL.S.map((s, i) => (fit.f(s) - HAL.Y[i]) ** 2));
    const ek = document.getElementById('hk'), em = document.getElementById('hm');
    if (ek) ek.textContent = String(k); if (em) em.textContent = mse.toExponential(1).replace('e-', ' × 10⁻').replace('⁻4', '⁻⁴').replace('⁻3', '⁻³').replace('⁻5', '⁻⁵');
  },
});

/* ------------------------------------------------------------ 5. a whole network, and its knobs */
const NET_X = [34.3, 7.05];
Deck.add({
  id: 'network', title: 'A network and its 65 knobs', beats: 3, dur: [4200, 4600, 4200], min: 3,
  notes: [
    'Now stack them. Two inputs, temperature and pH, standardised. A hidden layer of 16 ReLU neurons, each its own weighted sum and bend. One output neuron with a sigmoid that gives the probability the batch is good. Every line is one weight: cyan positive, pink negative, thicker is bigger. These are the actual trained weights.',
    'A forward pass for one batch, at 34.3 degrees and pH 7.05. The inputs flow into the hidden layer; some neurons light up, others stay at zero because their ReLU is off. The output neuron weighs the active ones and says: good batch, with high probability.',
    'In matrix form it is two lines. h equals ReLU of W1 x plus b1; p equals sigma of w2 dot h plus b2. Counting knobs: 2 times 16 weights plus 16 biases in the first layer, 16 weights plus 1 bias in the second: 65 knobs. GPT-3 has 175 billion, trained with the same loop. Next chapter: how to get the gradient for all of them at once.',
  ],
  html: () => `${hdr('Chapter 4', 'A network is neurons feeding neurons')}
    ${at(1320, 280, '<div class="lbl">this batch</div><div class="big mono" style="margin-top:8px">T = 34.3 °C · pH 7.05</div><div class="lbl" style="margin-top:22px">p(good)</div><div id="np" class="num c-good" style="margin-top:8px;font-size:110px"></div>', 1, 'rise')}
    ${at(1320, 640, `<div class="eq-xs">${M(T`\mathbf{h} = \mathrm{ReLU}(W_1\mathbf{x} + \mathbf{b}_1)`)}</div><div class="eq-xs" style="margin-top:14px">${M(T`p = \sigma(\mathbf{w}_2\!\cdot\!\mathbf{h} + b_2)`)}</div><div class="big" style="margin-top:18px">2×16 + 16 + 16 + 1 = <span class="c-data mono">65</span> knobs</div>`, 2, 'rise')}
    ${sub('2 inputs → 16 ReLU neurons → 1 output. Every line is a knob.', 0, 1)}
    ${sub('A forward pass: numbers flow left to right.', 1, 2)}
    ${sub('65 knobs here. GPT-3: 175 billion. Same loop.', 2)}`,
  frame(g, S) {
    const net = NET16, xin = [380, 1180, 380], X = WIN.z(NET_X), fw = net.forward(X), hs = fw.as[1];
    const inY = [440, 700], hidY = i => 270 + i * 40, outY = 570, hx = 760, ox = 1140;
    const p = S.at(0, 1600, 100);
    const wmax1 = Math.max(...net.W[0].flat().map(Math.abs)), wmax2 = Math.max(...net.W[1][0].map(Math.abs));
    const fwd = S.beat >= 1 ? S.at(1, 2400, 300) : 0;
    net.W[0].forEach((row, j) => row.forEach((w, i) => {
      const q = clamp(p * 1.6 - j * .03);
      D.line([[xin[0] + 36, inY[i]], [lerp(xin[0] + 36, hx - 20, q), lerp(inY[i], hidY(j), q)]], w > 0 ? COL.model : COL.loss, .6 + 4 * Math.abs(w) / wmax1, .25 + .5 * Math.abs(w) / wmax1);
      if (fwd > 0 && fwd < .5) { const t = fwd * 2; D.dot(lerp(xin[0] + 36, hx - 20, t), lerp(inY[i], hidY(j), t), 4, COL.data, .9); }
    }));
    net.W[1][0].forEach((w, j) => {
      const q = clamp(p * 1.6 - .5 - j * .02);
      D.line([[hx + 20, hidY(j)], [lerp(hx + 20, ox - 40, q), lerp(hidY(j), outY, q)]], w > 0 ? COL.model : COL.loss, .6 + 4 * Math.abs(w) / wmax2, .25 + .5 * Math.abs(w) / wmax2);
      if (fwd >= .5 && fwd < 1 && hs[j] > 0) { const t = (fwd - .5) * 2; D.dot(lerp(hx + 20, ox - 40, t), lerp(hidY(j), outY, t), 4 + 3 * Math.min(1, hs[j]), COL.data, .9); }
    });
    ['T', 'pH'].forEach((l, i) => { D.ring(xin[0], inY[i], 36, COL.model, 3, p); D.text(l, xin[0], inY[i], { size: 28, align: 'center', col: COL.fg, font: '"IBM Plex Sans"', weight: 600, a: p }); });
    const hmax = Math.max(...hs, 1e-6);
    for (let j = 0; j < 16; j++) {
      const lit = fwd > .45 ? hs[j] / hmax : 0;
      D.dot(hx, hidY(j), 16, mixHex('#141b26', COL.data, lit), p); D.ring(hx, hidY(j), 16, COL.data, 2, p * (.5 + .5 * (lit > 0)));
    }
    D.ring(ox, outY, 40, COL.good, 3, p); D.text('p', ox, outY, { size: 30, align: 'center', col: COL.fg, font: '"IBM Plex Sans"', weight: 600, a: p });
    D.text('inputs', xin[0], 800, { size: 22, align: 'center', a: p }); D.text('16 hidden ReLU neurons', hx, 930 - 24, { size: 22, align: 'center', a: p }); D.text('output', ox, outY + 70, { size: 22, align: 'center', a: p });
    const el = document.getElementById('np'); if (el) el.textContent = fwd >= .95 || S.beat > 1 ? fmt(fw.y, 2) : '…';
  },
});

/* ------------------------------------------------------------ 6. hidden layers bend space */
Deck.add({
  id: 'warp', title: 'Hidden layers bend space', beats: 3, dur: [3800, 6200, 5600], min: 3.5,
  notes: [
    'Here is the deepest picture in this chapter. Take a tiny network with three tanh neurons, trained on the window data; it gets ' + Math.round(100 * WARP.acc) + ' percent right. Start with the input plane: temperature and pH, the grid of every possible batch, amber inside the window and blue outside. No flat cut separates the amber island from the blue sea.',
    'Now move every point to where the hidden layer sends it: the three neuron outputs become its new coordinates in three dimensions. Watch the flat sheet fold and curl. The network is learning a new coordinate system for your process.',
    'In the new coordinates a single flat plane, the output neuron, separates good from bad. That is what every hidden layer does: it bends space until the problem becomes linear for the next layer. Deep networks just do this many times in a row. Remember this picture when we talk about embeddings in Lecture 3 and attention in Lecture 4.',
  ],
  html: () => `${hdr('Chapter 4', 'Bend space, then cut')}
    ${sub('Input space: no flat cut separates the amber island.', 0, 1)}
    ${sub('Move each point to its hidden-layer coordinates. The sheet folds.', 1, 2)}
    ${sub('Now one flat plane separates them. <b>Bend, then cut.</b>', 2)}`,
  enter(st) {
    st.cam = new Cam(960, 640, 250, -0.7, 0.42, 0.1);
    const net = WARP.net, us = linspace(-1.9, 1.9, 15);
    const H = (u, v) => net.W[0].map((row, j) => Math.tanh(row[0] * u + row[1] * v + net.b[0][j]));
    st.lines = [...us.map(u => linspace(-1.9, 1.9, 40).map(v => [u, v])), ...us.map(v => linspace(-1.9, 1.9, 40).map(u => [u, v]))];
    st.H = H; st.pts = WIN.Z.map((z, i) => ({ z, y: WIN.y[i] }));
  },
  frame(g, S) {
    const st = S.st, cam = st.cam;
    const f = S.beat === 0 ? 0 : Ease.inOut(S.at(1, 4200, 400));
    cam.yaw = -0.7 + S.t / 9000; cam.pitch = lerp(1.05, .42 + .1 * Math.sin(S.t / 5000), f); cam.scale = lerp(270, 232, f); cam.cy = lerp(450, 585, f);
    const pos = (u, v) => { const h = st.H(u, v), a = [u * .52, v * .52, -0.95]; return cam.p(...a.map((c, i) => lerp(c, h[i], f))); };
    // faint unit cube for orientation
    const cube = [[-1, -1, -1], [1, -1, -1], [1, 1, -1], [-1, 1, -1], [-1, -1, 1], [1, -1, 1], [1, 1, 1], [-1, 1, 1]];
    [[0, 1], [1, 2], [2, 3], [3, 0], [4, 5], [5, 6], [6, 7], [7, 4], [0, 4], [1, 5], [2, 6], [3, 7]].forEach(([a, b]) => D.line([cam.p(...cube[a]), cam.p(...cube[b])].map(q => [q[0], q[1]]), '#1d2430', 1.2, f));
    st.lines.forEach(L => D.line(L.map(([u, v]) => pos(u, v)).map(q => [q[0], q[1]]), '#3a4a60', 1.2, .55));
    st.pts.forEach(({ z, y }) => { const q = pos(z[0], z[1]); D.dot(q[0], q[1], 6, y ? COL.cls1 : COL.cls0, 1, 6); });
    if (S.beat >= 2) {
      const w = WARP.net.W[1][0], b = WARP.net.b[1][0], n = Math.hypot(...w), nn = w.map(v => v / n), c0 = nn.map(v => -b / n * v);
      let e1 = Math.abs(nn[0]) < .9 ? [1, 0, 0] : [0, 1, 0]; const dot = e1.reduce((s, v, i) => s + v * nn[i], 0);
      e1 = e1.map((v, i) => v - dot * nn[i]); const l1 = Math.hypot(...e1); e1 = e1.map(v => v / l1);
      const e2 = [nn[1] * e1[2] - nn[2] * e1[1], nn[2] * e1[0] - nn[0] * e1[2], nn[0] * e1[1] - nn[1] * e1[0]];
      const R = 1.05, pa = S.at(2, 1000);
      const corner = (a, b2) => cam.p(...c0.map((c, i) => c + a * R * e1[i] + b2 * R * e2[i]));
      D.poly([corner(-1, -1), corner(1, -1), corner(1, 1), corner(-1, 1)].map(q => [q[0], q[1]]), hexA(COL.good, .16), pa, hexA(COL.good, .8), 2);
    }
    D.text('INPUT SPACE', 1420, 300, { size: 22, spacing: 3, a: (1 - f) * S.at(0, 800) }); D.text('temperature × pH, standardised', 1420, 334, { size: 24, font: '"IBM Plex Sans"', a: (1 - f) * S.at(0, 800) });
    D.text('HIDDEN SPACE', 1420, 300, { size: 22, spacing: 3, col: COL.grad, a: f }); D.text('axes = the three hidden neurons', 1420, 334, { size: 24, font: '"IBM Plex Sans"', a: f });
  },
});

quizScene('quiz-4', 'Chapter 4', [
  ['How many parameters does a 3–8–1 network have (one hidden layer, biases included)?', '3×8 + 8 + 8 + 1 = 41.'],
  ['Remove every activation from a 10-layer network. What is left?', 'One linear model. The product of ten matrices is one matrix, so ten layers collapse to a single weighted sum.'],
], ['Counting parameters.', '24 weights plus 8 biases in the first layer, 8 weights plus 1 bias in the second: 41.', 'No activations.', 'It collapses to a single linear map, so it can only ever draw a straight boundary, however deep it is.']);
