/* x5.js: extended cut, chapter 5. Forward versus reverse mode on the Arrhenius rate; exploding gradients and
 * clipping; the matrix shapes of backprop (dW = X^T delta as a sum of outer products). */
'use strict';

/* ------------------------------------------------------------ forward mode vs reverse mode */
const AD = (() => {
  const R = 8.314, A = 1e7, Ea = 55000, T = 330;
  const u = -Ea / (R * T), e = Math.exp(u), k = A * e;
  const fw = { T: 1, u: Ea / (R * T * T), e: 0, k: 0 }; fw.e = e * fw.u; fw.k = A * fw.e;           // tangents, seeded on T
  const rv = { k: 1, e: A, A: e, u: 0, Ea: 0, T: 0 }; rv.u = rv.e * e; rv.Ea = rv.u * (-1 / (R * T)); rv.T = rv.u * Ea / (R * T * T);   // adjoints
  return { R, A, Ea, T, u, e, k, fw, rv };
})();
const ADN = { A: [230, 320], Ea: [230, 550], T: [230, 780], u: [640, 645], e: [1010, 645], k: [1380, 440] };
const ADE = [['A', 'k'], ['Ea', 'u'], ['T', 'u'], ['u', 'e'], ['e', 'k']];
const adLbl = (n, txt, dy, b, cls, d = 0, xb) => at(ADN[n][0] - 150, ADN[n][1] + dy, `<div class="mono ${cls}" style="width:300px;text-align:center;font-size:26px">${txt}</div>`, b, 'rise', `--d:${d}ms`, xb);
Deck.insert('grad-check', {
  id: 'modes', title: 'Forward mode and reverse mode', beats: 3, dur: [4400, 5600, 6400], min: 4,
  notes: [
    'The Arrhenius rate constant as a computational graph. Three inputs: the pre-exponential factor A, the activation energy Ea, and the temperature T. Then u equals minus Ea over R T, e equals exp of u, and k equals A times e. At 330 kelvin with Ea of 55 kilojoules per mole, k is about 0.02 per second.',
    'There are two ways to push derivatives through a graph. Forward mode carries a tangent along with each value: seed the input you care about, here T with a one, and apply each node\'s local slope as you go forward. At the end you have dk/dT, about ' + sci(AD.fw.k, 1) + ' per kelvin. But only dk/dT. For dk/dA and dk/dEa you need two more forward passes: one pass per input.',
    'Reverse mode, backpropagation, starts at the output with a one and pushes adjoints backward, each node multiplying by its local slope, exactly as in our hand example. One backward pass delivers all three: dk/dA, dk/dEa and dk/dT together. The rule of thumb: forward mode costs one pass per input, reverse mode one pass per output. Training has one output, the loss, and millions of inputs, the knobs, so it is always reverse mode. Forward mode wins when there are few inputs and many outputs, like the sensitivity of a whole simulated concentration profile to one parameter. JAX gives you both, as jvp and vjp.',
  ],
  html: () => `${hdr('Chapter 5 · automatic differentiation', 'Forward mode or reverse mode?')}
    ${[['A', 'A'], ['Ea', 'Eₐ'], ['T', 'T'], ['u', 'u = −Eₐ/RT'], ['e', 'e = exp(u)'], ['k', 'k = A·e']].map(([n, t]) => adLbl(n, t, -92, 0, 'c-dim')).join('')}
    ${[['A', sci(AD.A, 0)], ['Ea', '55 000'], ['T', '330'], ['u', AD.u.toFixed(2).replace('-', '−')], ['e', sci(AD.e, 2)], ['k', AD.k.toFixed(4)]].map(([n, t], i) => adLbl(n, t, 52, 0, 'c-data', 300 + i * 150)).join('')}
    ${[['T', '1'], ['u', sci(AD.fw.u, 2)], ['e', sci(AD.fw.e, 2)], ['k', sci(AD.fw.k, 2)]].map(([n, t], i) => adLbl(n, 'ṫ ' + t, 90, 1, 'c-good', 600 + i * 1100, 2)).join('')}
    ${[['k', '1'], ['e', sci(AD.rv.e, 0)], ['u', AD.rv.u.toFixed(4)], ['A', sci(AD.rv.A, 2)], ['Ea', sci(AD.rv.Ea, 2)], ['T', sci(AD.rv.T, 2)]].map(([n, t], i) => adLbl(n, 'ā ' + t, 90, 2, 'c-grad', 500 + [0, 1, 2, 3, 3, 3][i] * 1000)).join('')}
    ${at(1250, 700, '<div class="card" style="width:560px"><h4 class="c-good">forward mode</h4><p>one pass per input: here, dk/dT only</p></div>', 1, 'rise', '--d:4800ms', 2)}
    ${at(1250, 700, '<div class="card" style="width:560px"><h4 class="c-grad">reverse mode</h4><p>one pass per output: all three slopes at once. Training is always reverse mode.</p></div>', 2, 'rise', '--d:4200ms')}
    ${sub('The Arrhenius rate as a graph: values flow forward.', 0, 1)}
    ${sub('Forward mode: seed T, carry a tangent forward. <b>One input per pass.</b>', 1, 2)}
    ${sub('Reverse mode: seed k, push adjoints back. <b>Every input in one pass.</b>', 2)}`,
  frame(g, S) {
    const a0 = S.at(0, 800);
    ADE.forEach(([p, q]) => { const [x1, y1] = ADN[p], [x2, y2] = ADN[q], L = Math.hypot(x2 - x1, y2 - y1), ux = (x2 - x1) / L, uy = (y2 - y1) / L; D.arrow(x1 + ux * 44, y1 + uy * 44, x2 - ux * 46, y2 - uy * 46, '#3a4352', 3, 16, a0); });
    Object.entries(ADN).forEach(([n, [x, y]]) => { const inp = ['A', 'Ea', 'T'].includes(n); D.dot(x, y, 40, '#0f141c', a0); D.ring(x, y, 40, n === 'k' ? COL.loss : inp ? COL.model : '#c8d1de', 3, a0); });
    const pulse = (order, col, t0, per, b) => order.forEach((edge, i) => {
      const q = seg(S.bt, t0 + i * per, t0 + (i + 1) * per); if (S.beat !== b || q <= 0 || q >= 1) return;
      const [p1, p2] = edge, [x1, y1] = ADN[p1], [x2, y2] = ADN[p2]; D.dot(lerp(x1, x2, q), lerp(y1, y2, q), 9, col, 1, 16);
    });
    pulse([['T', 'u'], ['u', 'e'], ['e', 'k']], COL.good, 800, 1100, 1);
    if (S.beat === 2) { [[['k', 'e'], ['k', 'A']], [['e', 'u']], [['u', 'Ea'], ['u', 'T']]].forEach((grp, i) => grp.forEach(([p1, p2]) => { const q = seg(S.bt, 0 + i * 1000, (i + 1) * 1000); if (q <= 0 || q >= 1) return; const [x1, y1] = ADN[p1], [x2, y2] = ADN[p2]; D.dot(lerp(x1, x2, q), lerp(y1, y2, q), 9, COL.grad, 1, 16); })); }
    if (S.beat === 1) { const [x, y] = ADN.T; D.ring(x, y, 52, COL.good, 3, S.at(1, 500, 200)); }
    if (S.beat === 2) { const [x, y] = ADN.k; D.ring(x, y, 52, COL.grad, 3, S.at(2, 500)); }
  },
});

/* ------------------------------------------------------------ exploding gradients and clipping */
const CLIFF = (() => {
  const L = w => 0.12 * (w - 0.6) ** 2 + 3 * sigmoid(20 * (w - 2.4));
  const dL = w => { const h = 1e-5; return (L(w + h) - L(w - h)) / (2 * h); };
  const run = clip => { let w = 4.6; const p = [w]; for (let s = 0; s < 18; s++) { let gr = dL(w); if (clip) gr = Math.sign(gr) * Math.min(Math.abs(gr), 1); w -= 0.5 * gr; p.push(w); } return p; };
  return { L, dL, raw: run(false), clip: run(true) };
})();
Deck.insert('numpy', {
  id: 'exploding', title: 'Exploding gradients and clipping', beats: 3, dur: [4800, 5200, 5600], min: 3.5,
  notes: [
    'The vanishing gradient has an evil twin. The gradient reaching layer one is a product of one factor per layer above it. If those factors are typically 0.8, thirty layers multiply to about a thousandth: vanishing. If they are typically 1.25, thirty layers multiply to about eight hundred: exploding. Only factors near one keep the signal alive, which is what careful initialisation, normalisation and residual connections are designed to do. Recurrent networks, which apply the same layer at every time step of a long sequence, suffer this worst.',
    'What an explosion looks like on the loss surface: a cliff. Here the loss is gentle almost everywhere, with one steep wall. The ball walks down the plateau, reaches the edge, and the gradient there is huge. One ordinary step multiplies that huge slope by the learning rate, and the ball is flung far off to the left, undoing many steps of progress. In a real training log this is a sudden spike in the loss, sometimes followed by NaN.',
    'The fix is almost embarrassingly simple: gradient clipping. If the gradient\'s length exceeds a threshold c, scale it down to length c, keeping its direction. Now the step at the cliff is no bigger than any other, and the ball steps down the wall and continues. In PyTorch it is one line before optimizer.step: torch.nn.utils.clip_grad_norm_. Almost every large model is trained with it, typically with c equal to one.',
  ],
  html: () => `${hdr('Chapter 5 · exploding gradients', 'Gains above one explode; clip them')}
    ${at(1200, 300, `<div class="eq-s">${M(T`\big\|\tfrac{\partial L}{\partial \mathbf{h}_1}\big\| \sim \textstyle\prod_{l} g_l`)}</div>`, 0, 'rise')}
    ${at(1200, 640, `<div class="lbl c-good">gradient clipping</div><div class="eq-s" style="margin-top:12px">${M(T`\mathbf g \leftarrow \mathbf g\cdot \min\!\Big(1, \frac{c}{\lVert \mathbf g\rVert}\Big)`)}</div><div class="lede mono" style="font-size:24px;margin-top:12px">clip_grad_norm_(model.parameters(), 1.0)</div>`, 2, 'rise')}
    ${sub('Thirty layers: 0.8 per layer vanishes, 1.25 per layer explodes.', 0, 1)}
    ${sub('On the surface: a cliff. <b>One huge gradient, one wild step.</b>', 1, 2)}
    ${sub('Clip the gradient\'s length: the step at the cliff stays sane.', 2)}`,
  frame(g, S) {
    if (S.beat === 0) {
      const P = new Plot(150, 300, 900, 500, [0, 30], [-4, 4]), a0 = S.at(0, 700);
      P.axes({ xt: [0, 10, 20, 30], yt: [-3, 0, 3], yf: v => (v === 0 ? '1' : '10' + (v < 0 ? '⁻³' : '³')), xl: 'layers the gradient passes through', yl: 'gradient size (log scale)', a: a0, grid: true });
      const q = S.at(0, 3000, 600) * 30;
      [[0.8, COL.cls0], [1.0, COL.good], [1.25, COL.loss]].forEach(([gn, col]) => {
        D.fn(P, n => n * Math.log10(gn), 0, q, col, 4, a0, 120, null, 8);
        D.text('× ' + gn.toFixed(2).replace(/0$/, ''), P.X(Math.min(q, 30)) + 14, P.Y(Math.min(q, 30) * Math.log10(gn)), { size: 24, weight: 600, col, a: a0 * seg(q, 4, 8) });
      });
      return;
    }
    const P = new Plot(150, 300, 960, 500, [-3, 5], [-0.2, 4.2]), a0 = S.at(1, 600);
    P.axes({ xt: [-2, 0, 2, 4], yt: [0, 2, 4], xl: 'knob  w', yl: 'loss', a: a0, grid: true });
    D.fn(P, CLIFF.L, -3, 5, '#c8d1de', 3, a0, 400);
    const path = S.beat === 1 ? CLIFF.raw : CLIFF.clip, col = S.beat === 1 ? COL.loss : COL.good;
    const q = S.p(4400, 400) * (path.length - 1), k = Math.min(path.length - 2, Math.floor(q)), f = Ease.inOut(q - k);
    for (let j = 0; j < k; j++) D.dot(P.X(clamp(path[j], -3.2, 5)), P.Y(CLIFF.L(path[j])), 5, col, .6);
    const w = lerp(path[k], path[k + 1], f), hop = Math.sin(f * Math.PI) * Math.min(160, Math.abs(path[k + 1] - path[k]) * 40);
    const x = P.X(w), y = P.Y(CLIFF.L(w)) - hop;
    if (w < -3) { D.text('← flung out', P.x + 10, P.y + 40, { size: 28, weight: 600, col: COL.loss }); return; }
    D.dot(x, y - 12, 13, col, 1, 22); D.dot(x - 4, y - 16, 4, '#fff', .9);
  },
});

/* ------------------------------------------------------------ the shapes of backprop */
const SHP = (() => {
  const r = RNG(29), mat = (m, n) => Array.from({ length: m }, () => Array.from({ length: n }, () => r.n()));
  const X = mat(4, 3), W = mat(3, 5), dl = mat(4, 5);
  const dW = Array.from({ length: 3 }, (_, i) => Array.from({ length: 5 }, (_, j) => sum([0, 1, 2, 3].map(b => X[b][i] * dl[b][j]))));
  return { X, W, dl, dW };
})();
const CELL = 46;
function drawMat(M_, x, y, col, a = 1, opts = {}) {
  M_.forEach((row, i) => row.forEach((v, j) => {
    const hl = opts.hl ? opts.hl(i, j) : 1, s = clamp(Math.abs(v) / 2.2);
    D.rect(x + j * CELL + 3, y + i * CELL + 3, CELL - 6, CELL - 6, v >= 0 ? col : COL.loss, a * (0.18 + 0.7 * s) * hl, null, 0, 6);
  }));
  D.rect(x - 4, y - 4, M_[0].length * CELL + 8, M_.length * CELL + 8, null, a * .5, '#3a4352', 1.5, 8);
}
const T_ = M_ => M_[0].map((_, j) => M_.map(r => r[j]));
Deck.insert('numpy', {
  id: 'shapes', title: 'The shapes of backprop', beats: 3, dur: [4400, 7200, 5200], min: 3.5,
  notes: [
    'In code, a layer works on a whole mini-batch at once. Four samples, three inputs each, is a four by three matrix X. The layer\'s weights are three by five, five neurons. One matrix product gives the pre-activations Z, four by five: one row per sample, one column per neuron. Shapes chain like dominoes: the inner dimensions must match.',
    'Backward, the gradient of the loss with respect to Z, call it delta, has the shape of Z: four by five. The gradient for the weights is X transpose times delta. Look at what that means, sample by sample: each sample contributes the outer product of its input column with its delta row, a full three by five block, and the four contributions are added up. Every weight\'s gradient is the sum over the batch of input times blame, the same input times upstream gradient as in the hand example, done for all weights at once.',
    'And the gradient passed on to the layer below is delta times W transpose: four by three, the shape of X. That gives the one rule that catches most bugs in a hand-written backward pass: a gradient always has exactly the shape of the thing it is the gradient of. If your dW is not the shape of W, something is transposed.',
  ],
  html: () => `${hdr('Chapter 5 · matrices', 'Backprop in matrix shapes')}
    ${at(150, 590, `<div class="eq-s">${M(T`Z = X\,W`)}</div>`, 0, 'rise', '', 1)}
    ${at(150, 590, `<div class="eq-s">${M(T`\frac{\partial L}{\partial W} = X^{\top}\delta = \textstyle\sum_b \mathbf{x}_b\,\boldsymbol\delta_b^{\top}`)}</div>`, 1, 'rise', '', 2)}
    ${at(150, 590, `<div class="eq-s">${M(T`\frac{\partial L}{\partial X} = \delta\, W^{\top}`)}</div>`, 2, 'rise')}
    ${[[0, [['X', '4 × 3', 200], ['W', '3 × 5', 470], ['Z', '4 × 5', 870]]], [1, [['Xᵀ', '3 × 4', 200], ['δ', '4 × 5', 470], ['∂L/∂W', '3 × 5', 870]]], [2, [['δ', '4 × 5', 200], ['Wᵀ', '5 × 3', 570], ['∂L/∂X', '4 × 3', 920]]]]
      .map(([b, ms]) => ms.map(([n, sh, x]) => at(x, 270, `<div class="cap">${n} <span class="mono c-dim" style="font-size:24px">${sh}</span></div>`, b, 'rise', '', b < 2 ? b + 1 : undefined)).join('')).join('')}
    ${at(1250, 300, '<div class="lbl">batch of 4 samples · 3 inputs · 5 neurons</div>', 0, 'rise')}
    ${at(1250, 640, '<div class="card" style="width:580px"><h4 class="c-good">the shape rule</h4><p>a gradient has exactly the shape of the thing it is the gradient of</p></div>', 2, 'rise', '--d:1200ms')}
    ${sub('Forward: (4 × 3)(3 × 5) = (4 × 5). Inner sizes must match.', 0, 1)}
    ${sub('dW = Xᵀδ: <b>one outer product per sample, summed over the batch.</b>', 1, 2)}
    ${sub('Pass back δWᵀ, the shape of X. Check every shape.', 2)}`,
  frame(g, S) {
    const a = S.at(S.beat, 600), y0 = 330;
    if (S.beat === 0) {
      drawMat(SHP.X, 200, y0, COL.data, a); D.text('×', 410, y0 + 70, { size: 40, align: 'center', a });
      drawMat(SHP.W, 470, y0, COL.model, a); D.text('=', 780, y0 + 70, { size: 40, align: 'center', a });
      const Z = SHP.X.map(r => [0, 1, 2, 3, 4].map(j => sum(r.map((v, k) => v * SHP.W[k][j])) / 1.6));
      const q = S.at(0, 2400, 900) * 20;
      drawMat(Z, 870, y0, COL.grad, a, { hl: (i, j) => clamp(q - (i * 5 + j)) });
    } else if (S.beat === 1) {
      const q = S.at(1, 6000, 600) * 4, b = Math.min(3, Math.floor(q));
      drawMat(T_(SHP.X), 200, y0, COL.data, a, { hl: (i, j) => (j === b ? 1 : .25) });
      D.text('×', 410 + CELL, y0 + 70, { size: 40, align: 'center', a });
      drawMat(SHP.dl, 470 + CELL, y0, COL.grad, a, { hl: i => (i === b ? 1 : .25) });
      D.text('=', 780 + CELL, y0 + 70, { size: 40, align: 'center', a });
      const part = SHP.dW.map((row, i) => row.map((_, j) => sum([0, 1, 2, 3].slice(0, b + 1).map(k => SHP.X[k][i] * SHP.dl[k][j])) * (b < 3 ? 1 : 1)));
      drawMat(part, 870 + CELL, y0, COL.good, a);
      D.text(`sample ${b + 1} of 4 added`, 870 + CELL, y0 + 3 * CELL + 40, { size: 22, col: COL.good, a });
      // the outer product of this sample, drawn small to the right
      const op = [0, 1, 2].map(i => [0, 1, 2, 3, 4].map(j => SHP.X[b][i] * SHP.dl[b][j]));
      const fx = seg(q - b, .1, .6);
      op.forEach((row, i) => row.forEach((v, j) => D.rect(1180 + 80 + j * 22, y0 + 260 + i * 22, 18, 18, v >= 0 ? COL.good : COL.loss, (1 - seg(q - b, .6, .95)) * fx * clamp(Math.abs(v) / 2), null, 0, 3)));
      D.text('x_b δ_bᵀ', 1180 + 80, y0 + 240, { size: 20, col: COL.dim, a: fx * (1 - seg(q - b, .6, .95)) });
    } else {
      drawMat(SHP.dl, 200, y0, COL.grad, a); D.text('×', 470 + 40, y0 + 70, { size: 40, align: 'center', a });
      drawMat(T_(SHP.W), 570, y0 - 20, COL.model, a); D.text('=', 830 + 30, y0 + 70, { size: 40, align: 'center', a });
      const dX = SHP.dl.map(r => [0, 1, 2].map(k => sum(r.map((v, j) => v * SHP.W[k][j])) / 2));
      drawMat(dX, 920, y0, COL.data, a, { hl: (i, j) => clamp(S.at(2, 2000, 600) * 12 - (i * 3 + j)) });
    }
  },
});
