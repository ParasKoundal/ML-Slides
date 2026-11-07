/* ch3.js: Yes or no? Logistic regression, log-odds, cross-entropy as surprise, both losses from
 * maximum likelihood, why squared error fails for classifiers, a 2-D boundary trained live, softmax. */
'use strict';

Deck.chapter('Yes or no?', 'logistic regression, likelihood and cross-entropy');

const PL = new Plot(150, 270, 1040, 520, [0.5, 5.2], [-0.4, 1.4]);
const lacAxes = (a = 1) => {
  PL.axes({ xt: [1, 2, 3, 4, 5], yt: [0, 0.5, 1], xl: 'lactate at 48 h  (g/L)', a, yf: v => (v === 0 ? 'pass 0' : v === 1 ? 'fail 1' : '0.5') });
  D.seg(PL.x, PL.Y(0), PL.x + PL.w, PL.Y(0), '#1d2430', 1, a, [4, 6]); D.seg(PL.x, PL.Y(1), PL.x + PL.w, PL.Y(1), '#1d2430', 1, a, [4, 6]);
};
const lacPoints = (a = 1, drop = 1) => LAC.x.forEach((x, i) => {
  const q = Ease.back(clamp(drop * 1.5 - i * .02));
  D.dot(PL.X(x), lerp(PL.Y(0.5), PL.Y(LAC.y[i]), q), 9, LAC.y[i] ? COL.cls1 : COL.cls0, a * clamp(q * 3), 10);
});

chapterCard(3, 'Yes or no?', 'logistic regression, likelihood and cross-entropy',
  'Chapter three. Not every answer is a number. Sometimes the question is: will this batch pass QC or not?',
  (g, S) => { const P = new Plot(1200, 320, 540, 320, [-6, 6], [0, 1]); D.fn(P, z => sigmoid(z), -6, lerp(-6, 6, Ease.inOut(S.p(2200, 300))), COL.model, 5, 1, 160, null, 14); });

/* ------------------------------------------------------------ 1. a line gives impossible probabilities; squash it */
Deck.add({
  id: 'yes-no', title: 'From a line to an S-curve', beats: 3, dur: [3800, 4200, 4600], min: 2.5,
  notes: [
    'Thirty batches, simulated. On the x-axis, lactate measured at 48 hours. Each batch either passed QC, zero, in blue, or failed, one, in amber. High lactate tends to mean failure, but there is overlap in the middle.',
    'Try our trusty least-squares line on these zeros and ones. It is happy to predict a 137 percent chance of failure on the right and minus 33 percent on the left. Probabilities live between 0 and 1; a line does not know that.',
    'So keep the weighted sum, call it z, and squash it with the sigmoid, one over one plus e to the minus z. Any number in, a number between 0 and 1 out. The weighted sum is still the model; the sigmoid only changes how we read it.',
  ],
  html: () => `${hdr('Chapter 3', 'A line makes impossible probabilities')}
    ${at(1290, 330, `<div class="eq">${M(T`p = \sigma(\color{#3fd0ff}{w}x + \color{#3fd0ff}{b})`)}</div><div class="eq-s" style="margin-top:22px">${M(T`\sigma(z) = \frac{1}{1 + e^{-z}}`)}</div><div class="lede" style="font-size:27px;margin-top:16px">any number in, a probability out</div>`, 2, 'pop')}
    ${sub('30 batches (simulated): passed QC (blue) or failed (amber).', 0, 1)}
    ${sub('A least-squares line predicts a 137 % chance of failure.', 1, 2)}
    ${sub('Keep the weighted sum. Squash it into 0 to 1.', 2)}`,
  frame(g, S) {
    lacAxes(); lacPoints(1, S.at(0, 2400, 200));
    if (S.beat >= 1) {
      const f = Ease.inOut(S.at(2, 2400, 300)), a = S.at(1, 700);
      const fn = x => lerp(LAC.line.w * x + LAC.line.b, sigmoid(LAC.w * x + LAC.b), f);
      D.fn(PL, fn, 0.5, 5.2, COL.model, 5, a, 200, null, 14);
      const la = a * (1 - f);
      D.text(`${Math.round(100 * (LAC.line.w * 5.2 + LAC.line.b))} %?`, PL.X(5.2) - 10, PL.Y(LAC.line.w * 5.2 + LAC.line.b) - 30, { size: 34, col: COL.loss, align: 'right', weight: 600, a: la });
      D.text(`${Math.round(100 * (LAC.line.w * 0.5 + LAC.line.b))} %?`, PL.X(0.5) + 20, PL.Y(LAC.line.w * 0.5 + LAC.line.b) + 34, { size: 34, col: COL.loss, weight: 600, a: la });
    }
  },
});

/* ------------------------------------------------------------ 2. what the weight means: log-odds */
Deck.add({
  id: 'log-odds', title: 'Weights are log-odds', beats: 2, dur: [5200, 4600], min: 2.5,
  notes: [
    'What does the sigmoid do to numbers? On the left is the raw score z, any real number. On the right, the probability. Watch where evenly spaced scores land: the middle is stretched out, the ends are crushed toward 0 and 1. Going from z = 4 to z = 8 barely changes p; going from 0 to 1 changes it a lot.',
    'Run it backwards and z is the log-odds: the log of p over one minus p. That gives the weights a meaning researchers can report. In our fit w is ' + LAC.w.toFixed(2) + ' per gram per litre, so every extra gram per litre of lactate multiplies the odds of failure by e to the w, about ' + Math.exp(LAC.w).toFixed(0) + '. The batch is a coin flip, p = 0.5, exactly where z = 0: at ' + LAC.thresh.toFixed(2) + ' grams per litre.',
  ],
  html: () => `${hdr('Chapter 3', 'The score z is the log-odds')}
    ${at(1250, 300, `<div class="eq-s">${M(T`z = \ln\frac{p}{1-p} = \color{#3fd0ff}{w}\,x + \color{#3fd0ff}{b}`)}</div>`, 1, 'rise')}
    ${at(1250, 430, `<div class="lbl">fitted on the 30 batches</div><div class="big mono c-model" style="margin-top:10px">w = ${LAC.w.toFixed(2)} per g/L</div><div class="big" style="margin-top:14px">odds × <span class="mono c-data">e<sup>w</sup> = ${Math.exp(LAC.w).toFixed(0)}</span> per g/L</div><div class="big" style="margin-top:14px">coin flip at <span class="mono c-data">${LAC.thresh.toFixed(2)} g/L</span></div>`, 1, 'rise', '--d:300ms')}
    ${sub('Even steps in z: the middle stretches, the ends are crushed.', 0, 1)}
    ${sub('Odds multiply, log-odds add. That is what a weight means.', 1)}`,
  frame(g, S) {
    const zx = 260, px = 900, y0 = 860, y1 = 280, zY = z => lerp(y0, y1, (z + 8) / 16), pY = p => lerp(y0, y1, p);
    D.seg(zx, y0, zx, y1, '#3a4352', 2); D.seg(px, y0, px, y1, '#3a4352', 2);
    D.text('score z', zx, y1 - 34, { size: 24, align: 'center', col: COL.model }); D.text('probability p', px, y1 - 34, { size: 24, align: 'center', col: COL.model });
    [-8, -4, 0, 4, 8].forEach(z => D.text(String(z), zx - 20, zY(z), { size: 20, align: 'right' }));
    [0, 0.5, 1].forEach(p => D.text(String(p), px + 20, pY(p), { size: 20 }));
    const pr = S.at(0, 3800, 300);
    for (let k = -8; k <= 8; k += 1) {
      const p = sigmoid(k), f = Ease.inOut(clamp(pr * 1.6 - (k + 8) * .035));
      const pts = []; for (let i = 0; i <= 30; i++) { const t = i / 30, e = Ease.inOut(t); pts.push([lerp(zx, px, t), lerp(zY(k), pY(p), e)]); }
      D.line(pts.slice(0, Math.max(2, Math.round(31 * f))), k === 0 ? COL.data : COL.model, k === 0 ? 3 : 1.6, k === 0 ? .9 : .5);
      D.dot(zx, zY(k), 5, COL.model, .9);
      if (f > .98) D.dot(px, pY(p), 6, k === 0 ? COL.data : COL.model, 1, 8);
    }
    if (S.beat >= 1) {
      const a = S.at(1, 800);
      D.text('p = 0.5 ⟺ z = 0', (zx + px) / 2, zY(0) - 22, { size: 24, col: COL.data, align: 'center', a });
    }
  },
});

/* ------------------------------------------------------------ 3. cross-entropy as surprise */
const SUR = { bad: [1.2, -2.0], fit: [LAC.w, LAC.b] };
Deck.add({
  id: 'surprise', title: 'Cross-entropy is surprise', beats: 3, dur: [5000, 5200, 4600], min: 3,
  notes: [
    'How do we score a probability forecast? By how surprised it was by what actually happened. If the model said 99 percent pass and the batch failed, that is a huge surprise. Surprise is minus the log of the probability the model gave to the actual outcome. At p = 1 it is zero; as p goes to zero it shoots to infinity. Confident and wrong is punished without limit.',
    'Apply it to every batch: each pink bar under the plot is one batch\'s surprise. A poor model, the flat curve, is surprised often. Now let the model move toward the fitted one and watch the bars collapse. The average height is the loss: cross-entropy, also called log-loss. It is the same log-loss you met for boosting in Lecture 1.',
    'Written out: y is 1 for a failed batch and 0 for a pass, so only one of the two terms is ever switched on. It picks the log of the probability given to what actually happened.',
  ],
  html: () => `${hdr('Chapter 3', 'Score a forecast by its surprise')}
    ${at(1330, 300, '<div class="lbl">mean surprise · cross-entropy</div><div id="ce" class="num c-loss" style="margin-top:12px;font-size:110px"></div>', 1, 'rise')}
    ${at(1250, 640, `<div class="eq-xs">${M(T`L = -\frac{1}{n}\sum_i \Big[\color{#ffb547}{y_i}\ln p_i + (1-\color{#ffb547}{y_i})\ln(1-p_i)\Big]`)}</div><div class="lede" style="font-size:26px;margin-top:12px">y = 1 switches on the first term, y = 0 the second</div>`, 2, 'rise')}
    ${sub('Surprise = −ln(probability you gave to what happened).', 0, 1)}
    ${sub('One bar per batch. The average surprise is the loss.', 1, 2)}
    ${sub('Cross-entropy: only the term for what actually happened is on.', 2)}`,
  frame(g, S) {
    const P = new Plot(150, 270, 1000, 400, [0.5, 5.2], [-0.05, 1.05]);
    P.axes({ xt: [1, 2, 3, 4, 5], yt: [0, 1], xl: 'lactate  (g/L)', yf: v => (v ? 'fail' : 'pass') });
    const f = S.beat === 0 ? 0 : Ease.inOut(S.at(1, 3800, 500));
    const w = lerp(SUR.bad[0], SUR.fit[0], f), b = lerp(SUR.bad[1], SUR.fit[1], f);
    D.fn(P, x => sigmoid(w * x + b), 0.5, 5.2, COL.model, 5, 1, 160, null, 12);
    const base = 880, sc = 45; let tot = 0;
    D.seg(P.x, base, P.x + P.w, base, '#3a4352', 1.5, S.at(1, 600));
    LAC.x.forEach((x, i) => {
      const p = sigmoid(w * x + b), pc = LAC.y[i] ? p : 1 - p, s = -Math.log(clamp(pc, 1e-6, 1)); tot += s;
      D.dot(P.X(x), P.Y(LAC.y[i]), 9, LAC.y[i] ? COL.cls1 : COL.cls0, 1, 8);
      if (S.beat >= 1) D.rect(P.X(x) - 7, base - Math.min(150, s * sc) * S.at(1, 700), 14, Math.min(150, s * sc) * S.at(1, 700), COL.loss, .85);
    });
    if (S.beat >= 1) D.text('surprise per batch', P.x, base + 24, { size: 20, col: COL.loss });
    const el = document.getElementById('ce'); if (el) el.textContent = fmt(tot / LAC.n, 2);
    // the -ln p curve with a sliding forecast
    if (S.beat === 0) {
      const Q = new Plot(1300, 300, 480, 360, [0, 1], [0, 5]);
      Q.axes({ xt: [0, 0.5, 1], yt: [0, 5], xl: 'probability given to what happened', yl: 'surprise  −ln p' });
      D.fn(Q, p => -Math.log(p), 0.0067, 1, COL.loss, 4, 1, 200);
      const p = lerp(0.95, 0.012, Ease.inOut(S.p(3600, 800)));
      D.dot(Q.X(p), Q.Y(Math.min(5, -Math.log(p))), 12, COL.loss, 1, 18);
      D.text(`p = ${p.toFixed(2)}  →  ${(-Math.log(p)).toFixed(2)}`, Q.x + Q.w, Q.y - 6, { size: 26, align: 'right', col: COL.fg });
    }
  },
});

/* ------------------------------------------------------------ 4. both losses from maximum likelihood */
Deck.add({
  id: 'likelihood', title: 'Both losses come from likelihood', beats: 3, dur: [3800, 3800, 4200], min: 3,
  notes: [
    'Neither loss was invented. For a number, assume Gaussian noise around the prediction, as we did in chapter one. Minus the log likelihood is the squared error, plus a constant.',
    'For a yes or no, assume each batch is a weighted coin that fails with probability p. The Bernoulli probability is p to the y times one minus p to the one minus y. Minus its log is exactly cross-entropy.',
    'That is the recipe for every loss in this series: write down how you think the data were generated, then minimise minus the log likelihood. LLMs use the many-outcome version of the coin, over the next word. And if your noise is not Gaussian, or your outcome is a count, the recipe tells you the right loss: Poisson for counts, for example.',
  ],
  html: () => `${hdr('Chapter 3 · where losses come from', 'Pick a noise model, the loss falls out')}
    ${at(150, 545, `<div class="h3 c-model">a number</div><div class="lede" style="font-size:26px;margin:6px 0 22px">Gaussian noise around ŷ</div><div class="eq-xs">${M(T`p(y) \propto e^{-(y-\hat y)^2/2\sigma^2}`)}</div><div class="eq-xs" style="margin-top:20px">${M(T`-\ln p = \frac{(y-\hat y)^2}{2\sigma^2} + c`)}</div><div class="big c-loss" style="margin-top:20px">→ squared error</div>`, 0, 'rise')}
    ${at(1000, 545, `<div class="h3 c-data">a yes / no</div><div class="lede" style="font-size:26px;margin:6px 0 22px">a weighted coin, fails with p</div><div class="eq-xs">${M(T`p(y) = p^{\,y}(1-p)^{1-y}`)}</div><div class="eq-xs" style="margin-top:20px">${M(T`-\ln p = -\big[y\ln p + (1-y)\ln(1-p)\big]`)}</div><div class="big c-loss" style="margin-top:20px">→ cross-entropy</div>`, 1, 'rise')}
    ${sub('Assume Gaussian noise: minus log likelihood is squared error.', 0, 1)}
    ${sub('Assume a weighted coin: minus log likelihood is cross-entropy.', 1, 2)}
    ${sub('Write down how the data were made; minimise −log likelihood.', 2)}`,
  frame(g, S) {
    const a0 = S.at(0, 900), a1 = S.at(1, 900);
    const P = new Plot(150, 300, 620, 210, [-3, 3], [0, 1.1]);
    D.fn(P, x => Math.exp(-x * x / 2), -3, lerp(-3, 3, Ease.inOut(a0)), COL.model, 4, 1, 120, null, 10);
    D.seg(P.x, P.y + P.h, P.x + P.w, P.y + P.h, '#3a4352', 2, a0);
    const bw = 140, by = 488;
    [[0.3, COL.cls0, 'pass'], [0.7, COL.cls1, 'fail']].forEach(([p, c, l], k) => {
      const h = 200 * p * Ease.back(clamp(a1 * 1.4 - k * .2));
      D.rect(1060 + k * 200, by - h, bw, h, c, .9);
      D.text(`${l}  ${p}`, 1060 + k * 200 + bw / 2, by + 24, { size: 22, align: 'center', a: a1 });
    });
    if (S.beat >= 2) { const q = S.at(2, 700); D.rect(130, 528, 760, 385, null, q, COL.good, 2, 14); D.rect(980, 528, 820, 385, null, q, COL.good, 2, 14); }
  },
});

/* ------------------------------------------------------------ 5. why not squared error for a classifier */
const WHY = (() => {
  const run = (loss, steps = 300, lr = 0.6) => {
    let w = -1.2, b = 3.4; const acc = [];
    const xs = LAC.x.map(v => v - 2.8);
    for (let k = 0; k <= steps; k++) {
      let gw = 0, gb = 0, ok = 0;
      xs.forEach((x, i) => {
        const p = sigmoid(w * x + b), y = LAC.y[i];
        ok += ((p > .5 ? 1 : 0) === y) ? 1 : 0;
        const dz = loss === 'ce' ? p - y : 2 * (p - y) * p * (1 - p);
        gw += dz * x; gb += dz;
      });
      acc.push(ok / LAC.n);
      w -= lr * gw / LAC.n; b -= lr * gb / LAC.n;
    }
    return acc;
  };
  return { ce: run('ce'), mse: run('mse') };
})();
Deck.add({
  id: 'why-ce', title: 'Why not squared error for yes/no?', beats: 2, dur: [4600, 6200], min: 2.5,
  notes: [
    'Why not just use squared error on the probabilities? Look at the gradient that reaches the score z for a batch that failed, y = 1. With cross-entropy it is p minus y: when the model is confidently wrong, p near zero, the push is nearly one, full strength. With squared error through a sigmoid there is an extra factor p times one minus p, which goes to zero exactly when the model is confidently wrong. At z = minus 6 the push is about 0.005. The model is wrong and barely hears about it.',
    'Here is what that does in practice. Start both from the same confidently wrong model and run gradient descent. Cross-entropy recovers quickly. Squared error sits on a plateau for a long time before it gets going. This vanishing push through a saturated sigmoid will come back in Lecture 3 as vanishing gradients in deep networks.',
  ],
  html: () => `${hdr('Chapter 3', 'Squared error goes quiet when wrong')}
    ${at(1250, 300, `<div class="eq-xs">${M(T`\text{cross-entropy:}\ \ \frac{\partial L}{\partial z} = p - y`)}</div><div class="eq-xs" style="margin-top:22px">${M(T`\text{squared error:}\ \ \frac{\partial L}{\partial z} = 2(p-y)\,\color{#ff4d7d}{p(1-p)}`)}</div><div class="lede" style="font-size:26px;margin-top:14px">the pink factor vanishes when the model is sure</div>`, 0, 'rise')}
    ${sub('A failed batch scored z = −6: cross-entropy pushes ≈ 1.0, squared error ≈ 0.005.', 0, 1)}
    ${sub('Same start, same step size. Squared error sits on a plateau.', 1)}`,
  frame(g, S) {
    if (S.beat === 0) {
      const P = new Plot(150, 300, 960, 520, [-8, 4], [0, 1.05]);
      P.axes({ xt: [-8, -6, -4, -2, 0, 2, 4], yt: [0, 0.5, 1], xl: 'score z for a batch that failed  (y = 1)', yl: 'size of the push  |∂L/∂z|', grid: true });
      const p = S.p(2400, 200);
      D.fn(P, z => 1 - sigmoid(z), -8, lerp(-8, 4, p), COL.loss, 5, 1, 200, null, 10);
      D.fn(P, z => 2 * (1 - sigmoid(z)) * sigmoid(z) * (1 - sigmoid(z)), -8, lerp(-8, 4, p), COL.model, 5, 1, 200, null, 10);
      const a = S.p(800, 2600);
      D.text('cross-entropy', P.X(-7.8), P.Y(1) - 22, { size: 24, col: COL.loss, a });
      D.text('squared error', P.X(-1.2), P.Y(0.36), { size: 24, col: COL.model, a });
      D.seg(P.X(-6), P.y, P.X(-6), P.y + P.h, COL.fg, 1.5, a, [6, 6]);
      D.text('confidently wrong', P.X(-6) + 12, P.Y(0.6), { size: 22, col: COL.fg, a });
    } else {
      const P = new Plot(150, 300, 960, 520, [0, 300], [0.3, 1.0]);
      P.axes({ xt: [0, 100, 200, 300], yt: [0.4, 0.6, 0.8, 1], xl: 'gradient-descent step', yl: 'accuracy', yf: v => Math.round(v * 100) + '%', grid: true });
      const n = Math.floor(S.p(5400, 300) * 300);
      D.line(WHY.ce.slice(0, n + 1).map((v, k) => [P.X(k), P.Y(v)]), COL.loss, 4, 1, null, 8);
      D.line(WHY.mse.slice(0, n + 1).map((v, k) => [P.X(k), P.Y(v)]), COL.model, 4, 1, null, 8);
      if (n > 40) { D.text('cross-entropy', P.X(45), P.Y(WHY.ce[45]) - 22, { size: 24, col: COL.loss }); D.text('squared error', P.X(10), P.Y(WHY.mse[10]) - 22, { size: 24, col: COL.model }); }
    }
  },
});

/* ------------------------------------------------------------ 6. two inputs: a straight boundary, trained live */
const PR = new Plot(150, 270, 960, 580, [30, 40], [10, 60]);
Deck.add({
  id: 'two-inputs', title: 'Two inputs: a straight boundary', beats: 3, dur: [3800, 7000, 4400], min: 3,
  notes: [
    'Now two inputs. 120 simulated batches, temperature on x, dissolved oxygen on y. Amber batches had a thermal runaway, blue did not. Runaway tends to happen hot and oxygen-starved.',
    'Logistic regression with two weights, trained live with gradient descent on cross-entropy. The colour is the predicted probability of runaway; the bright line is where it equals one half. Watch the line swing into place as the loss falls.',
    'However long you train, the boundary is a straight line: w1 T plus w2 DO plus b equals zero. In d dimensions it is a flat hyperplane. For this data that is fine. Hold that thought, because the next data set will break it.',
  ],
  html: () => `${hdr('Chapter 3', 'Two inputs: the boundary is a line')}
    ${at(1250, 300, `<div class="eq-s">${M(T`p = \sigma(\color{#3fd0ff}{w_1}T + \color{#3fd0ff}{w_2}\,\mathrm{DO} + \color{#3fd0ff}{b})`)}</div>`, 1, 'rise')}
    ${at(1250, 420, '<div class="lbl">step</div><div id="rs" class="mono" style="font-size:48px;margin:6px 0 16px">0</div><div class="lbl">cross-entropy</div><div id="rl" class="mono c-loss" style="font-size:48px;margin:6px 0 16px"></div><div class="lbl">accuracy</div><div id="ra" class="mono c-good" style="font-size:48px;margin-top:6px"></div>', 1, 'rise', '--d:300ms')}
    ${at(1250, 780, `<div class="eq-xs">${M(T`\text{boundary: } w_1T + w_2\,\mathrm{DO} + b = 0`)}</div>`, 2, 'rise')}
    ${sub('120 batches (simulated): thermal runaway (amber) or not (blue).', 0, 1)}
    ${sub('Gradient descent on cross-entropy, live. Colour = predicted risk.', 1, 2)}
    ${sub('However long it trains, the boundary stays a straight line.', 2)}`,
  enter(st) { st.w = [0, 0]; st.b = 0; st.k = 0; st.img = null; st.last = -1; },
  frame(g, S) {
    const st = S.st;
    const train = () => {
      let g0 = 0, g1 = 0, gb = 0; RUN.Z.forEach(([a, c], i) => { const d = sigmoid(st.w[0] * a + st.w[1] * c + st.b) - RUN.y[i]; g0 += d * a; g1 += d * c; gb += d; });
      st.w[0] -= .5 * g0 / RUN.n; st.w[1] -= .5 * g1 / RUN.n; st.b -= .5 * gb / RUN.n; st.k++;
    };
    if (S.beat === 1) { const target = Math.floor(Math.pow(S.p(6400, 300), 1.6) * 400); while (st.k < target) train(); }
    if (S.beat === 2) while (st.k < 400) train();
    if (S.beat === 0) { st.w = [0, 0]; st.b = 0; st.k = 0; }
    const pf = (T_, DO) => sigmoid(st.w[0] * (T_ - RUN.mu[0]) / RUN.sd[0] + st.w[1] * (DO - RUN.mu[1]) / RUN.sd[1] + st.b);
    if (S.beat >= 1 && st.last !== st.k) { st.img = heatImage(PR, pf, { res: 8, lo: 0, hi: 1, stops: ['#0b2347', '#12345a', '#2a2f45', '#5a3b2c', '#8a5a1f'] }); st.last = st.k; }
    if (st.img) drawImg(st.img, PR, .95 * S.at(1, 600));
    PR.axes({ xt: [30, 32, 34, 36, 38, 40], yt: [10, 30, 50], xl: 'temperature  (°C)', yl: 'dissolved oxygen  (% sat)' });
    if (S.beat >= 1 && st.k > 0) drawSegs(contourSegs(PR, pf, .5, 60, 40), COL.data, 4, 1, 12);
    RUN.X.forEach(([T_, DO], i) => D.dot(PR.X(T_), PR.Y(DO), 8, RUN.y[i] ? COL.cls1 : COL.cls0, clamp(S.at(0, 1500) * 2 - i / RUN.n), 6));
    let L = 0, ok = 0; RUN.X.forEach(([T_, DO], i) => { const p = clamp(pf(T_, DO), 1e-9, 1 - 1e-9); L -= RUN.y[i] ? Math.log(p) : Math.log(1 - p); ok += (p > .5 ? 1 : 0) === RUN.y[i] ? 1 : 0; });
    const set = (id, v) => { const el = document.getElementById(id); if (el) el.textContent = v; };
    set('rs', String(st.k)); set('rl', fmt(L / RUN.n, 3)); set('ra', Math.round(100 * ok / RUN.n) + ' %');
  },
});

/* ------------------------------------------------------------ 7. many outcomes: softmax */
const SMX = (() => { const z = [2.0, 0.5, -1.0], e = z.map(Math.exp), s = sum(e); return { z, e, p: e.map(v => v / s), s, names: ['on spec', 'low titre', 'contaminated'], cols: [COL.good, COL.data, COL.loss] }; })();
Deck.add({
  id: 'softmax', title: 'Many outcomes: softmax', beats: 3, dur: [3600, 3600, 4200], min: 2,
  notes: [
    'More than two outcomes? Say a batch ends on spec, low titre, or contaminated. Give each class its own weighted sum, its own score z. Scores can be any real numbers, negative included.',
    'Exponentiate them: now everything is positive, and bigger scores are much bigger.',
    'Divide by the total and they sum to one: probabilities. That is softmax, and the loss is again cross-entropy: minus the log of the probability given to the class that happened. With two classes it reduces exactly to the sigmoid. An LLM does exactly this at every step, over a vocabulary of around a hundred thousand possible next words. Lecture 5.',
  ],
  html: () => `${hdr('Chapter 3', 'Many outcomes: softmax')}
    ${at(1250, 330, `<div class="eq">${M(T`p_k = \frac{e^{z_k}}{\sum_j e^{z_j}}`)}</div><div class="lede" style="font-size:27px;margin-top:20px">two classes: exactly the sigmoid<br>an LLM: ~100,000 classes, one per word</div>`, 2, 'rise')}
    ${sub('One score per outcome. Any real number.', 0, 1)}
    ${sub('Exponentiate: everything positive, gaps magnified.', 1, 2)}
    ${sub('Normalise: probabilities that sum to one.', 2)}`,
  frame(g, S) {
    const base = 640, scale = [70, 40, 330], x0 = 230, bw = 170, gap = 120;
    const f1 = Ease.inOut(S.at(1, 1600, 200)), f2 = Ease.inOut(S.at(2, 1600, 200));
    D.seg(180, base, 1080, base, '#3a4352', 2);
    SMX.z.forEach((z, k) => {
      const v0 = z * scale[0], v1 = SMX.e[k] * scale[1], v2 = SMX.p[k] * scale[2];
      const h = lerp(lerp(v0, v1, f1), v2, f2) * Ease.back(S.at(0, 900, k * 150));
      const x = x0 + k * (bw + gap);
      D.rect(x, h >= 0 ? base - h : base, bw, Math.abs(h), SMX.cols[k], .9, null, 0, 6);
      const label = f2 > .5 ? SMX.p[k].toFixed(2) : f1 > .5 ? SMX.e[k].toFixed(2) : (z > 0 ? '+' : '') + z.toFixed(1);
      D.text(label, x + bw / 2, h >= 0 ? base - h - 26 : base - h + 26, { size: 30, align: 'center', col: COL.fg, weight: 600 });
      D.text(SMX.names[k], x + bw / 2, (z < 0 && f1 < .5) ? base - 24 : base + 34, { size: 24, align: 'center', font: '"IBM Plex Sans"' });
    });
    const stage = f2 > .5 ? 'probability p' : f1 > .5 ? 'e^z' : 'score z';
    D.text(stage, 180, 300, { size: 26, col: COL.model });
  },
});

quizScene('quiz-3', 'Chapter 3', [
  ['A model gives p = 0.2 to a batch that then fails. What is its cross-entropy for that batch? (ln 0.2 ≈ −1.61)', '1.61. Had it said p = 0.9, the loss would be 0.11. Confident and right is cheap.'],
  ['w = 2 per g/L of lactate. How do the odds of failure change for +1 g/L? For +2 g/L?', '× e² ≈ 7.4 per g/L, and × e⁴ ≈ 55 for +2 g/L. Log-odds add, so odds multiply.'],
], ['Cross-entropy by hand.', 'Minus ln 0.2 is 1.61. If it had said 0.9 the loss would be minus ln 0.9, about 0.11.', 'Interpreting a weight.', 'e squared is about 7.4; two grams per litre gives e to the fourth, about 55. This is how epidemiologists report odds ratios, and it works for your process data too.']);
