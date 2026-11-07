/* ch8.js: Wrap-up. Three questions answered; the training loop as one picture; exercises; reading; next lecture. */
'use strict';

Deck.chapter('Wrap-up', 'what to take away, and what to try');

/* ------------------------------------------------------------ 1. three questions, three answers */
Deck.add({
  id: 'recap', title: 'Three questions, three answers', beats: 3, dur: [4200, 4200, 4600], min: 3,
  notes: [
    'Back to the three questions from the start. Model: what shape of function? Today we went from a straight line, to a sigmoid for yes-or-no answers, to networks of bent neurons that can draw any boundary, an operating window included.',
    'Loss: how do we score it? Write down how the data were generated and take minus the log likelihood. Gaussian noise gives squared error; a weighted coin gives cross-entropy; many outcomes give softmax cross-entropy, which is exactly the loss used to train an LLM.',
    'Learning: how do we improve it? Gradient descent: feel the slope, step downhill, repeat. Backpropagation gives every slope in one backward pass, whatever the number of knobs. Momentum and Adam make the steps smarter; validation data tells you when to stop and whether to trust the result.',
  ],
  html: () => `${hdr('Wrap-up', 'Three questions, three answers')}
    ${[['Model', 'what shape of function?', 'line → sigmoid → network of bent neurons', COL.model],
       ['Loss', 'how do we score it?', 'minus log-likelihood: squared error, cross-entropy', COL.loss],
       ['Learning', 'how do we improve it?', 'gradient descent + backprop, checked on held-out data', COL.grad]]
      .map(([t, q, a, c], k) => at(150 + k * 560, 600, `<div class="h3" style="color:${c}">${t}</div><div class="lbl" style="margin-top:12px">${q}</div><div class="lede" style="margin-top:16px;width:480px">${a}</div>`, k, 'rise')).join('')}
    ${sub('Model: from a line to a network that can bend.', 0, 1)}
    ${sub('Loss: write down the noise, minimise −log likelihood.', 1, 2)}
    ${sub('Learning: slope, step, repeat. <b>Backprop gives every slope at once.</b>', 2)}`,
  frame(g, S) {
    const cx = k => 150 + k * 560, top = 280, w = 440, h = 260;
    // model glyph: a line morphing to a sigmoid to a bumpy network fit
    if (S.beat >= 0) {
      const P = new Plot(cx(0), top, w, h, [-3, 3], [-0.2, 1.2]), a = S.at(0, 600), m = (S.t / 2600) % 3, k = Math.floor(m), f = Ease.inOut(clamp((m - k) * 1.6));
      const fs = [x => .5 + .15 * x, x => sigmoid(2.2 * x), x => .5 + .4 * Math.sin(1.6 * x) * Math.exp(-x * x / 6) + .1 * x];
      D.fn(P, x => lerp(fs[k](x), fs[(k + 1) % 3](x), f), -3, 3, COL.model, 4, a, 160, null, 12);
      D.seg(P.x, P.y + P.h, P.x + P.w, P.y + P.h, '#2a3240', 1.5, a);
    }
    if (S.beat >= 1) {
      const P = new Plot(cx(1), top, w, h, [-3, 3], [0, 4.5]), a = S.at(1, 600);
      D.fn(P, x => x * x / 2, -3, 3, COL.loss, 4, a, 160, null, 10);
      D.fn(P, x => Math.min(4.4, Math.log(1 + Math.exp(-2 * x)) * 1.4), -3, 3, COL.data, 3, a, 160, [8, 6]);
      D.seg(P.x, P.y + P.h, P.x + P.w, P.y + P.h, '#2a3240', 1.5, a);
    }
    if (S.beat >= 2) {
      const P = new Plot(cx(2), top, w, h, [-3, 3], [0, 4.5]), a = S.at(2, 600);
      D.fn(P, x => x * x / 2, -3, 3, '#3a4352', 3, a, 160);
      const q = (S.at(2, 600, 300) * 1.4 + S.bt / 6000) % 1, xs = [-2.8]; for (let k = 0; k < 12; k++) xs.push(xs[k] * .62);
      const n = Math.min(12, Math.floor(q * 13));
      for (let k = 0; k < n; k++) D.arrow(P.X(xs[k]), P.Y(xs[k] ** 2 / 2), P.X(xs[k + 1]), P.Y(xs[k + 1] ** 2 / 2), COL.grad, 2.5, 12, a);
      D.dot(P.X(xs[n]), P.Y(xs[n] ** 2 / 2), 10, COL.model, a, 16);
    }
  },
});

/* ------------------------------------------------------------ 2. the loop */
Deck.add({
  id: 'loop', title: 'The loop that trains everything', beats: 2, dur: [6400, 4600], min: 2,
  notes: [
    'If you remember one picture from today, make it this one. Take a batch of data. Run it forward through the model. Score it with the loss. Run backward to get the gradient of the loss for every knob. Take a small step downhill. Repeat.',
    'This loop trained the straight line through the feed data, the logistic model for lactate, the 65-knob network for the operating window, and, run for about a million steps on trillions of words with hundreds of billions of knobs, the large language models in Lecture 5. The pieces change; the loop does not.',
  ],
  html: () => `${hdr('Wrap-up', 'The loop that trains everything')}
    ${at(1300, 420, '<div class="lbl">same loop for</div><div class="lede" style="margin-top:14px;width:660px;line-height:1.6">a straight line · 2 knobs<br>the window network · 65 knobs<br>GPT-3 · 175,000,000,000 knobs</div>', 1, 'rise')}
    ${sub('Batch → forward → loss → backward → step. Repeat.', 0, 1)}
    ${sub('<b>The pieces change; the loop does not.</b>', 1)}`,
  frame(g, S) {
    const cx = 680, cy = 630, R = 240;
    const st = [['batch', COL.data], ['forward', COL.model], ['loss', COL.loss], ['backward', COL.grad], ['step', COL.good]];
    const ang = k => -Math.PI / 2 + k * TAU / 5, a0 = S.at(0, 1000);
    D.g.save(); D.g.globalAlpha *= a0; D.g.strokeStyle = '#2a3240'; D.g.lineWidth = 3; D.g.beginPath(); D.g.arc(cx, cy, R, 0, TAU); D.g.stroke(); D.g.restore();
    const ph = (S.t / 5200) % 1, pa = -Math.PI / 2 + ph * TAU;
    for (let k = 0; k < 24; k++) { const b = pa - k * .035; D.dot(cx + R * Math.cos(b), cy + R * Math.sin(b), 10 - k * .35, st[Math.floor(((b + Math.PI / 2 + 10 * TAU) % TAU) / (TAU / 5))][1], (1 - k / 24) * a0, k ? 0 : 20); }
    st.forEach(([t, c], k) => {
      const x = cx + R * Math.cos(ang(k)), y = cy + R * Math.sin(ang(k)), lit = Math.floor(ph * 5) === k;
      const ak = clamp(S.at(0, 800, 200 + k * 250) * 1);
      D.dot(x, y, 46, '#0f141c', ak); D.ring(x, y, 46, c, lit ? 5 : 3, ak);
      const lx = cx + (R + 120) * Math.cos(ang(k)), ly = cy + (R + 100) * Math.sin(ang(k));
      D.text(t, lx, ly, { size: 34, align: 'center', col: c, font: '"IBM Plex Sans"', weight: 600, a: ak });
    });
    const it = Math.floor(S.t / 5200) + 1;
    D.text('step ' + it.toLocaleString('en-US'), cx, cy - 10, { size: 40, align: 'center', col: COL.fg, a: a0 });
    D.text('repeat', cx, cy + 40, { size: 24, align: 'center', a: a0 });
  },
});

/* ------------------------------------------------------------ 3. hands-on */
const EXERCISES = [
  ['01', 'Backprop from scratch', 'Run the numpy 2-16-1 trainer on the window data. Try η = 0.01, 0.1, 1, 10 and plot the four loss curves. Which ones match the loss-curve doctor?', 'lecture2_exercises.py · ex1'],
  ['02', 'Check your gradients', 'Add a central-difference check to the trainer. Then break the ReLU mask on purpose and watch the relative error jump.', 'ex2'],
  ['03', 'Conditioning, measured', 'Fit Arrhenius ln k = ln A − Ea/RT by gradient descent with raw 1/T, then standardised. Count the steps to converge.', 'ex3'],
  ['04', 'Network vs physics', 'Fit the Haldane data with a 1-16-1 network and with the 3-knob Haldane form. Compare predictions at S = 40 g/L.', 'ex4'],
  ['05', 'Rate constants by gradient descent', 'Fit k₁, k₂ of A → B → C with Adam, on ln k and on k. Then start with the order swapped.', 'ex5'],
  ['06', 'Split by batch', 'Compare random 5-fold CV with leave-one-batch-out on simulated fermentation batches, then test on 30 new batches.', 'ex6'],
];
Deck.add({
  id: 'exercises', title: 'Hands-on', beats: 2, dur: [3200, 3200], min: 2,
  notes: [
    'Six exercises, all in the code folder that comes with this deck. The first two are about backpropagation itself: train the numpy network with four learning rates and diagnose the curves, then write a gradient check and use it to catch a bug you plant on purpose. The third measures how badly conditioning slows an Arrhenius fit.',
    'The other three are chemical engineering: see for yourself that a network extrapolates nonsense where a three-parameter rate law does not; fit the rate constants of a series reaction with Adam, in log space and in linear space; and find out how optimistic a random cross-validation split is when your runs come in batches. Each takes about twenty minutes. Do at least the first one before Lecture 3.',
  ],
  html: () => `${hdr('Wrap-up', 'Try it yourself')}
    ${EXERCISES.map(([n, t, d, f], k) => at(110 + (k % 3) * 580, 260 + Math.floor(k / 3) * 350, `<div class="card" style="width:500px;min-height:280px"><div class="mono c-loss" style="font:600 40px/1 var(--mono)">${n}</div><h4 style="margin-top:12px">${t}</h4><p>${d}</p></div>`, k < 3 ? 0 : 1, 'pop', `--d:${(k % 3) * 200}ms`)).join('')}`,
});

/* ------------------------------------------------------------ 4. reading */
Deck.add({
  id: 'reading', title: 'Further reading', beats: 1, dur: 3600, min: 1.5,
  notes: 'For going deeper. Goodfellow, Bengio and Courville, chapters 4 to 6, for the mathematics of everything today. Michael Nielsen\'s free online book for backpropagation explained slowly. The 3Blue1Brown neural network videos for intuition. Andrej Karpathy\'s micrograd, a hundred lines of Python that implement backprop: read it after the exercises. Rumelhart, Hinton and Williams 1986, the paper that made backprop famous. For chapter 7: Raissi, Perdikaris and Karniadakis on physics-informed networks, and Chen and colleagues on neural ODEs. For uncertainty, Lakshminarayanan and colleagues on deep ensembles. And Venkatasubramanian 2019 in the AIChE Journal for a chemical engineer\'s view of where AI fits in our field.',
  html: () => `${hdr('Wrap-up', 'Going deeper')}
    ${at(150, 270, `<table class="table" style="width:1620px;font-size:24px">
      <tr><th>source</th><th>for</th></tr>
      <tr><td>Goodfellow, Bengio &amp; Courville, <i>Deep Learning</i>, ch. 4–6</td><td class="c-dim">the mathematics of today</td></tr>
      <tr><td>Nielsen, <i>Neural Networks and Deep Learning</i>, ch. 2</td><td class="c-dim">backprop, slowly and clearly</td></tr>
      <tr><td>3Blue1Brown, <i>Neural networks</i> video series</td><td class="c-dim">visual intuition</td></tr>
      <tr><td>Karpathy, <i>micrograd</i> (GitHub)</td><td class="c-dim">autograd in ~100 lines of Python</td></tr>
      <tr><td>Rumelhart, Hinton &amp; Williams, <i>Nature</i> 323, 533 (1986)</td><td class="c-dim">the backprop paper</td></tr>
      <tr><td>Kingma &amp; Ba, <i>Adam</i>, ICLR (2015)</td><td class="c-dim">the default optimizer</td></tr>
      <tr><td>Raissi, Perdikaris &amp; Karniadakis, <i>J. Comput. Phys.</i> 378, 686 (2019)</td><td class="c-dim">physics-informed networks</td></tr>
      <tr><td>Chen, Rubanova, Bettencourt &amp; Duvenaud, <i>Neural ODEs</i>, NeurIPS (2018)</td><td class="c-dim">networks inside differential equations</td></tr>
      <tr><td>Lakshminarayanan, Pritzel &amp; Blundell, <i>Deep Ensembles</i>, NeurIPS (2017)</td><td class="c-dim">uncertainty from disagreement</td></tr>
      <tr><td>Venkatasubramanian, <i>AIChE J.</i> 65, 466 (2019)</td><td class="c-dim">AI in chemical engineering</td></tr>
    </table>`, 0, 'rise')}`,
});

/* ------------------------------------------------------------ 5. next time */
Deck.add({
  id: 'next', title: 'Next: making deep learning work', beats: 1, dur: 5200, chrome: false, min: 1,
  notes: 'Next time: making deep learning work. Today we met initialisation and regularisation; next time, how normalisation and residual connections beat the vanishing gradient in really deep networks, architectures as built-in assumptions, and embeddings: how words become vectors whose geometry carries meaning. That is the bridge to attention and transformers. Thank you.',
  html: () => `
    ${at(150, 300, '<div class="eyebrow rise" data-b="0">Next · Lecture 3</div>', 0, 'rise')}
    ${at(150, 370, words('Making deep learning work', 'h1', 0, 'style="width:1700px;white-space:nowrap;font-size:100px;--d:200ms"'), 0, 'fade')}
    ${at(156, 560, '<div class="lede" style="font-style:italic">normalisation · residuals · architectures · embeddings</div>', 0, 'rise', '--d:700ms')}
    ${at(156, 820, '<div class="lbl">From Decision Trees to LLMs · Lecture 2 · Paras Koundal</div>', 0, 'rise', '--d:1100ms')}`,
  frame(g, S) {
    // a field of tiny balls all rolling to their own minima, as in the cold open
    const a = S.p(1600, 400);
    for (let i = 0; i < 70; i++) {
      const r = RNG(100 + i), x0 = 1000 + r.u(0, 800), y0 = 650 + r.u(-120, 260), ph = r.u(0, TAU);
      const t = ((S.t / 4000) + r.u(0, 1)) % 1;
      const x = lerp(x0, 1450 + 200 * Math.cos(ph), Ease.out(t)), y = lerp(y0, 760 + 80 * Math.sin(ph), Ease.out(t));
      D.dot(x, y, 4, [COL.model, COL.data, COL.grad][i % 3], a * (1 - t * .6), 8);
    }
  },
});
