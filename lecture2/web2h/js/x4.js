/* x4.js: extended cut, chapter 4. Depth versus width (composition doubles the kinks); a network learns a
 * reactor's yield surface from 70 runs (the surrogate used again in chapter 7). */
'use strict';

/* ------------------------------------------------------------ depth: each layer doubles the kinks */
const TENT = x => 2 * Math.max(0, x) - 4 * Math.max(0, x - 0.5);      // two ReLU neurons: a tent on [0, 1]
const tentN = (x, n) => { for (let k = 0; k < n; k++) x = TENT(x); return x; };
const DEPTH_ROWS = [[1, 2, 2, 1], [2, 4, 4, 3], [3, 6, 8, 7], [5, 10, 32, 31], [20, 40, '≈ 10⁶', '≈ 10⁶']];
Deck.insert('network', {
  id: 'depth', title: 'Why deep? Composition doubles the kinks', beats: 3, dur: [4600, 6000, 5400], min: 3.5,
  notes: [
    'Why stack layers at all, if one wide hidden layer can approximate any curve? Here is the cleanest answer. Two ReLU neurons, one hinge going up at zero and one coming down at a half, make a tent: up from 0 to 1, then back down. Two straight pieces.',
    'Now feed the tent into a copy of itself: a second layer, two more neurons. Each half of the input range gets mapped onto the whole range, so the tent is drawn twice: four pieces. A third layer gives eight. Each extra layer costs only two neurons but doubles the number of kinks. A single wide layer gets one new kink per neuron, so to match three layers it needs seven neurons, and the gap grows exponentially.',
    'Five layers, ten neurons: a sawtooth with 32 pieces. One hidden layer would need 31 neurons; twenty layers would give a million pieces from forty neurons. This is the theory behind deep networks: composition reuses features, so depth buys complexity far more cheaply than width. Real functions are not sawtooths, and depth brings its own problems, the vanishing gradients we will see in the next chapter, but this is why the networks that work best are deep rather than just wide.',
  ],
  html: () => `${hdr('Chapter 4 · depth', 'Why deep? Each layer doubles the kinks')}
    ${at(1180, 300, `<table class="table" style="font-size:24px"><tr><th>layers</th><th>neurons</th><th>pieces</th><th>one wide layer needs</th></tr>
      ${DEPTH_ROWS.map((r, k) => `<tr class="rise" data-b="${[0, 1, 1, 2, 2][k]}" style="--d:${[0, 0, 2800, 0, 1600][k]}ms"><td class="mono">${r[0]}</td><td class="mono">${r[1]}</td><td class="mono c-data">${r[2]}</td><td class="mono c-dim">${r[3]}</td></tr>`).join('')}</table>`, 0, 'rise')}
    ${at(1180, 750, `<div class="eq-xs">${M(T`f(x) = 2\,\mathrm{ReLU}(x) - 4\,\mathrm{ReLU}(x - \tfrac12)`)}</div><div class="lede" style="font-size:27px;margin-top:12px;width:600px">two neurons per layer, the same tent each time</div>`, 0, 'rise', '--d:1200ms')}
    ${sub('Two ReLU neurons make a tent: two straight pieces.', 0, 1)}
    ${sub('Feed it into itself: <b>each layer doubles the pieces.</b>', 1, 2)}
    ${sub('5 layers, 10 neurons, 32 pieces. One wide layer would need 31.', 2)}`,
  frame(g, S) {
    const P = new Plot(150, 300, 900, 500, [0, 1], [-0.1, 1.15]), a0 = S.at(0, 700);
    P.axes({ xt: [0, 0.5, 1], yt: [0, 1], xl: 'input  x', yl: 'output', a: a0 });
    let nA, nB, f;
    if (S.beat === 0) { nA = 1; nB = 1; f = 0; }
    else if (S.beat === 1) { const q = S.at(1, 5200, 300) * 2; nA = 1 + Math.min(1, Math.floor(q)); nB = Math.min(3, nA + 1); f = Ease.inOut(clamp(q - Math.floor(q))); if (q >= 2) { nA = 3; nB = 3; f = 0; } }
    else { const q = Ease.inOut(S.at(2, 2400, 400)); nA = 3; nB = 5; f = q; }
    const draw = S.at(0, 1600, 300);
    D.fn(P, x => lerp(tentN(x, nA), tentN(x, nB), f), 0, draw, COL.model, 4, a0, 1200, null, 12);
    const n = f > .5 ? nB : nA;
    D.text(`${n} layer${n > 1 ? 's' : ''} · ${2 ** n} pieces`, P.x + P.w, P.y + 10, { size: 30, weight: 600, col: COL.data, align: 'right', a: a0 });
    // a strip of the stacked layers under the plot
    for (let k = 0; k < n; k++) { const x = P.x + 40 + k * 70; D.ring(x, P.y + P.h + 120, 12, COL.data, 2, a0); D.ring(x + 26, P.y + P.h + 120, 12, COL.data, 2, a0); if (k) D.seg(x - 30, P.y + P.h + 120, x - 14, P.y + P.h + 120, '#3a4352', 2, a0); }
  },
});

/* ------------------------------------------------------------ a network learns a yield surface */
const YS_GRID = (() => {
  const us = linspace(-1, 1, 31);
  return { truth: us.map(u => us.map(v => YIELD.true_(u, v))), snaps: YIELD.snaps.map(s => ({ e: s.e, g: us.map(u => us.map(v => YIELD.evalS(s, u, v))) })) };
})();
Deck.insert('quiz-4', {
  id: 'yield-surface', title: 'A network learns a reactor', beats: 3, dur: [5000, 8000, 5600], min: 4,
  notes: [
    'A two-input regression problem from a reactor. Temperature across, residence time on a log scale going back, and the yield of an intermediate B up. On the left, the true yield surface of the simulated reactor: a curved ridge, because hotter reactors need shorter residence times before B degrades, and too cold lets a side reaction win. We do not get to see this surface. We get 70 experiments, the dots, each with a little measurement noise.',
    'On the right, a network with two inputs, 16 tanh neurons and a linear output, trained by gradient descent on squared error to those 70 points. At epoch zero it is a nearly flat random sheet. Watch it rise, tilt, and then curl into the ridge. No one told it about Arrhenius or about series reactions; it learned the shape from the data, with 65 knobs.',
    'After 2500 epochs it matches the true surface to about ' + (YIELD.rmse * 100).toFixed(1) + ' percent yield, root-mean-square, inside the region we sampled. That is a surrogate model: cheap to evaluate, smooth, and, because it is a network, differentiable. In chapter 7 we will use exactly this network to optimise the reactor and to find which input matters most. Keep in mind where its data came from: outside the sampled box it knows nothing.',
  ],
  html: () => `${hdr('Chapter 4 · regression', 'A network learns a reactor from 70 runs')}
    ${at(330, 250, '<div class="lbl">true reactor (hidden from us)</div>', 0, 'rise')}
    ${at(1190, 250, '<div class="lbl c-model">network, 2 → 16 tanh → 1</div>', 1, 'rise')}
    ${at(1190, 880, `<div class="lede" style="font-size:27px">RMS error ${(YIELD.rmse * 100).toFixed(1)} % yield, inside the sampled box</div>`, 2, 'rise')}
    ${sub('The reactor\'s true yield surface, and 70 noisy experiments.', 0, 1)}
    ${sub('A 65-knob network, trained on the 70 points: <b>it finds the ridge.</b>', 1, 2)}
    ${sub('A differentiable surrogate. Chapter 7 puts it to work.', 2)}`,
  enter(st) { st.c1 = new Cam(520, 720, 280, -0.75, 0.6, 0.12); st.c2 = new Cam(1400, 720, 280, -0.75, 0.6, 0.12); },
  frame(g, S) {
    const a0 = S.at(0, 800), yaw = -0.75 + Math.sin(S.t / 5000) * .15, zs = 1.6, o = { zscale: zs, zmin: 0, zmax: .5, stops: YSTOPS };
    S.st.c1.yaw = yaw; S.st.c2.yaw = yaw;
    drawSurface(S.st.c1, YS_GRID.truth, { ...o, a: a0 });
    const c1 = S.st.c1;
    YIELD.X.forEach(([u, v], i) => { const q = clamp(S.at(0, 2400, 1200) * 1.5 - i / YIELD.X.length * .5); const [x, y] = c1.p(u, v, YIELD.Y[i] * zs + .02); D.dot(x, y, 4.5, '#ffffff', q); });
    if (S.beat >= 1) {
      const sn = YS_GRID.snaps, q = S.beat === 1 ? S.at(1, 7000, 500) * (sn.length - 1) : sn.length - 1, k = Math.min(sn.length - 2, Math.floor(q)), f = Ease.inOut(q - k);
      const grid = sn[k].g.map((row, i) => row.map((v, j) => lerp(v, sn[k + 1].g[i][j], f)));
      drawSurface(S.st.c2, grid, { ...o, a: S.at(1, 600) });
      const e = Math.round(lerp(sn[k].e, sn[k + 1].e, f));
      D.text('epoch ' + e, 1400, 900, { size: 30, weight: 600, col: COL.model, align: 'center', a: S.at(1, 600) * (S.beat === 1 ? 1 : 0) });
    }
  },
});
