/* x3.js: extended cut, chapter 3. A probability is not a decision: thresholds and the confusion matrix with
 * real costs; ROC and precision-recall on rare events; calibration and temperature scaling. */
'use strict';

/* simulated: 2000 batches, an off-gas index x, contamination with probability sigma(2x - 3.4) (about 10 %) */
const CONT = (() => {
  const r = RNG(83), n = 2000, x = [], y = [];
  for (let i = 0; i < n; i++) { const xi = r.n(); x.push(xi); y.push(r() < sigmoid(2 * xi - 3.4) ? 1 : 0); }
  const z = x.map(v => 2 * v - 3.4), p = z.map(sigmoid);
  const pos = sum(y), neg = n - pos;
  const conf = (t, pp = p) => { let tp = 0, fp = 0; pp.forEach((v, i) => { if (v >= t) { if (y[i]) tp++; else fp++; } }); return { tp, fp, fn: pos - tp, tn: neg - fp }; };
  const ts = [1.01, ...p.slice().sort((a, b) => b - a), 0];
  const roc = ts.map(t => { const c = conf(t); return [c.fp / neg, c.tp / pos, t, c.tp + c.fp ? c.tp / (c.tp + c.fp) : 1]; });
  // thin the curves for drawing (every 10th threshold)
  const curve = roc.filter((_, i) => i % 10 === 0 || i === roc.length - 1);
  let auc = 0; for (let i = 1; i < roc.length; i++) auc += (roc[i][0] - roc[i - 1][0]) * (roc[i][1] + roc[i - 1][1]) / 2;
  const hist = (sel, nb = 25) => { const h = Array(nb).fill(0); p.forEach((v, i) => { if (y[i] === sel) h[Math.min(nb - 1, Math.floor(v * nb))]++; }); return h; };
  const CFN = 40000, CFP = 800, cost = t => { const c = conf(t); return c.fn * CFN + c.fp * CFP; };
  // an over-confident model: same ranking, logits stretched by 2.2
  // calibration is checked on a larger history (20,000 batches, same process) so the bins are not noisy
  const rc = RNG(84), zc = [], yc = [];
  for (let i = 0; i < 20000; i++) { const v = 2 * rc.n() - 3.4; zc.push(v); yc.push(rc() < sigmoid(v) ? 1 : 0); }
  const T0 = 2.6, zOver = zc.map(v => v * T0);
  const reli = (pp, nb = 10) => Array.from({ length: nb }, (_, k) => { const idx = pp.map((v, i) => [v, i]).filter(([v]) => v >= k / nb && v < (k + 1) / nb + (k === nb - 1 ? 1e-9 : 0)); return idx.length > 20 ? { p: mean(idx.map(([v]) => v)), f: mean(idx.map(([, i]) => yc[i])), n: idx.length } : null; }).filter(Boolean);
  const pc = zc.map(sigmoid);
  return { n, x, y, p, z, pos, neg, conf, curve, auc, hist, CFN, CFP, cost, tStar: CFP / (CFP + CFN), T0, zOver, reli, pc };
})();
const tSweep = (S) => S.beat === 0 ? 0.5 : S.beat === 1 ? lerp(0.5, 0.1, Ease.inOut(S.at(1, 3600, 600))) : lerp(0.1, CONT.tStar, Ease.inOut(S.at(2, 2400, 1200)));

Deck.insert('softmax', {
  id: 'threshold', title: 'A probability is not a decision', beats: 3, dur: [5600, 5600, 6400], min: 4,
  notes: [
    'Logistic regression gives a probability. Someone still has to decide. Here are 2000 simulated batches with an off-gas index that predicts bacterial contamination; about 10 percent are contaminated. Clean batches are stacked above the axis in blue, contaminated ones below it in amber, by the predicted probability. The default rule is flag the batch if p is above one half. The confusion matrix on the right counts the outcomes: contaminated and flagged, contaminated and missed, clean but flagged, clean and passed. At one half we catch 59 and miss 143: most contaminated batches slip through.',
    'Slide the threshold down. Every contaminated batch to the right of the line is caught, every clean batch to the right of it is a false alarm. Lowering the threshold trades misses for false alarms. There is no setting that removes both: the two distributions overlap, and the model has said everything the off-gas index can tell it.',
    'So which threshold? That is not a machine-learning question; it is an engineering-economics one. Suppose a missed contamination ruins a 10,000 litre batch, about forty thousand euros, and a false alarm costs a lab test and a delay, about eight hundred. Flag whenever the expected cost of passing, p times forty thousand, exceeds the cost of flagging, one minus p times eight hundred. That gives a threshold of about two percent, not fifty. We now catch 191 of 202, at the price of about a thousand extra lab tests, and the expected cost falls from about 5.7 to 1.3 million euros. The probabilities are the model\'s job; the costs are yours. This only works if the probabilities are honest, which is the subject of the next two scenes.',
  ],
  html: () => `${hdr('Chapter 3 · decisions', 'A probability is not a decision')}
    ${at(1200, 268, `<div style="display:grid;grid-template-columns:140px 210px 210px;grid-template-rows:30px 120px 120px;gap:10px;align-items:start">
      <div></div><div class="lbl" style="text-align:center">flagged</div><div class="lbl" style="text-align:center">passed</div>
      <div class="lbl c-data" style="margin-top:46px">contam.</div><div class="card" style="height:120px;padding:14px;text-align:center"><div class="lbl c-good">caught</div></div><div class="card" style="height:120px;padding:14px;text-align:center"><div class="lbl c-loss">missed</div></div>
      <div class="lbl" style="color:#3f8cff;margin-top:46px">clean</div><div class="card" style="height:120px;padding:14px;text-align:center"><div class="lbl c-loss">false alarm</div></div><div class="card" style="height:120px;padding:14px;text-align:center"><div class="lbl c-good">passed</div></div></div>`, 0, 'rise')}
    ${[['cm-tp', 0, 0, 'c-good'], ['cm-fn', 1, 0, 'c-loss'], ['cm-fp', 0, 1, 'c-loss'], ['cm-tn', 1, 1, 'c-good']].map(([id, i, j, c]) => at(1200 + 150 + i * 220, 268 + 40 + j * 130 + 50, `<div id="${id}" class="mono ${c}" style="width:210px;text-align:center;font:600 44px/1 var(--mono)"></div>`, 0, 'rise')).join('')}
    ${at(1200, 590, '<div class="lbl">threshold</div>', 0, 'rise')}
    ${at(1200, 625, '<div id="cm-t" class="mono c-loss" style="font:600 64px/1 var(--mono)"></div>', 0, 'rise')}
    ${at(1480, 590, '<div class="lbl">expected cost</div>', 2, 'rise', '--d:800ms')}
    ${at(1480, 625, '<div id="cm-c" class="mono" style="font:600 64px/1 var(--mono)"></div>', 2, 'rise', '--d:800ms')}
    ${at(1200, 730, `<div class="lbl">flag if</div><div class="eq-s" style="margin-top:10px">${M(T`p > \frac{C_{\mathrm{FP}}}{C_{\mathrm{FP}} + C_{\mathrm{FN}}} = \frac{800}{40\,800} \approx 0.02`)}</div>`, 2, 'rise', '--d:600ms')}
    ${sub('Flag if p > 0.5? Most contaminated batches slip through.', 0, 1)}
    ${sub('Lower the threshold: <b>fewer misses, more false alarms.</b> Never neither.', 1, 2)}
    ${sub('A miss costs €40,000, a false alarm €800: flag above 2 %, not 50 %.', 2)}`,
  frame(g, S) {
    const P = new Plot(150, 300, 960, 520, [0, 1], [-1, 1]), a0 = S.at(0, 700), nb = 25;
    const hc = CONT.hist(0, nb), hp = CONT.hist(1, nb), bw = P.w / nb;
    const mc = Math.sqrt(Math.max(...hc)), mp = Math.max(...hp);   // each class on its own scale, so the rare one is visible
    hc.forEach((v, k) => D.rect(P.x + k * bw + 2, P.Y(0) - Math.sqrt(v) / mc * P.h * .46, bw - 4, Math.sqrt(v) / mc * P.h * .46, COL.cls0, .85 * a0));
    hp.forEach((v, k) => D.rect(P.x + k * bw + 2, P.Y(0), bw - 4, v / mp * P.h * .46, COL.cls1, .9 * a0));
    D.seg(P.x, P.Y(0), P.x + P.w, P.Y(0), '#3a4352', 2, a0);
    [0, 0.25, 0.5, 0.75, 1].forEach(t => D.text(String(t), P.X(t), P.y + P.h + 26, { size: 20, align: 'center', a: a0 }));
    D.text('predicted probability of contamination  p', P.x + P.w, P.y + P.h + 62, { size: 22, align: 'right', a: a0, font: '"IBM Plex Sans", sans-serif' });
    D.text('clean batches  (√ count)', P.x, P.y - 10, { size: 22, col: COL.cls0, a: a0, font: '"IBM Plex Sans", sans-serif' });
    D.text('contaminated  (count)', P.x, P.y + P.h - 20, { size: 22, col: COL.cls1, a: a0, font: '"IBM Plex Sans", sans-serif' });
    const t = tSweep(S), X = P.X(t);
    D.rect(X, P.y, P.x + P.w - X, P.h, COL.loss, .07 * a0);
    D.seg(X, P.y - 6, X, P.y + P.h + 6, COL.loss, 3, a0);
    D.text('flag →', X + 12, P.y + 22, { size: 22, col: COL.loss, a: a0 });
    const c = CONT.conf(t), put = (id, v) => { const el = document.getElementById(id); if (el) el.textContent = v; };
    put('cm-tp', c.tp); put('cm-fn', c.fn); put('cm-fp', c.fp); put('cm-tn', c.tn); put('cm-t', t.toFixed(t < 0.1 ? 3 : 2));
    put('cm-c', '€' + (CONT.cost(t) / 1e6).toFixed(2) + 'M');
  },
});

Deck.insert('softmax', {
  id: 'roc', title: 'ROC and precision-recall for rare events', beats: 3, dur: [5800, 5400, 5600], min: 3.5,
  notes: [
    'To judge the model itself rather than one threshold, sweep the threshold through every value and plot what happens. The ROC curve puts the false-alarm rate across and the catch rate, the recall, up. A useless model sits on the diagonal; a perfect one hugs the top-left corner. The area under it, the AUC, is the probability that a randomly chosen contaminated batch gets a higher score than a randomly chosen clean one. Here it is ' + CONT.auc.toFixed(2) + '.',
    'With rare events, the ROC curve flatters. A 5 percent false-alarm rate sounds small, but 5 percent of 1800 clean batches is 90 false alarms, against about 200 contaminated batches in total. Precision answers the question the plant manager asks: of the batches we flagged, how many were really contaminated? The precision-recall curve shows it falling fast as we try to catch the last few. The dashed line is what random flagging would give: the 10 percent base rate.',
    'And never report plain accuracy for rare events. The model that says clean for every batch, and catches nothing, is 90 percent accurate. Report recall and precision at the threshold you will actually use, or the PR curve, and always say what the base rate is.',
  ],
  html: () => `${hdr('Chapter 3 · evaluation', 'Rare events: judge the whole curve')}
    ${at(150, 258, '<div class="lbl">ROC curve</div>', 0)}
    ${at(560, 258, `<div class="lbl c-model">AUC = ${CONT.auc.toFixed(2)}</div>`, 0, 'rise', '--d:3600ms')}
    ${at(1010, 258, '<div class="lbl">precision-recall curve</div>', 1)}
    ${at(1010, 430, '<div class="card" style="width:760px"><h4 class="c-loss">the 90 % trap</h4><p>“always clean” is 90 % accurate and catches nothing. Report recall and precision, and the base rate.</p></div>', 2, 'rise')}
    ${sub('Sweep the threshold: catch rate against false-alarm rate.', 0, 1)}
    ${sub('Precision: of the flagged batches, how many were real? <b>It falls fast.</b>', 1, 2)}
    ${sub('With 10 % positives, accuracy lies. Report precision, recall and base rate.', 2)}`,
  frame(g, S) {
    const P = new Plot(150, 310, 760, 500, [0, 1], [0, 1]), Q = new Plot(1010, 310, 760, 500, [0, 1], [0, 1]), a0 = S.at(0, 700);
    P.axes({ xt: [0, 0.5, 1], yt: [0, 0.5, 1], xl: 'false-alarm rate', yl: '', a: a0, grid: true });
    D.text('recall (catch rate)', P.x + 10, P.y + 16, { size: 20, a: a0, font: '"IBM Plex Sans", sans-serif' });
    D.seg(P.X(0), P.Y(0), P.X(1), P.Y(1), '#3a4352', 2, a0, [6, 8]);
    const cv = CONT.curve, n0 = S.beat === 0 ? Math.floor(S.at(0, 3400, 500) * (cv.length - 1)) : cv.length - 1;
    D.poly([[P.X(0), P.Y(0)], ...cv.slice(0, n0 + 1).map(([f, r]) => [P.X(f), P.Y(r)]), [P.X(cv[n0][0]), P.Y(0)]], COL.model, .12 * a0);
    D.line(cv.slice(0, n0 + 1).map(([f, r]) => [P.X(f), P.Y(r)]), COL.model, 4, a0, null, 10);
    D.dot(P.X(cv[n0][0]), P.Y(cv[n0][1]), 9, '#fff', a0);
    if (S.beat >= 1) {
      const a = S.at(1, 700), n1 = S.beat === 1 ? Math.floor(S.at(1, 3400, 500) * (cv.length - 1)) : cv.length - 1;
      Q.axes({ xt: [0, 0.5, 1], yt: [0, 0.5, 1], xl: 'recall', a: S.beat === 2 ? a * .3 : a, grid: true });
      D.text('precision', Q.x + 10, Q.y + 16, { size: 20, a, font: '"IBM Plex Sans", sans-serif' });
      const base = CONT.pos / CONT.n;
      D.seg(Q.X(0), Q.Y(base), Q.X(1), Q.Y(base), COL.data, 2, a, [6, 8]);
      D.text('base rate ' + Math.round(base * 100) + ' %', Q.X(0) + 10, Q.Y(base) - 18, { size: 20, col: COL.data, a });
      const pr = cv.slice(1, n1 + 1).map(([, r, , p]) => [Q.X(r), Q.Y(p)]);
      D.line(pr, COL.grad, 4, S.beat === 2 ? a * .3 : a, null, 10);
    }
  },
});

Deck.insert('softmax', {
  id: 'calibration', title: 'Calibration: are the probabilities honest?', beats: 3, dur: [5200, 5200, 6000], min: 3.5,
  notes: [
    'The threshold rule only works if a predicted 10 percent really means 10 percent. Check it with a reliability diagram. Group the batches by predicted probability, then in each group compare the average prediction with how often contamination actually happened. Dot size is the number of batches in the group. Logistic regression, trained by cross-entropy on enough data, is usually close to the diagonal: when it says 30 percent, about 30 percent are contaminated.',
    'Here is an over-confident model. It ranks the batches exactly as well, same ROC curve, same AUC, but its probabilities are pushed toward zero and one. When it says 90 percent, the truth is nearer 70; when it says 5 percent, nearer 20. (These diagrams use a longer history, 20,000 batches, so the bins are not noisy.) Big networks trained for a long time are often like this, because the cross-entropy keeps rewarding them for pushing the training examples to the extremes. Feed these numbers into the cost rule and you get the wrong threshold.',
    'The fix is cheap. Temperature scaling divides the model\'s logits, the z before the sigmoid, by a single number T, fitted on held-out validation data. It cannot change the ranking, only the confidence. Here T of about 2.6 brings the curve back to the diagonal. Platt scaling and isotonic regression are the close relatives you will find in scikit-learn as CalibratedClassifierCV. Any time a probability feeds a decision, a cost, or a safety case, check calibration.',
  ],
  html: () => `${hdr('Chapter 3 · calibration', 'When it says 30 %, is it 30 %?')}
    ${at(1250, 300, '<div class="lbl">reliability diagram</div><div class="lede" style="font-size:27px;margin-top:10px;width:540px">bin the batches by predicted p, then count how often contamination really happened</div>', 0)}
    ${at(1250, 520, '<div class="lbl c-loss">over-confident model</div><div class="lede" style="font-size:27px;margin-top:10px;width:540px">same ranking, same AUC, wrong probabilities</div>', 1, 'rise', '', 2)}
    ${at(1250, 520, `<div class="lbl c-good">temperature scaling</div><div class="eq-s" style="margin-top:14px">${M(T`p = \sigma\!\left(z / T\right)`)}</div><div class="lede" style="font-size:27px;margin-top:10px;width:540px">one number, fitted on validation data</div>`, 2, 'rise')}
    ${sub('A well-trained logistic model sits on the diagonal.', 0, 1)}
    ${sub('Over-confident: says 90 %, means 70 %. <b>The ranking is fine; the numbers are not.</b>', 1, 2)}
    ${sub('Divide the logits by T, fitted on validation: back on the diagonal.', 2)}`,
  frame(g, S) {
    const P = new Plot(150, 300, 900, 520, [0, 1], [0, 1]), a0 = S.at(0, 700);
    P.axes({ xt: [0, 0.2, 0.4, 0.6, 0.8, 1], yt: [0, 0.5, 1], xl: 'predicted probability', a: a0, grid: true, xf: v => v.toFixed(1) });
    D.text('observed frequency', P.x + 10, P.y + 16, { size: 20, a: a0, font: '"IBM Plex Sans", sans-serif' });
    D.seg(P.X(0), P.Y(0), P.X(1), P.Y(1), '#c8d1de', 2, a0, [8, 8]);
    const draw = (rel, col, a) => { D.line(rel.map(b => [P.X(b.p), P.Y(b.f)]), col, 3, a); rel.forEach(b => D.dot(P.X(b.p), P.Y(b.f), 5 + Math.sqrt(b.n) / 6, col, .9 * a, 8)); };
    const show = S.beat === 0 ? Ease.out(S.at(0, 1600, 900)) : 1;
    draw(CONT.reli(CONT.pc), COL.model, show * (S.beat === 0 ? 1 : .35));
    if (S.beat >= 1) {
      const T = S.beat === 1 ? 1 : lerp(1, CONT.T0, Ease.inOut(S.at(2, 3000, 600)));
      const pp = CONT.zOver.map(z => sigmoid(z / T));
      draw(CONT.reli(pp), S.beat === 1 ? COL.loss : mixHex(COL.loss, COL.good, (T - 1) / (CONT.T0 - 1)), S.at(1, 800));
      if (S.beat === 2) D.text('T = ' + T.toFixed(2), P.X(0.06), P.Y(0.9), { size: 34, weight: 600, col: COL.good, a: S.at(2, 600) });
    }
  },
});
