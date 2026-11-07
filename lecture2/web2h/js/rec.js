/* rec.js: hooks used by tools/record.cjs and tools/extract.cjs.
 * index.html?rec          the deck runs on a virtual clock that only moves when told to, so frames are exact.
 * index.html?rec&split    as above, but static text is hidden from the picture: it is exported separately
 *                         (REC.extract) and rebuilt as native, editable PowerPoint objects.
 * Text that changes every frame (any [data-b] block containing an element with an id) stays in the video. */
'use strict';
if (/[?&]rec\b/.test(location.search)) Deck.rec = true;
const SPLIT = /[?&]split\b/.test(location.search);

function markKeep() {
  const layer = Deck.layer; if (!layer) return;
  layer.querySelectorAll('[data-b]').forEach(el => { if (el.querySelector('[id]') || el.id) el.classList.add('keep'); });
}

/* ---- text / shape / maths export of the DOM layer, in stage pixels (1920 x 1080) */
function rgba(c) { const m = c.match(/rgba?\(([^)]+)\)/); if (!m) return null; const v = m[1].split(',').map(s => parseFloat(s)); return { r: v[0], g: v[1], b: v[2], a: v.length > 3 ? v[3] : 1 }; }
function hex({ r, g, b }) { return [r, g, b].map(v => Math.round(v).toString(16).padStart(2, '0')).join(''); }
const INLINE = new Set(['inline', 'inline-block']);
function effOpacity(el) { let o = 1; for (let e = el; e && e !== Deck.layer && e !== document.documentElement; e = e.parentElement) o *= parseFloat(getComputedStyle(e).opacity); return o; }
function revealOf(el) { const r = el.closest('[data-b]'); return r ? { b: +r.dataset.b, d: parseFloat(r.style.getPropertyValue('--d')) || 0, kind: ['words', 'pop', 'blur', 'wipe', 'slam', 'rise', 'fade'].find(k => r.classList.contains(k)) || 'fade' } : { b: 0, d: 0, kind: 'fade' }; }

function extract() {
  const layer = Deck.layer, st = Deck.stage.getBoundingClientRect(), sc = st.width / 1920;
  const box = el => { const r = el.getBoundingClientRect(); return { x: (r.left - st.left) / sc, y: (r.top - st.top) / sc, w: r.width / sc, h: r.height / sc }; };
  const items = [];
  const all = [...layer.querySelectorAll('*')].filter(el => !el.closest('.keep') && !el.closest('.m svg'));
  // 1. shapes: backgrounds and borders, in document order (they sit under all text)
  for (const el of all) {
    if (el.closest('.m')) continue;
    const cs = getComputedStyle(el), op = effOpacity(el); if (op < 0.08) continue;
    const bg = rgba(cs.backgroundColor), bw = parseFloat(cs.borderTopWidth), bb = parseFloat(cs.borderBottomWidth);
    const full = bw > 0 && parseFloat(cs.borderLeftWidth) > 0 && cs.borderTopStyle !== 'none';
    const b = box(el); if (b.w < 2 || b.h < 2) continue;
    if ((bg && bg.a > 0.02) || full) {
      items.push({ type: 'rect', ...b, fill: bg && bg.a > 0.02 ? hex(bg) : null, fillA: bg ? bg.a * op : 0,
        stroke: full ? hex(rgba(cs.borderTopColor)) : null, strokeA: full ? rgba(cs.borderTopColor).a * op : 0, strokeW: full ? bw : 0,
        radius: parseFloat(cs.borderTopLeftRadius) || 0, reveal: revealOf(el) });
    } else if (bb > 0 && cs.borderBottomStyle !== 'none') {
      const c = rgba(cs.borderBottomColor);
      items.push({ type: 'line', x: b.x, y: b.y + b.h - bb / 2, w: b.w, h: 0, stroke: hex(c), strokeA: c.a * op, strokeW: bb, reveal: revealOf(el) });
    }
  }
  // 2. maths (captured as images by the extractor)
  layer.querySelectorAll('.m[data-tex]').forEach((el, k) => {
    if (el.closest('.keep')) return; const op = effOpacity(el); if (op < 0.08) return;
    el.dataset.mi = String(k);
    items.push({ type: 'math', mi: k, tex: el.dataset.tex, ...box(el), alpha: op, reveal: revealOf(el) });
  });
  // 3. text blocks: block-level elements that directly hold text or inline children
  const isBlock = el => !INLINE.has(getComputedStyle(el).display) || el.classList.contains('code');
  const textBlock = (el, reveal) => {
    const op = effOpacity(el); if (op < 0.08) return;
    const cs = getComputedStyle(el), pre = /pre/.test(cs.whiteSpace);
    const runs = []; let first = null, lastTop = null, lineH = parseFloat(cs.lineHeight) || parseFloat(cs.fontSize) * 1.2;
    const push = (t, n) => {
      const ps = getComputedStyle(n.parentElement), col = rgba(ps.color) || { r: 233, g: 238, b: 245, a: 1 };
      const o = effOpacity(n.parentElement); if (o < 0.08) return;
      const st_ = { font: ps.fontFamily.split(',')[0].replace(/["']/g, '').trim(), size: parseFloat(ps.fontSize), weight: +ps.fontWeight || 400,
        italic: ps.fontStyle === 'italic', color: hex(col), alpha: col.a * o, spacing: parseFloat(ps.letterSpacing) || 0, upper: ps.textTransform === 'uppercase' };
      if (st_.upper) t = t.toUpperCase();
      const last = runs[runs.length - 1];
      if (last && !last.br && JSON.stringify(last.s) === JSON.stringify(st_)) last.t += t; else runs.push({ t, s: st_ });
    };
    const walk = node => {
      for (const n of node.childNodes) {
        if (n.nodeType === 3) {
          let txt = n.textContent; if (!pre) txt = txt.replace(/\s+/g, ' ');
          // find where the browser wrapped the line, so PowerPoint breaks in the same places
          const rg = document.createRange(); let cur = '';
          for (let i = 0; i < txt.length && i < n.textContent.length; i++) {
            if (!pre && /\s/.test(n.textContent[i])) { cur += ' '; continue; }
            rg.setStart(n, i); rg.setEnd(n, i + 1); const rs = rg.getClientRects();
            if (rs.length) { const top = rs[0].top / sc; if (first === null) { const pcs = getComputedStyle(n.parentElement); first = { top: (rs[0].top - st.top) / sc, h: rs[0].height / sc, lh: parseFloat(pcs.lineHeight) || parseFloat(pcs.fontSize) * 1.2 }; } if (lastTop !== null && top > lastTop + lineH * 0.5) { if (cur) push(cur.replace(/ +$/, ''), n); cur = ''; const l = runs[runs.length - 1]; if (l && !l.br) l.t = l.t.replace(/ +$/, ''); runs.push({ br: true }); } lastTop = top; }
            cur += n.textContent[i];
          }
          if (cur) push(cur, n);
        } else if (n.nodeType === 1) {
          if (n.tagName === 'BR') { runs.push({ br: true }); lastTop = null; continue; }
          if (n.classList.contains('m')) continue;
          if (el.classList.contains('code') && n.classList.contains('ln')) { if (runs.length) runs.push({ br: true }); lastTop = null; walk(n); continue; }
          if (!INLINE.has(getComputedStyle(n).display) && !el.classList.contains('code')) continue;   // nested block: its own text box
          walk(n);
        }
      }
    };
    walk(el);
    // tidy: drop leading/trailing spaces and empty runs
    while (runs.length && runs[0].br) runs.shift();
    if (runs[0] && !pre) runs[0].t = runs[0].t.replace(/^ +/, '');
    const lastR = runs[runs.length - 1]; if (lastR && !lastR.br && !pre) lastR.t = lastR.t.replace(/ +$/, '');
    if (!runs.some(r => !r.br && r.t.trim())) return;
    const b = box(el), pl = parseFloat(cs.paddingLeft) || 0, pt = parseFloat(cs.paddingTop) || 0, pr = parseFloat(cs.paddingRight) || 0, pb = parseFloat(cs.paddingBottom) || 0;
    const blt = parseFloat(cs.borderLeftWidth) || 0, btw = parseFloat(cs.borderTopWidth) || 0;
    // place the first line where the browser drew it (the glyphs sit centred in their line box), so inline-block
    // wrappers and padding cannot shift the text; line spacing is the line-height of the text itself
    let y = b.y + pt + btw;
    if (first) { lineH = first.lh; y = first.top - (lineH - first.h) / 2; }
    items.push({ type: 'text', x: b.x + pl + blt, y, w: b.w - pl - pr - 2 * blt, h: b.h - pt - pb - 2 * btw - (y - b.y - pt - btw),
      align: cs.textAlign, lineH, runs, reveal: reveal || revealOf(el) });
  };
  for (const el of all) {
    if (el.closest('.m') || !isBlock(el)) continue;
    if (el.closest('.code') && el.closest('.code') !== el) continue;           // a code block is one text box
    const direct = [...el.childNodes].some(n => (n.nodeType === 3 && n.textContent.trim()) || (n.nodeType === 1 && INLINE.has(getComputedStyle(n).display) && !n.classList.contains('m') && n.textContent.trim()));
    if (!direct && !el.classList.contains('code')) continue;
    textBlock(el);
  }
  // 4. the footer row (chapter label, slide counter): native text, the progress track stays in the video
  const ch = Deck.chromeEl;
  if (ch && ch.style.display !== 'none') ['chapLbl', 'cnt'].forEach(id => { const n = document.getElementById(id); const k = items.length; textBlock(n, { b: -1, d: 0, kind: 'none' }); if (items[k]) items[k].footer = id; });
  return items;
}

window.REC = {
  scenes() {
    return Deck.scenes.map((s, i) => ({
      i, id: s.id, title: s.title || s.id, ch: s.ch, chTitle: (Deck.chapters[s.ch] || {}).title || '', chrome: s.chrome !== false, beats: s.beats,
      dur: Array.from({ length: s.beats }, (_, b) => (Array.isArray(s.dur) ? (s.dur[b] ?? s.dur[s.dur.length - 1]) : (s.dur || 3000))),
      notes: Array.isArray(s.notes) ? s.notes : [s.notes || ''],
    }));
  },
  chapters() { return Deck.chapters; },
  start(i) { Deck.vnow += 10000; Deck.entered = false; Deck.go(i, 0); markKeep(); Deck.frame(Deck.vnow); },
  beat(b) { Deck.setBeat(b); markKeep(); Deck.frame(Deck.vnow); },
  step(ms) { Deck.vnow += ms; Deck.frame(Deck.vnow); },
  errors() { return Deck.errors || 0; },
  extract,
  /* show only one maths element on a transparent page, for an element screenshot */
  mathMode(on) {
    document.documentElement.classList.toggle('mathshot', on);
    ['viewport', 'stage'].forEach(id => { document.getElementById(id).style.background = on ? 'transparent' : ''; });
    document.body.style.background = on ? 'transparent' : '';
    Deck.cv.style.visibility = on ? 'hidden' : '';
  },
};
if (SPLIT) document.documentElement.classList.add('split');
