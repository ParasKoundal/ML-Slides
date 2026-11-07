/* rec.js: hooks used by tools/record.mjs to render each beat as a video clip.
 * Open index.html?rec and the deck runs on a virtual clock that only moves
 * when the recorder says so, so every frame is exact and repeatable.        */
'use strict';
if (/[?&]rec\b/.test(location.search)) Deck.rec = true;
window.REC = {
  scenes() {
    return Deck.scenes.map((s, i) => ({
      i, id: s.id, title: s.title || s.id, ch: s.ch, beats: s.beats,
      dur: Array.from({ length: s.beats }, (_, b) => (Array.isArray(s.dur) ? (s.dur[b] ?? s.dur[s.dur.length - 1]) : (s.dur || 3000))),
      notes: Array.isArray(s.notes) ? s.notes : [s.notes || ''],
    }));
  },
  chapters() { return Deck.chapters; },
  start(i) { Deck.vnow += 10000; Deck.entered = false; Deck.go(i, 0); Deck.frame(Deck.vnow); },
  beat(b) { Deck.setBeat(b); Deck.frame(Deck.vnow); },
  step(ms) { Deck.vnow += ms; Deck.frame(Deck.vnow); },
  errors() { return Deck.errors || 0; },
};
