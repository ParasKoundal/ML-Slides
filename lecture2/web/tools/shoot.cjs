// QA: render the last frame of every beat to PNG (fast check of layout).
//   node tools/shoot.cjs [outdir] [sceneIdFilter]
const { chromium } = require('playwright');
const path = require('path'), fs = require('fs');
(async () => {
  const out = process.argv[2] || 'shots', filt = process.argv[3] ? new RegExp(process.argv[3]) : null;
  fs.mkdirSync(out, { recursive: true });
  const browser = await chromium.launch();
  const page = await browser.newPage({ viewport: { width: 1920, height: 1080 }, deviceScaleFactor: 1 });
  const errs = []; page.on('console', m => { if (m.type() === 'error') errs.push(m.text()); }); page.on('pageerror', e => errs.push(String(e)));
  await page.goto('file://' + path.resolve(__dirname, '../index.html') + '?rec');
  await page.waitForFunction(() => typeof Deck !== 'undefined' && Deck.stage);
  const scenes = await page.evaluate(() => REC.scenes());
  for (const s of scenes) {
    if (filt && !filt.test(s.id)) continue;
    for (let b = 0; b < s.beats; b++) {
      await page.evaluate(([i, b]) => (b === 0 ? REC.start(i) : REC.beat(b)), [s.i, b]);
      await page.evaluate(ms => { for (let t = 0; t < ms; t += 40) REC.step(40); }, s.dur[b]);
      await page.screenshot({ path: `${out}/${String(s.i).padStart(2, '0')}-${s.id}-b${b}.png` });
    }
  }
  console.log('scenes', scenes.length, 'beats', scenes.reduce((a, s) => a + s.beats, 0), 'minutes of clips', (scenes.reduce((a, s) => a + s.dur.reduce((x, y) => x + y, 0), 0) / 60000).toFixed(1));
  console.log('errors', errs.length, errs.slice(0, 8).join('\n'));
  await browser.close();
})();
