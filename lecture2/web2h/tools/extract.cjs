// Export the static text, shapes and maths of every beat, for native PowerPoint objects.
//   node tools/extract.cjs [outdir=clips/text] [sceneIdFilter]
// Writes sNN_<id>_bB.json (items in stage pixels) and sNN_<id>_bB_mK.png (each equation, transparent, 2x).
const { chromium } = require('playwright');
const path = require('path'), fs = require('fs');
(async () => {
  const out = path.resolve(process.argv[2] || 'clips/text'), filt = process.argv[3] ? new RegExp(process.argv[3]) : null;
  fs.mkdirSync(out, { recursive: true });
  const browser = await chromium.launch();
  const page = await browser.newPage({ viewport: { width: 1920, height: 1080 }, deviceScaleFactor: 2 });
  const errs = []; page.on('pageerror', e => errs.push(String(e)));
  await page.goto('file://' + path.resolve(__dirname, '../index.html') + '?rec');
  await page.waitForFunction(() => typeof Deck !== 'undefined' && Deck.stage);
  const scenes = await page.evaluate(() => REC.scenes());
  let n = 0;
  for (const s of scenes) {
    if (filt && !filt.test(s.id)) continue;
    for (let b = 0; b < s.beats; b++) {
      await page.evaluate(([i, b]) => (b === 0 ? REC.start(i) : REC.beat(b)), [s.i, b]);
      await page.evaluate(ms => { for (let t = 0; t < ms; t += 40) REC.step(40); }, s.dur[b]);
      const name = `s${String(s.i).padStart(2, '0')}_${s.id}_b${b}`;
      const items = await page.evaluate(() => REC.extract());
      const maths = items.filter(it => it.type === 'math');
      if (maths.length) {
        await page.evaluate(() => REC.mathMode(true));
        for (const m of maths) {
          const pad = 8, clip = { x: Math.max(0, m.x - pad), y: Math.max(0, m.y - pad), width: m.w + 2 * pad, height: m.h + 2 * pad };
          m.file = `${name}_m${m.mi}.png`; m.x = clip.x; m.y = clip.y; m.w = clip.width; m.h = clip.height;
          await page.screenshot({ path: path.join(out, m.file), clip, omitBackground: true });
        }
        await page.evaluate(() => REC.mathMode(false));
      }
      fs.writeFileSync(path.join(out, name + '.json'), JSON.stringify(items));
      n++;
    }
  }
  console.log('beats exported', n, 'errors', errs.length, errs.slice(0, 5).join('\n'));
  await browser.close();
})();
