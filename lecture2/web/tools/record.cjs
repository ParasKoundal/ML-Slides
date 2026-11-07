// Render every beat of the deck to an MP4 clip (plus a poster frame) for the PowerPoint build.
//   node tools/record.cjs [outdir=clips] [sceneIdFilter] [--fps 25] [--workers 3] [--skip-existing]
// Each worker opens index.html?rec in its own browser page, where the deck runs on a virtual clock,
// steps it frame by frame, screenshots each frame and pipes the JPEGs into ffmpeg (libx264).
const { chromium } = require('playwright');
const { spawn } = require('child_process');
const path = require('path'), fs = require('fs');

const args = process.argv.slice(2);
const opt = (k, d) => { const i = args.indexOf(k); return i >= 0 ? args.splice(i, 2)[1] : d; };
const FPS = +opt('--fps', 25), WORKERS = +opt('--workers', 3), CRF = opt('--crf', '23');
const SKIP = args.includes('--skip-existing'); if (SKIP) args.splice(args.indexOf('--skip-existing'), 1);
const OUT = path.resolve(args[0] || 'clips'), FILT = args[1] ? new RegExp(args[1]) : null;
fs.mkdirSync(OUT, { recursive: true });

function encoder(file) {
  const ff = spawn('ffmpeg', ['-y', '-loglevel', 'error', '-f', 'image2pipe', '-framerate', String(FPS), '-c:v', 'mjpeg', '-i', '-',
    '-c:v', 'libx264', '-preset', 'veryfast', '-crf', CRF, '-pix_fmt', 'yuv420p', '-movflags', '+faststart', file], { stdio: ['pipe', 'ignore', 'inherit'] });
  const done = new Promise((res, rej) => ff.on('close', c => (c === 0 ? res() : rej(new Error('ffmpeg exit ' + c)))));
  const write = buf => new Promise(res => (ff.stdin.write(buf) ? res() : ff.stdin.once('drain', res)));
  return { write, end: () => { ff.stdin.end(); return done; } };
}

(async () => {
  const browser = await chromium.launch();
  const probe = await browser.newPage({ viewport: { width: 1920, height: 1080 } });
  await probe.goto('file://' + path.resolve(__dirname, '../index.html') + '?rec');
  await probe.waitForFunction(() => typeof Deck !== 'undefined' && Deck.stage);
  const scenes = (await probe.evaluate(() => REC.scenes())).filter(s => !FILT || FILT.test(s.id));
  await probe.close();
  fs.writeFileSync(path.join(OUT, 'manifest.json'), JSON.stringify(scenes, null, 1));

  // balance whole scenes across workers by total duration (live simulations need the scene's own state)
  const jobs = scenes.map(s => ({ s, w: s.dur.reduce((a, b) => a + b, 0) })).sort((a, b) => b.w - a.w);
  const bins = Array.from({ length: WORKERS }, () => ({ w: 0, list: [] }));
  jobs.forEach(j => { const b = bins.reduce((m, x) => (x.w < m.w ? x : m)); b.list.push(j.s); b.w += j.w; });
  const t0 = Date.now(); let doneFrames = 0;
  const total = scenes.reduce((a, s) => a + s.dur.reduce((x, d) => x + Math.ceil(d * FPS / 1000), 0), 0);

  await Promise.all(bins.map(async bin => {
    const page = await browser.newPage({ viewport: { width: 1920, height: 1080 }, deviceScaleFactor: 1 });
    page.on('pageerror', e => console.error('page error:', String(e)));
    await page.goto('file://' + path.resolve(__dirname, '../index.html') + '?rec');
    await page.waitForFunction(() => typeof Deck !== 'undefined' && Deck.stage);
    for (const s of bin.list.sort((a, b) => a.i - b.i)) {
      for (let b = 0; b < s.beats; b++) {
        const name = `s${String(s.i).padStart(2, '0')}_${s.id}_b${b}`;
        const file = path.join(OUT, name + '.mp4');
        await page.evaluate(([i, b]) => (b === 0 ? REC.start(i) : REC.beat(b)), [s.i, b]);
        const n = Math.ceil(s.dur[b] * FPS / 1000);
        if (SKIP && fs.existsSync(file) && fs.existsSync(path.join(OUT, name + '_last.jpg'))) {
          await page.evaluate(([k, ms]) => { for (let f = 1; f < k; f++) REC.step(ms); }, [n, 1000 / FPS]);   // keep the scene's clock and state in step
          doneFrames += n; continue;
        }
        const enc = encoder(file);
        for (let f = 0; f < n; f++) {
          if (f) await page.evaluate(ms => REC.step(ms), 1000 / FPS);
          const jpg = await page.screenshot({ type: 'jpeg', quality: 92 });
          if (f === 0) fs.writeFileSync(path.join(OUT, name + '_first.jpg'), jpg);
          if (f === n - 1) fs.writeFileSync(path.join(OUT, name + '_last.jpg'), jpg);
          await enc.write(jpg); doneFrames++;
        }
        await enc.end();
        const el = (Date.now() - t0) / 1000;
        console.log(`${name}  ${doneFrames}/${total} frames  ${el.toFixed(0)}s  eta ${(el / doneFrames * (total - doneFrames) / 60).toFixed(1)} min`);
      }
    }
    await page.close();
  }));
  await browser.close();
  console.log('done in', ((Date.now() - t0) / 60000).toFixed(1), 'min');
})();
