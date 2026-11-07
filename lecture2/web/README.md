# Lecture 2: Learning by gradients (animated build)

Every visual in this lecture is drawn in code: an HTML canvas engine plus JavaScript scenes. A recorder then renders each step to an HD video clip. The PowerPoint is just those clips, full-screen and one per click, with the narration in the speaker notes.

## What is here

| path | what it does |
|---|---|
| `index.html` | The live deck. Open it in Chrome and use → / space to advance, ← to go back, `o` for the overview and `n` for the notes. |
| `js/core.js` | The engine: the 1920×1080 stage, the drawing kit (`D`, `Plot`, the 3-D `Cam`, `drawSurface`, `heatImage`, contours), text reveals and the deck controller. |
| `js/lab.js` | The data and models, all simulated and seeded: feed/yield, Arrhenius, lactate pass/fail, the operating window, Haldane kinetics, an MLP with hand-written backprop, and the worked backprop example. |
| `js/ch0.js` … `js/ch7.js` | The scenes, one file per chapter. Each scene has `beats` (clicks), `dur` (ms per beat), `notes` (narration per beat), `html` (text overlays) and `frame` (canvas drawing as a pure function of time). |
| `js/rec.js` | Record mode (`index.html?rec`): a virtual clock, so every frame is exact. |
| `tools/build.mjs` | Pre-renders all TeX to SVG with MathJax into `js/math.js`, then bundles `dist/*.html` as single files. |
| `tools/record.cjs` | Renders every beat to `sNN_<scene>_bB.mp4` (Playwright screenshots piped to ffmpeg/libx264) plus poster frames. |
| `tools/assemble.py` | Builds the .pptx: one slide per beat, video set to start automatically, narration in the notes, one section per chapter. |
| `tools/shoot.cjs` | QA: saves the last frame of every beat as a PNG. |
| `exercises/lecture2_exercises.py` | The four hands-on exercises from the wrap-up (numpy + matplotlib). |

## Rebuild everything

```bash
cd tools && npm install && cd ..           # mathjax-full (once); Playwright and ffmpeg must be installed
node tools/build.mjs                       # maths → js/math.js, single-file HTML → dist/
node tools/record.cjs clips --workers 3    # ~15 min on 4 cores; clips/ holds 172 mp4s
python3 tools/assemble.py clips "Lecture 2 - Learning by gradients.pptx"   # needs python-pptx
```

To change a scene, edit its `js/chN.js`, open `index.html` to check it live, then re-record only that scene: `node tools/record.cjs clips "^scene-id$"`. Then re-run `assemble.py`.

## Conventions

- Colours mean the same thing everywhere: amber = data, cyan = model, pink = loss/error, violet = gradient, green = success. Class 0 is blue and class 1 is amber.
- All data are simulated and labelled as such on screen. Every number shown (fits, gradients, accuracies) is computed live from `lab.js`, not typed in.
- Fonts (Unbounded, IBM Plex Sans and Mono) are vendored in `fonts/`, so the deck works offline.

## Presenting

In the PowerPoint, each click starts the next clip. Each clip ends on its final frame and holds there while you talk, so the narration in the notes can run as long as you like. The deck is about 60 minutes with the six "pause and think" quizzes. Autoplay works in PowerPoint for Windows and Mac. In Keynote or Google Slides you may need to click each video to start it, so the browser version (`dist/Lecture 2 - Learning by gradients.html`) is the fallback: it is the same deck, live, with no video at all.
