# Lecture 2: Learning by gradients, 2-hour edition (animated build)

Every visual in this lecture is drawn in code: an HTML canvas engine plus JavaScript scenes. A recorder renders each step to an HD video clip containing only the graphics. The PowerPoint puts each clip full-screen, one per click, and lays every title, label, caption, card, table and equation over it as a native PowerPoint object you can edit, move or restyle. The narration is in the speaker notes.

## What is here

| path | what it does |
|---|---|
| `index.html` | The live deck. Open it in Chrome and use → / space to advance, ← to go back, `o` for the overview and `n` for the notes. |
| `js/core.js` | The engine: the 1920×1080 stage, the drawing kit (`D`, `Plot`, the 3-D `Cam`, `drawSurface`, `heatImage`, contours), text reveals and the deck controller. |
| `js/lab.js` | The data and models, all simulated and seeded: feed/yield, Arrhenius, lactate pass/fail, the operating window, Haldane kinetics, an MLP with hand-written backprop, and the worked backprop example. |
| `js/ch0.js` … `js/ch8.js`, `js/x1.js` … `js/x6.js` | The scenes. `chN.js` holds the core chapters (chapter 7, Gradients for engineers, is new in this edition; ch8 is the wrap-up). `xN.js` holds the extra scenes for chapter N, inserted with `Deck.insert(beforeId, scene)`. Each scene has `beats` (clicks), `dur` (ms per beat), `notes` (narration per beat), `html` (text overlays) and `frame` (canvas drawing as a pure function of time). |
| `js/rec.js` | Record mode (`index.html?rec`): a virtual clock, so every frame is exact. `?rec&split` hides the static text, and `REC.text()` exports its position, font, colour and reveal timing. |
| `tools/build.mjs` | Pre-renders all TeX to SVG with MathJax into `js/math.js`, then bundles `dist/*.html` as single files. |
| `tools/record.cjs` | Renders every beat to `sNN_<scene>_bB.mp4` (Playwright screenshots piped to ffmpeg/libx264) plus poster frames. |
| `tools/extract.cjs` | Exports the text layer of every beat (boxes, lines, rectangles, equation PNGs with their LaTeX) as JSON. |
| `tools/assemble2.py` | Builds the .pptx: per beat, the graphics-only clip at the back, native text, shapes and equation pictures on top with fade-in timings, narration in the notes, one section per chapter. |
| `tools/assemble.py` | The older all-video assembler (text burned into the clips). |
| `fonts-ttf/` | TTF files of the three fonts. Install them on the presenting machine. |
| `tools/shoot.cjs` | QA: saves the last frame of every beat as a PNG. |
| `exercises/lecture2_exercises.py` | The six hands-on exercises from the wrap-up (numpy + matplotlib). |

## Rebuild everything

```bash
cd tools && npm install && cd ..           # mathjax-full (once); Playwright and ffmpeg must be installed
node tools/build.mjs                       # maths → js/math.js, single-file HTML → dist/
node tools/record.cjs clips --split --workers 4   # graphics-only clips, ~25 min on 4 cores
node tools/extract.cjs text                       # the native text layer, as JSON + equation PNGs
python3 tools/assemble2.py clips text "Lecture 2 - Learning by gradients (2-hour).pptx"   # needs python-pptx
```

To change a scene, edit its `js/chN.js`, open `index.html` to check it live, then re-record only that scene: `node tools/record.cjs clips "^scene-id$" --split`. Then re-run `extract.cjs` and `assemble2.py`. For text-only edits, just change the text in PowerPoint.

## Conventions

- Colours mean the same thing everywhere: amber = data, cyan = model, pink = loss/error, violet = gradient, green = success. Class 0 is blue and class 1 is amber.
- All data are simulated and labelled as such on screen. Every number shown (fits, gradients, accuracies) is computed live from `lab.js`, not typed in.
- Fonts (Unbounded, IBM Plex Sans and Mono) are vendored in `fonts/`, so the deck works offline. For the PowerPoint, install the TTFs in `fonts-ttf/` (double-click each on Mac or Windows); the boxes name the faces `Unbounded`, `Unbounded SemiBold`, `IBM Plex Sans`, `IBM Plex Sans Medium` and so on.
- Text that changes every frame (axis tick labels drawn on the canvas, live counters such as epoch numbers) stays in the video.

## Presenting

In the PowerPoint, each click starts the next clip. Each clip ends on its final frame and holds there while you talk, so the narration in the notes can run as long as you like. The deck is about two hours: roughly 100 minutes of narration plus seven "pause and think" quizzes. Autoplay works in PowerPoint for Windows and Mac. In Keynote or Google Slides you may need to click each video to start it, so the browser version (`dist/Lecture 2 - Learning by gradients (2-hour edition).html`) is the fallback: it is the same deck, live, with no video at all.
