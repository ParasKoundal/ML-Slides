# Lecture 2: Learning by gradients (video-style deck)

`Lecture 2 - Learning by gradients.pptx` is built entirely by code in this folder. Every plot,
number and animation comes from running the real algorithms on seeded, simulated bioprocess data.

102 slides in seven chapters. Each chapter tells the idea as a short "video" scene first and then works it in
detail, at the depth of Lecture 1: formulas explained symbol by symbol, tables worked by hand, a chemical
engineering example and a "check your understanding" quiz. The answers are in the speaker notes.

| file | what it does |
|---|---|
| `data.py` | the maths. Seeded simulated data sets, least squares, gradient descent, logistic regression, ReLU networks with hand-written backprop, and every number shown on a slide. That includes the hand-worked tables, the Arrhenius fit, Haldane kinetics, feature scaling, learning-rate stability, the gradient check and the overfitting sweep |
| `motion.py` | a small toolkit on top of python-pptx: named shapes, freeform curves, Morph transitions, chained entrance/exit animations (fade, pop with overshoot, zoom, wipe, rise), auto-advance |
| `gifs.py` | the 10 matplotlib animations embedded in the deck (3-D regression plane, scaling, learning-rate step response, non-convex descent, mini-batch SGD, 2-D logistic boundary, hidden-neuron lines, network fitting Haldane kinetics, more neurons gives a closer fit, overfitting vs. degree) plus the extrapolation figure. Output goes to `assets/gifs/` |
| `scenes_core.py` | the cinematic scenes, driven by Morph: rolling ball, line fitting with squared residuals, the loss valley, sigmoid and surprise, neurons and layers, the backprop pass, the training montage |
| `scenes_detail.py` | the detail scenes: formulas, worked tables, the ChemE examples (Arrhenius, Haldane, mixed units, controller gain, transfer functions in series, runaway prediction), quizzes, glossary, hands-on PyTorch, reading list |
| `build_lecture2.py` | the storyboard: calls the scenes in order, then adds the chapter scrubber and transitions and saves the deck |

## Rebuild

```bash
pip install python-pptx numpy matplotlib lxml pillow
python gifs.py            # animations -> assets/gifs (about a minute)
python build_lecture2.py   # the deck
```

## How the "video" feel works

* **Morph between slides.** A scene is a run of slides. Shapes with the same name starting with `!!`
  (Morph's forced-match prefix) glide between slides: the ball rolls, the line swings, the squares
  shrink, the neuron splits into a layer. Curves keep the same number of points so their geometry tweens.
* **Auto-playing builds.** Animations start on their own after each slide appears, in a timed chain,
  so the presenter clicks once per idea instead of once per object. Playback runs (gradient-descent
  steps, the training montage) also auto-advance.
* **Overlays instead of bullets.** Lower-third subtitles, a YouTube-style chapter scrubber, a dimmed
  overlay for the key equation, "pause and think" cards that pop in on a click, and an end screen
  with an "up next" card.
* **Speaker notes** hold the narration for every slide.

Morph needs PowerPoint 2019 or later, Microsoft 365, or PowerPoint for the web. Older versions and
Keynote fall back to a cross-fade. Run the slideshow (F5) to see the motion: the editor and
PDF exports only show the final frame of each slide.

To add a scene: write a function that calls `new(chapter, notes)`, draws shapes with names, and
registers animations with `tl.add(shape, "pop", at_ms, dur_ms)`. Reuse a `!!name` from the previous
slide to make that object glide.
