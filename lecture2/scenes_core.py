"""
scenes_core.py - the main storyline scenes of Lecture 2 (the deck is assembled by build_lecture2.py).

The deck is designed to play like an explainer video:
  * one idea per scene, a scene is a run of slides joined by Morph, so objects glide
  * shapes that should glide share a name starting with "!!" (Morph's forced-match prefix)
  * entrance animations are chained automatically (no clicking through builds)
  * "playback" runs (gradient-descent steps, the training montage) auto-advance
  * subtitles in a lower-third pill, a YouTube-style chapter scrubber along the bottom
  * the voice-over script is in the speaker notes
"""
import os
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from pptx import Presentation
from pptx.util import Inches

import data as D
from motion import (Ax, Timeline, dot, rect, poly, seg, text, picture, background, set_transition, add_glow, set_alpha)

HERE = os.path.dirname(os.path.abspath(__file__))
ASSETS = os.path.join(HERE, "assets")
os.makedirs(ASSETS, exist_ok=True)
OUT = os.path.join(HERE, "Lecture 2 - Learning by gradients.pptx")

# ----------------------------------------------------------------------------- look
BG = "0E1116"
PANEL = "171B23"
INK = "F2F0EA"
MUTED = "8B93A1"
FAINT = "3A414D"
ORANGE = "F4A259"   # class 1 / data (same orange as Lecture 1)
BLUE = "5BA4E6"     # class 0
YELLOW = "FFD166"   # the model, the ball, the forward pass
RED = "EF476F"      # errors, gradients, the backward pass
TEAL = "3DDC97"     # "good" / improved
CRIMSON = "E63946"  # chapter numbers (Lecture 1 accent)
YT_RED = "FF3B3B"   # scrubber
HEAD = "Arial"
BODY = "Calibri"
MATH = "Cambria"
W, H = 13.333, 7.5

CHAPTERS = ["Intro", "A line that learns", "Rolling downhill", "Yes or no?",
            "Stacking neurons", "Blame flows backwards", "Watch it learn", "Wrap-up"]

prs = Presentation()
prs.slide_width = Inches(W)
prs.slide_height = Inches(H)
BLANK = prs.slide_layouts[6]

SCENES = []  # dicts: slide, tl, chapter, morph, advance, scrub


def new(chapter, notes, morph=1200, advance=None, scrub=True, transition="morph"):
    sl = prs.slides.add_slide(BLANK)
    background(sl, BG)
    tl = Timeline(sl)
    sl.notes_slide.notes_text_frame.text = notes.strip()
    SCENES.append(dict(slide=sl, tl=tl, chapter=chapter, morph=morph, advance=advance,
                       scrub=scrub, transition=transition))
    return sl, tl


def fmt(v, nd=2):
    s = f"{v:.{nd}f}"
    return s.replace("-", "−")


_cap_n = [0]


def caption(sl, tl, msg, at=300):
    """Lower-third subtitle pill. The pill is shared ('!!cap_bg') so it resizes smoothly;
    the words get a fresh name each time so the old line fades and the new one rises."""
    _cap_n[0] += 1
    size = 20
    cw = 0.118
    one_line = len(msg) * cw + 0.9
    if one_line <= 11.2:
        w, h = one_line, 0.56
    else:
        w, h = min(11.6, len(msg) * cw / 2 + 1.3), 0.95
    x = (W - w) / 2
    y = 6.78 - h
    bg = rect(sl, "!!cap_bg", x, y, w, h, fill="000000", alpha=0.62, radius=0.14)
    t = text(sl, f"cap_{_cap_n[0]}", x + 0.3, y, w - 0.6, h, msg, size=size, color=INK, font=BODY,
             align="c", anchor="m")
    tl.add(t, "rise", at, 450)
    return t


def kicker(sl, msg, color=MUTED, x=0.75, y=0.5, name="!!kicker"):
    return text(sl, name, x, y, 8, 0.35, msg.upper(), size=12, color=color, bold=True, font=HEAD, spacing=2)


def axes(sl, ax, xlabel, ylabel, prefix="!!ax", label_size=14, xl_w=None):
    seg(sl, f"{prefix}_x", ax.x0, ax.y0 + ax.h, ax.x0 + ax.w, ax.y0 + ax.h, FAINT, 1.25)
    seg(sl, f"{prefix}_y", ax.x0, ax.y0, ax.x0, ax.y0 + ax.h, FAINT, 1.25)
    a = text(sl, f"{prefix}_xl", ax.x0 + ax.w - (xl_w or 4.5), ax.y0 + ax.h + 0.1, xl_w or 4.5, 0.3, xlabel,
             size=label_size, color=MUTED, align="r")
    b = text(sl, f"{prefix}_yl", ax.x0 + 0.12, ax.y0 - 0.05, 4.5, 0.3, ylabel, size=label_size, color=MUTED)
    return a, b


def ball(sl, cx, cy, r=0.17):
    return dot(sl, "!!ball", cx, cy, r, YELLOW, glow=YELLOW)


def pause_card(sl, tl, question, click=1, x=8.55, y=0.45):
    """YouTube-style info card: pops up on a click."""
    w, h = 4.3, 1.3
    c = rect(sl, "pause_bg", x, y, w, h, fill=PANEL, line=YELLOW, line_w=1.25, radius=0.12)
    k = text(sl, "pause_k", x + 0.25, y + 0.18, w - 0.5, 0.3, "PAUSE AND THINK", size=12, bold=True,
             color=YELLOW, font=HEAD, spacing=2)
    q = text(sl, "pause_q", x + 0.25, y + 0.5, w - 0.5, h - 0.6, question, size=16, color=INK)
    tl.add(c, "pop", 0, 400, click).add(k, "fade", 200, 300, click).add(q, "fade", 250, 400, click)


# ============================================================================= INTRO
def terrain(xs):
    # a gently convex valley with ripples, so a straight morph path stays above the ground
    return 1.25 + 0.085 * (xs - 8.6) ** 2 + 0.12 * np.sin(xs * 2.2) * np.exp(-((xs - 8.6) ** 2) / 30)


TX = np.linspace(-0.2, W + 0.2, 120)


def terrain_pts(scale=1.0, dx=0.0, dy=0.0, cx=W / 2):
    ys = np.minimum(terrain(TX), 9.0)
    pts = [(cx + (x - cx) * scale + dx, H - ys_i * scale + dy) for x, ys_i in zip(TX, ys)]
    return pts


def mark_pts(cx, bottom, width, height):
    """the cold-open landscape shrunk into a small logo mark (same 120 points, so it morphs)"""
    ys = terrain(TX)
    y0, y1 = ys.min(), ys.max()
    return [(cx + (x - W / 2) / W * width, bottom - (y - y0) / (y1 - y0) * height) for x, y in zip(TX, ys)]


def mark_ball(cx, bottom, width, r):
    return cx + (8.6 - W / 2) / W * width, bottom - r - 0.01


def ground_y(x):
    return H - terrain(np.array([x]))[0]


def cold_open(stage):
    notes = {
        0: """[Cold open. Let it breathe.] Every model you have heard of, from the straight line in Excel to ChatGPT, learns by doing one simple thing over and over.""",
        1: """It feels its way downhill. [auto-plays]""",
        2: """[auto-plays]""",
        3: """It rolls downhill, on a landscape made of its own mistakes. Today is about that landscape, and the rule for walking down it.""",
    }[stage]
    sl, tl = new(0, notes, morph=900 if stage else 1200, advance=0 if stage in (1, 2) else None, scrub=False)
    pts = terrain_pts()
    fill_pts = pts + [(W + 0.2, H + 0.2), (-0.2, H + 0.2)]
    poly(sl, "!!terrain_fill", fill_pts, fill=PANEL, closed=True)
    poly(sl, "!!terrain", pts, color=FAINT, width=3)
    bx = [1.4, 3.6, 6.2, 8.6][stage]
    r = 0.2
    ball(sl, bx, ground_y(bx) - r - 0.02, r)
    if stage == 0:
        a = text(sl, "open1", 1.0, 1.0, 11.3, 0.8, "Every model you have heard of", size=40, bold=True, font=HEAD,
                 color=INK, align="c")
        b = text(sl, "open2", 1.0, 1.85, 11.3, 0.8, "learns by doing one simple thing.", size=40, bold=True,
                 font=HEAD, color=MUTED, align="c")
        tl.add(a, "rise", 300, 600).add(b, "rise", 900, 600)
    if stage == 3:
        a = text(sl, "open3", 1.0, 1.2, 11.3, 1.0, "It rolls downhill.", size=54, bold=True, font=HEAD,
                 color=YELLOW, align="c")
        tl.add(a, "pop", 100, 500)


def title_sting():
    sl, tl = new(0, """[Title sting] Lecture 2 of 5: Learning by gradients. Last time, trees learned by searching and averaging. Today, models learn by sliding downhill.""",
                 morph=1400, scrub=False)
    # the landscape collapses into a small valley mark above the title
    pts = mark_pts(W / 2, 2.85, 3.2, 1.0)
    poly(sl, "!!terrain", pts, color=MUTED, width=2.5)
    ball(sl, *mark_ball(W / 2, 2.85, 3.2, 0.13), 0.13)
    t1 = text(sl, "title", 0.8, 3.15, W - 1.6, 1.1, "LEARNING BY GRADIENTS", size=60, bold=True, font=HEAD,
              color=INK, align="c", spacing=1)
    t2 = text(sl, "subtitle", 0.8, 4.35, W - 1.6, 0.45, "From linear regression to neural networks", size=22,
              color=MUTED, align="c", italic=True, font=MATH)
    t3 = text(sl, "lect", 0.8, 5.55, W - 1.6, 0.35, "LECTURE 2 OF 5  ·  FROM DECISION TREES TO LLMS", size=13, bold=True,
              color=CRIMSON, align="c", font=HEAD, spacing=3)
    t4 = text(sl, "author", 0.8, 5.95, W - 1.6, 0.35, "Paras Koundal", size=14, color=MUTED, align="c")
    tl.add(t1, "pop", 0, 500)
    tl.add(t2, "rise", 700, 500).add(t3, "fade", 1100, 500).add(t4, "fade", 1250, 500)


RECAP_COLS = [1.0, 4.95, 8.9]


def three_questions(stage):
    """stage 0: Lecture 1 answers, 1: question marks for today, 2: today's answers (recap)."""
    notes = [
        """Quick recap. Every lecture answers the same three questions. What family of models are we choosing from? How do we measure how wrong a choice is? And how do we move to a better one? Last time: trees, impurity, and greedy search plus averaging.""",
        """Today we keep the questions and change every answer. And the third answer, gradients, is the engine of every lecture from here to LLMs.""",
        """So, the recap. Model: weighted sums, with a bend in between, stacked. Loss: squared error for numbers, cross-entropy for yes or no, and both come from asking how likely the data was. Learning: gradient descent, with backpropagation computing the gradient. Lecture 3 is about making this work on big networks.""",
    ][stage]
    sl, tl = new(0 if stage < 2 else 7, notes, morph=1300)
    kicker(sl, ["Last time  ·  Lecture 1", "Today  ·  Lecture 2", "What we learned today"][stage],
           color=CRIMSON if stage else MUTED)
    words = ["Model", "Loss", "Learning"]
    qs = ["Which family of functions?", "How wrong is a choice?", "How do we get better?"]
    ans = [["trees: boxes in feature space", "impurity, log-loss", "greedy search & averaging"],
           ["?", "?", "?"],
           ["weighted sums + bends, stacked", "squared error  ·  cross-entropy", "gradient descent + backprop"]][stage]
    acol = [MUTED, YELLOW, YELLOW][stage]
    for i, (x, wd, q, a) in enumerate(zip(RECAP_COLS, words, qs, ans)):
        n = text(sl, f"!!rq_n{i}", x, 1.75, 3.5, 0.6, f"{i + 1}", size=30, color=CRIMSON, font=MATH, italic=True)
        t = text(sl, f"!!rq_w{i}", x, 2.35, 3.6, 0.8, wd, size=44, bold=True, font=HEAD, color=INK)
        s = text(sl, f"!!rq_q{i}", x, 3.2, 3.6, 0.4, q, size=18, color=MUTED)
        big = a == "?"
        an = text(sl, f"!!rq_a{i}", x, 3.9, 3.6, 1.2 if not big else 1.4, a, size=22 if not big else 60,
                  color=acol, bold=big, font=BODY if not big else HEAD)
        if stage == 0:
            tl.add(n, "fade", 200 + i * 250, 400).add(t, "rise", 250 + i * 250, 500)
            tl.add(s, "fade", 400 + i * 250, 400).add(an, "fade", 1300 + i * 250, 500)
    if stage == 1:
        caption(sl, tl, "Same three questions. New answers. And answer 3 powers everything up to ChatGPT.", 600)


def chapter_card(num, title, sub, notes):
    sl, tl = new(num, notes, morph=1000)
    n = text(sl, "ch_num", 0.9, 1.9, 4, 2.0, f"{num:02d}", size=150, bold=True, font=HEAD, color=CRIMSON)
    t = text(sl, "ch_title", 0.95, 4.2, 11, 1.0, title, size=54, bold=True, font=HEAD, color=INK)
    s = text(sl, "ch_sub", 0.98, 5.2, 11, 0.5, sub, size=22, italic=True, font=MATH, color=MUTED)
    tl.add(n, "pop", 0, 500).add(t, "wipe_l", 350, 500).add(s, "fade", 800, 500)


# ============================================================================= CH1 linear regression
AX1 = Ax(1.25, 1.05, 7.7, 4.85, (0, 3.4), (0, 5.4))
AX1_MINI = Ax(0.95, 1.55, 4.5, 3.3, (0, 3.4), (0, 5.4))


def scatter_lin(sl, ax, r=0.085, alpha=1.0):
    pts = []
    for i, (x, y) in enumerate(zip(D.lin_x, D.lin_y)):
        pts.append(dot(sl, f"!!lp{i}", ax.X(x), ax.Y(y), r, ORANGE, alpha=alpha))
    return pts


def line_pts(ax, w, b, n=2):
    xs = np.linspace(ax.xl[0] + 0.05, ax.xl[1] - 0.05, n)
    out = []
    for x in xs:
        y = np.clip(w * x + b, ax.yl[0], ax.yl[1])
        out.append(ax.P(x, y))
    return out


def residuals(sl, ax, w, b, squares):
    shapes = []
    for i, (x, y) in enumerate(zip(D.lin_x, D.lin_y)):
        yh = w * x + b
        top, bot = max(y, yh), min(y, yh)
        y0 = ax.Y(top)
        hh = ax.sy(top - bot)
        if squares:
            side = hh
            x0 = ax.X(x)
            if x0 + side > ax.x0 + ax.w:
                x0 = ax.X(x) - side
            s = rect(sl, f"!!res{i}", x0, y0, max(side, 0.01), max(hh, 0.01), fill=RED, alpha=0.22, line=RED,
                     line_w=1.0)
        else:
            s = rect(sl, f"!!res{i}", ax.X(x) - 0.018, y0, 0.036, max(hh, 0.01), fill=RED)
        shapes.append(s)
    return shapes


def eq_panel(sl, w, b, loss=None, show_loss=False):
    e = text(sl, "!!eq", 9.55, 1.45, 3.6, 0.7, [("ŷ", {"italic": True}), (" = ", {}), ("w", {"italic": True, "color": YELLOW}),
                                              ("·x + ", {}), ("b", {"italic": True, "color": YELLOW})],
             size=34, font=MATH, color=INK)
    v = text(sl, "!!wb", 9.55, 2.2, 3.6, 0.4, f"w = {fmt(w)}     b = {fmt(b)}", size=18, color=MUTED, font=MATH)
    out = [e, v]
    if show_loss:
        l1 = text(sl, "!!loss_k", 9.55, 3.35, 3.4, 0.35, "LOSS  (AVERAGE AREA)", size=12, bold=True, font=HEAD,
                  color=RED, spacing=2)
        l2 = text(sl, "!!loss_v", 9.5, 3.7, 3.6, 1.2, fmt(loss), size=72, bold=True, font=HEAD, color=INK)
        out += [l1, l2]
    return out


def ch1_scene(stage):
    w0, b0 = D.CH1_START
    notes = {
        "data": """Fourteen past batches. On the x-axis, how fast we fed glucose. On the y-axis, the titre we got. More feed, more product, roughly. We want a rule that predicts titre from feed rate.""",
        "guess": """Our model family is every straight line: y-hat equals w times x plus b. Two knobs, w for the slope and b for the height. Training means picking w and b. Step one: just guess. Here's a bad guess.""",
        "miss": """How wrong is it? For each batch, measure the vertical miss between the real titre and the line.""",
        "square": """Now square each miss. Literally: turn each miss into a square. Average the areas and you have one number, the mean squared error. That number is the loss. Big squares, bad line.""",
        "fit0": """Now watch what happens when we nudge the line in the right direction. [auto-plays]""",
        "fit1": """[auto-plays]""",
        "fit2": """The squares shrink as the line swings into place. The best line is the one with the smallest total area. That's least squares, and it's the oldest machine learning there is. [Click for the pause card.] Why squares? Squaring punishes big misses much more than small ones, it's smooth so it has a slope everywhere, and it comes from assuming bell-shaped noise, which we'll see later.""",
    }[stage]
    auto = stage in ("fit0", "fit1")
    sl, tl = new(1, notes, morph=1100 if not auto else 900, advance=0 if auto else None)
    axes(sl, AX1, "glucose feed rate (g/L/h)", "titre (g/L)")
    pts = scatter_lin(sl, AX1)
    if stage == "data":
        tl.stagger(pts, "pop", 200, 70, 350)
        caption(sl, tl, "14 past batches: more glucose feed, more product. Roughly.", 1300)
        return
    it = {"guess": 0, "miss": 0, "square": 0, "fit0": 2, "fit1": 6, "fit2": D.CH1_FRAMES[-1]}[stage]
    w, b = D.CH1_PATH[it]
    if stage in ("square", "fit0", "fit1", "fit2"):
        residuals(sl, AX1, w, b, squares=True)
    elif stage == "miss":
        rs = residuals(sl, AX1, w, b, squares=False)
        tl.stagger(rs, "wipe_d", 200, 60, 300)
    ln = poly(sl, "!!line", line_pts(AX1, w, b), color=YELLOW, width=3.5)
    ex = eq_panel(sl, w, b, D.mse(w, b), show_loss=stage in ("square", "fit0", "fit1", "fit2"))
    if stage == "guess":
        tl.add(ln, "wipe_l", 200, 700).add(ex[0], "fade", 700, 500).add(ex[1], "fade", 900, 500)
        caption(sl, tl, "Step one: guess a line. Any line.", 1200)
    if stage == "miss":
        caption(sl, tl, "How wrong is it? Measure each miss.", 1100)
    if stage == "square":
        tl.add(ex[2], "fade", 200, 400).add(ex[3], "pop", 300, 450)
        caption(sl, tl, "Square each miss. The average area is the loss.", 700)
    if stage == "fit0":
        caption(sl, tl, "Nudge the line. Watch the area shrink.", 0)
    if stage == "fit2":
        caption(sl, tl, "Best line = smallest total area. That's least squares.", 200)
        pause_card(sl, tl, "Why square the misses instead of just adding up their lengths?", x=1.75, y=1.3)


# ============================================================================= CH2 gradient descent
AXL = Ax(6.75, 1.25, 5.9, 4.45, (-1.0, 3.3), (0, 9.0))
_wg = np.linspace(-3, 5, 4001)
_ok = _wg[np.array([D.L1(w) for w in _wg]) <= 8.9]
W_LO, W_HI = float(_ok.min()), float(_ok.max())
WCURVE = np.linspace(W_LO, W_HI, 80)


def loss_curve_pts(ax, ws=WCURVE, fn=D.L1):
    return [ax.P(w, min(fn(w), ax.yl[1])) for w in ws]


def tangent(sl, ax, w, length=1.7, name="!!tan"):
    g = D.dL1(w)
    sx = ax.w / (ax.xl[1] - ax.xl[0])
    sy = ax.h / (ax.yl[1] - ax.yl[0])
    vx, vy = sx, -g * sy
    n = np.hypot(vx, vy)
    vx, vy = vx / n * length / 2, vy / n * length / 2
    cx, cy = ax.P(w, D.L1(w))
    return seg(sl, name, cx - vx, cy - vy, cx + vx, cy + vy, RED, 2.5)


def ch2_split(stage, step=0):
    notes = {
        "split": """Here's the trick that powers everything. Freeze b for a moment and only turn the slope knob w. Every possible line gives one loss value. Plot them all and you get this curve. Our bad line is this point up here. The best line is the bottom of the valley.""",
        "slope": """But the computer can't see the whole curve. Computing it everywhere is way too expensive once you have millions of knobs. Think of a hiker in thick fog: you can't see the valley, but you can feel which way the ground tilts under your feet. That tilt is the slope, the derivative of the loss. Here it's strongly negative: going right goes down.""",
        "rule": """So here's the entire learning rule. New w equals old w minus a step size times the slope. Negative slope? w goes up. Positive slope? w goes down. You always step downhill. The step size, eta, is called the learning rate.""",
        "step": """Step. [auto-plays: each step recomputes the slope and moves against it.]""",
        "steplast": """Notice the steps get smaller on their own: as the ground flattens, the slope shrinks, so the step shrinks. Four steps and we're almost at the bottom.""",
    }
    key = stage if stage != "step" else ("steplast" if step == len(D.CH2_WS) - 1 else "step")
    auto = stage == "step" and step < len(D.CH2_WS) - 1
    sl, tl = new(2, notes[key], morph=1300 if stage == "split" else 900, advance=0 if auto else None)
    w = D.CH2_WS[step]
    axes(sl, AX1_MINI, "feed rate", "titre", label_size=13, xl_w=2.5)
    scatter_lin(sl, AX1_MINI, r=0.065)
    poly(sl, "!!line", line_pts(AX1_MINI, w, D.B_STAR), color=YELLOW, width=3)
    axes(sl, AXL, "slope of the line,  w", "loss", prefix="!!axl")
    cv = poly(sl, "!!lcurve", loss_curve_pts(AXL), color=INK, width=2.5)
    bx, by = AXL.P(w, D.L1(w))
    b_ = ball(sl, bx, by)
    if stage == "split":
        tl.add(cv, "wipe_l", 600, 900)
        caption(sl, tl, "Every possible line is one point on this curve.", 1300)
        return
    if stage == "rule":
        ov = rect(sl, "ovl", 0, 0, W, H, fill=BG, alpha=0.9)
        parts = [("w", "new w", YELLOW), ("=", "", INK), ("w", "old w", INK), ("−", "", INK),
                 ("η", "step size", TEAL), ("×", "", INK), ("slope", "feel the ground", RED)]
        x = 2.15
        widths = [0.9, 0.7, 0.9, 0.7, 0.9, 0.7, 2.2]
        tl.add(ov, "fade", 0, 400)
        for i, ((sym, lab, col), wd) in enumerate(zip(parts, widths)):
            s = text(sl, f"rule{i}", x, 2.75, wd + 0.6, 1.2, sym, size=66, font=MATH, italic=sym in ("w", "η", "slope"),
                     color=col, align="c")
            tl.add(s, "pop", 450 + i * 260, 420)
            if lab:
                l = text(sl, f"rulel{i}", x - 0.5, 4.05, wd + 1.6, 0.4, lab, size=16, color=MUTED, align="c")
                tl.add(l, "fade", 700 + i * 260, 400)
            x += wd + 0.25
        caption(sl, tl, "The whole learning rule. Always step downhill.", 2600)
        return
    tangent(sl, AXL, w)
    g = D.dL1(w)
    sl_lab = text(sl, "!!slope_v", 9.9, 0.55, 3.0, 0.5, [("slope = ", {}), (fmt(g, 1), {"bold": True})], size=22,
                  color=RED, align="r")
    if stage == "slope":
        tl.add(sl_lab, "fade", 600, 400)
        caption(sl, tl, "In the fog you can't see the valley. You can feel the slope.", 900)
        return
    # stepping
    for k in range(step):
        tx, ty = AXL.P(D.CH2_WS[k], D.L1(D.CH2_WS[k]))
        dot(sl, f"!!trail{k}", tx, ty, 0.07, YELLOW, alpha=0.45)
    st = text(sl, "!!stepn", 6.95, 0.55, 3.0, 0.5, f"step {step}", size=22, bold=True, font=HEAD, color=YELLOW)
    if step == 1:
        caption(sl, tl, "Step, recompute the slope, step again.", 100)
    if step == len(D.CH2_WS) - 1:
        caption(sl, tl, "Flatter ground, smaller slope, smaller steps. Automatically.", 100)


def lr_trio():
    sl, tl = new(2, f"""The learning rate is the one dial you'll touch more than any other. Too small, eta {D.LR_SMALL}: correct direction, but you crawl. After seven steps we've barely moved. Just right, eta {D.LR_GOOD}: down in a handful of steps. Too big, eta {D.LR_BIG}: each step overshoots the valley and lands higher on the other side. The loss goes UP, and it blows up. If your training loss ever explodes to NaN, this is the first suspect.""",
                 morph=1300)
    panels = [("small", "too small", f"η = {D.LR_SMALL}", MUTED), ("good", "just right", f"η = {D.LR_GOOD}", TEAL),
              ("big", "too big", f"η = {D.LR_BIG}", RED)]
    t_all = 0
    for i, (key, name, eta, col) in enumerate(panels):
        ax = Ax(0.85 + i * 4.2, 1.6, 3.5, 3.6, (-1.0, 3.3), (0, 9.0))
        pre = "!!axl" if key == "good" else f"axp{i}"
        seg(sl, f"{pre}_x", ax.x0, ax.y0 + ax.h, ax.x0 + ax.w, ax.y0 + ax.h, FAINT, 1.25)
        cname = "!!lcurve" if key == "good" else f"lc{i}"
        c = poly(sl, cname, loss_curve_pts(ax), color=INK, width=2)
        t1 = text(sl, f"lrn{i}", ax.x0, 0.8, ax.w, 0.45, name, size=26, bold=True, font=HEAD, color=col)
        t2 = text(sl, f"lre{i}", ax.x0, 5.35, ax.w, 0.4, eta, size=20, font=MATH, color=MUTED, italic=True)
        if key != "good":
            tl.add(c, "fade", 300, 500)
        tl.add(t1, "fade", 300 + i * 150, 400).add(t2, "fade", 400 + i * 150, 400)
        ws = D.LR_PATHS[key]
        P = [ax.P(np.clip(w, W_LO, W_HI), min(D.L1(np.clip(w, W_LO, W_HI)), 8.9)) for w in ws]
        start = 900 + i * 300
        for k in range(len(P) - 1):
            s = seg(sl, f"lz{i}_{k}", P[k][0], P[k][1], P[k + 1][0], P[k + 1][1], col if key != "small" else YELLOW, 1.5,
                    alpha=0.7)
            tl.add(s, "wipe_l" if P[k + 1][0] > P[k][0] else "wipe_r", start + k * 280 + 120, 200)
        for k, (px, py) in enumerate(P):
            if key == "good" and k == len(P) - 1:
                d_ = ball(sl, px, py, 0.13)
                continue
            d_ = dot(sl, f"lp{i}_{k}", px, py, 0.075, YELLOW)
            tl.add(d_, "pop", start + k * 280, 250)
    caption(sl, tl, "The learning rate is the dial you will touch most.", 3500)


def contour_image():
    path = os.path.join(ASSETS, "landscape.png")
    wl, bl = (-0.8, 2.6), (-1.0, 3.8)
    ww, bb = np.meshgrid(np.linspace(*wl, 400), np.linspace(*bl, 400))
    L = np.mean((ww[..., None] * D.lin_x + bb[..., None] - D.lin_y) ** 2, -1)
    fig = plt.figure(figsize=(6.4, 5.3), dpi=200)
    ax = fig.add_axes([0, 0, 1, 1])
    fig.patch.set_facecolor("#" + BG)
    ax.set_facecolor("#" + BG)
    cmap = LinearSegmentedColormap.from_list("l", ["#1d3b4f", "#14202b", "#" + BG])
    lv = np.logspace(np.log10(L.min() * 1.05), np.log10(L.max()), 22)
    ax.contourf(ww, bb, np.log(L), levels=40, cmap=LinearSegmentedColormap.from_list("f", ["#1b3a4a", "#0E1116"]))
    ax.contour(ww, bb, L, levels=lv, colors="#5d6878", linewidths=0.8, alpha=0.8)
    ax.plot([D.W_STAR], [D.B_STAR], marker="x", color="#" + TEAL, ms=10, mew=2.5)
    ax.set_xlim(*wl)
    ax.set_ylim(*bl)
    ax.axis("off")
    fig.savefig(path, facecolor=fig.get_facecolor())
    plt.close(fig)
    return path, wl, bl


def ch2_landscape(stage):
    path, wl, bl = contour_image()
    ax = Ax(0.85, 0.95, 5.9, 4.9, wl, bl)
    notes = {
        "full": """Of course a real model has more than one knob. With two knobs, w and b, the curve becomes a landscape. These rings are contour lines, like on a hiking map; the green cross is the valley floor. The slope becomes the gradient: the slope in every direction at once, packed into one vector. The rule doesn't change: step against the gradient. Watch the path. It curves, because the valley is long and narrow.""",
        "sgd": """One last practical fix. Computing the exact gradient needs every batch in the data set, every step. With ten million examples that's painfully slow. So we cheat: estimate the gradient from a small random handful, a mini-batch, here just four batches. Each step is noisier, the yellow path wobbles, but each step is far cheaper, and it still gets to the valley. This is stochastic gradient descent, and it's how essentially every neural network is trained.""",
    }[stage]
    sl, tl = new(2, notes, morph=1200)
    im = picture(sl, "!!landscape", path, ax.x0, ax.y0, ax.w, ax.h)
    text(sl, "!!lsx", ax.x0, ax.y0 + ax.h + 0.08, ax.w, 0.3, "slope  w", size=14, color=MUTED, align="r")
    text(sl, "!!lsy", ax.x0 + 0.1, ax.y0 + 0.08, 3, 0.3, "height  b", size=14, color=MUTED)
    idx = list(range(0, 12)) + list(range(12, 121, 6))
    full = []
    for j, k in enumerate(idx):
        w, b = D.C_FULL[k]
        full.append(dot(sl, f"!!cf{j}", ax.X(w), ax.Y(b), 0.06 if stage == "full" else 0.045, INK,
                        alpha=1.0 if stage == "full" else 0.35))
    g1 = text(sl, "!!grad_eq", 7.9, 1.55, 5.0, 0.8,
              [("∇", {}), ("L", {"italic": True}), (" = (", {}), ("∂L/∂w", {"italic": True}), (",  ", {}),
               ("∂L/∂b", {"italic": True}), (")", {})], size=30, font=MATH, color=INK)
    g2 = text(sl, "!!grad_l", 7.9, 2.4, 4.8, 0.8, "the gradient: the slope in every direction at once", size=18,
              color=MUTED)
    if stage == "full":
        tl.add(im, "fade", 0, 600).add(g1, "fade", 600, 500).add(g2, "fade", 800, 500)
        tl.stagger(full, "pop", 1300, 70, 250)
        caption(sl, tl, "Two knobs: the curve becomes a landscape. Same rule, step against the gradient.", 1000)
        return
    sg = []
    for j, k in enumerate(range(0, 241, 3)):
        w, b = D.C_SGD[k]
        sg.append(dot(sl, f"sg{j}", ax.X(w), ax.Y(b), 0.05, YELLOW))
    k1 = text(sl, "mb_k", 7.9, 3.55, 4.8, 0.35, "GRADIENT ESTIMATED FROM", size=12, bold=True, font=HEAD, color=YELLOW, spacing=2)
    k2 = text(sl, "mb_v", 7.85, 3.85, 5.0, 1.2, [("4", {"color": YELLOW}), (f" of {D.N_LIN} batches", {"size": 26, "bold": False,
                                                                                                        "color": MUTED})],
              size=72, bold=True, font=HEAD, color=YELLOW)
    tl.add(k1, "fade", 200, 400).add(k2, "pop", 300, 450)
    tl.stagger(sg, "pop", 900, 35, 200)
    caption(sl, tl, "Mini-batches: noisy steps, far cheaper, still reaches the valley.", 600)


# ============================================================================= CH3 logistic regression
AXG = Ax(1.25, 1.15, 7.6, 4.6, (0, 6.0), (-0.45, 1.45))
AXG_MINI = Ax(0.9, 1.55, 5.2, 3.4, (0, 6.0), (-0.45, 1.45))
GX = np.linspace(0.05, 5.95, 60)


def log_scene(stage):
    notes = {
        "line": """New question, new kind of answer. Not "how much titre?" but "will this batch fail spec, yes or no?" Each dot is a past batch: lactate at 48 hours on the x-axis, and it either passed, down at zero, or failed, up at one. What if we just fit our straight line? At high lactate it predicts a 130 percent chance of failure. At low lactate, a negative chance. Nonsense.""",
        "sig": """Fix: keep the line, but squash it through an S-shaped function, the sigmoid. Huge negative inputs go to 0, huge positive inputs go to 1. Now every output is a probability. This is logistic regression, and it's exactly one neuron, remember that for later.""",
        "wrong": f"""How do we score a probability? With surprise. This batch failed, but the model only gave it a {D.log_p[D.CE_WRONG] * 100:.0f} percent chance. The loss is minus log of the probability it gave to what actually happened. Minus log of {D.log_p[D.CE_WRONG]:.2f} is {-np.log(D.log_p[D.CE_WRONG]):.1f}. Notice the shape: as that probability heads toward zero, the surprise shoots to infinity.""",
        "right": f"""This one also failed, and the model gave it {D.log_p[D.CE_RIGHT] * 100:.1f} percent. Barely surprised: loss {-np.log(D.log_p[D.CE_RIGHT]):.3f}. Confident and right costs almost nothing; confident and wrong costs a fortune. Average the surprise over all batches and you have cross-entropy, the loss behind every classifier including LLMs.""",
    }[stage]
    sl, tl = new(3, notes, morph=1200)
    ax = AXG if stage in ("line", "sig") else AXG_MINI
    big = ax is AXG
    seg(sl, "!!g0", ax.x0, ax.Y(0), ax.x0 + ax.w, ax.Y(0), FAINT, 1.0, dash="dash")
    seg(sl, "!!g1", ax.x0, ax.Y(1), ax.x0 + ax.w, ax.Y(1), FAINT, 1.0, dash="dash")
    text(sl, "!!g0l", ax.x0 - 0.75, ax.Y(0) - 0.15, 0.65, 0.3, "pass", size=14, color=BLUE, align="r")
    text(sl, "!!g1l", ax.x0 - 0.75, ax.Y(1) - 0.15, 0.65, 0.3, "fail", size=14, color=ORANGE, align="r")
    seg(sl, "!!lgx", ax.x0, ax.y0 + ax.h, ax.x0 + ax.w, ax.y0 + ax.h, FAINT, 1.25)
    text(sl, "!!lgxl", ax.x0 + ax.w - 4, ax.y0 + ax.h + 0.08, 4, 0.3, "lactate at 48 h (g/L)", size=14, color=MUTED, align="r")
    pts = []
    for i, (x, y) in enumerate(zip(D.log_x, D.log_y)):
        pts.append(dot(sl, f"!!gp{i}", ax.X(x), ax.Y(y), 0.08 if big else 0.065, ORANGE if y else BLUE))
    if stage == "line":
        ys = D.LS_W * GX + D.LS_B
    else:
        ys = D.sigmoid(D.LG_W * GX + D.LG_B)
    fx = poly(sl, "!!fx", [ax.P(x, y) for x, y in zip(GX, ys)], color=YELLOW, width=3.5)
    if stage == "line":
        tl.stagger(pts, "pop", 200, 45, 300)
        tl.add(fx, "wipe_l", 1500, 800)
        x_end = GX[-1]
        bad = text(sl, "bad130", ax.X(x_end) - 1.2, ax.Y(D.LS_W * x_end + D.LS_B) - 0.55, 1.3, 0.4,
                   f"{(D.LS_W * x_end + D.LS_B) * 100:.0f}%?", size=22, bold=True, font=HEAD, color=RED, align="r")
        bad2 = text(sl, "badneg", ax.X(0.1), ax.Y(D.LS_B) + 0.12, 1.4, 0.4, f"{D.LS_B * 100:.0f}%?".replace("-", "−"),
                    size=22, bold=True, font=HEAD, color=RED)
        tl.add(bad, "pop", 2500, 400).add(bad2, "pop", 2700, 400)
        caption(sl, tl, f"A straight line happily predicts a {(D.LS_W * x_end + D.LS_B) * 100:.0f}% chance of failure.", 2900)
        return
    if stage == "sig":
        f1 = text(sl, "!!sigeq", 9.3, 1.5, 3.9, 0.7, [("p", {"italic": True}), (" = σ(", {}), ("w", {"italic": True}),
                                                 ("·x + ", {}), ("b", {"italic": True}), (")", {})], size=32, font=MATH, color=INK)
        f2 = text(sl, "sigeq2", 9.3, 2.3, 3.9, 0.5, "σ squashes any number into 0 to 1", size=17, color=MUTED)
        tl.add(f1, "fade", 900, 500).add(f2, "fade", 1100, 500)
        caption(sl, tl, "Squash the line into an S. Every output is now a probability.", 1200)
        return
    # cross-entropy
    i = D.CE_WRONG if stage == "wrong" else D.CE_RIGHT
    p = D.log_p[i]
    px, py = ax.X(D.log_x[i]), ax.Y(1)
    dot(sl, "!!ring", px, py, 0.2, None, line=INK, line_w=2)
    cy = ax.Y(p)
    seg(sl, "!!pguide", px, py + 0.2, px, cy, INK, 1.25, dash="sysDot")
    dot(sl, "!!pmark", px, cy, 0.07, YELLOW)
    said = text(sl, "!!said", ax.x0 + 0.1, 0.65, 5.5, 0.45,
                [("Failed. Model said: ", {"color": MUTED}), (f"{p * 100:.0f}%" if p < 0.99 else f"{p * 100:.1f}%", {"bold": True, "color": YELLOW})],
                size=22)
    axs = Ax(7.3, 1.25, 5.2, 4.3, (0, 1.0), (0, 4.2))
    seg(sl, "!!sx", axs.x0, axs.y0 + axs.h, axs.x0 + axs.w, axs.y0 + axs.h, FAINT, 1.25)
    seg(sl, "!!sy", axs.x0, axs.y0, axs.x0, axs.y0 + axs.h, FAINT, 1.25)
    text(sl, "!!sxl", axs.x0 + axs.w - 4.8, axs.y0 + axs.h + 0.3, 4.5, 0.3, "probability given to what happened", size=14, color=MUTED, align="r")
    text(sl, "!!syl", axs.x0 + 0.12, axs.y0 - 0.05, 4, 0.3, "surprise = −log p", size=14, color=MUTED)
    ps = np.linspace(np.exp(-4.2), 1.0, 70)
    sc = poly(sl, "!!scurve", [axs.P(q, -np.log(q)) for q in ps], color=RED, width=3)
    bx, by = axs.P(p, -np.log(p))
    ball(sl, bx, by)
    sv = text(sl, "!!surp", 9.4, 2.0, 3.3, 1.2, fmt(-np.log(p), 1 if p < 0.99 else 3), size=60, bold=True, font=HEAD,
              color=RED if stage == "wrong" else TEAL, align="r")
    if stage == "wrong":
        tl.add(sc, "wipe_l", 300, 800)
        caption(sl, tl, "Loss for yes/no = surprise. Unlikely outcome, big surprise.", 900)
    else:
        caption(sl, tl, "Confident and right costs almost nothing. That's cross-entropy.", 300)


def loss_origins():
    sl, tl = new(3, """So where do these losses come from? They aren't arbitrary. Both answer the same question: if the model were true, how likely is the data we actually saw? Make that as likely as possible: maximum likelihood. If you assume the misses are bell-shaped noise around the line, maximising the likelihood is exactly minimising squared error. If you assume each outcome is a coin flip with the model's probability, it's exactly minimising cross-entropy. Pick a story for the noise, and the loss falls out.""",
                 morph=1100)
    h = text(sl, "lo_h", 0.75, 0.85, 11.8, 0.8, "Neither loss was invented.", size=36,
             bold=True, font=HEAD, color=INK)
    q = text(sl, "lo_q", 0.75, 1.65, 11.8, 0.5, "How likely is the data we actually saw?  Make it as likely as possible.", size=20,
             color=YELLOW, italic=True, font=MATH)
    tl.add(h, "rise", 100, 500).add(q, "fade", 600, 500)
    # left: bell-shaped noise
    bx0, by0 = 1.3, 2.7
    xs = np.linspace(-3, 3, 60)
    bell = [(bx0 + 2.5 + x * 0.75, by0 + 2.4 - 2.0 * np.exp(-x * x / 2)) for x in xs]
    b = poly(sl, "lo_bell", bell + [(bell[-1][0], by0 + 2.4), (bell[0][0], by0 + 2.4)], color=ORANGE, width=2.5,
             fill=ORANGE, alpha=0.18, closed=True)
    l1 = text(sl, "lo_l1", bx0, by0 + 2.6, 5.4, 0.4, "misses are bell-shaped noise", size=20, color=MUTED, align="c")
    l2 = text(sl, "lo_l2", bx0, by0 + 3.05, 5.4, 0.6, [("(", {}), ("y", {"italic": True}), (" − ", {}), ("ŷ", {"italic": True}),
                                                       (")²", {})], size=34, font=MATH, color=ORANGE, align="c")
    l3 = text(sl, "lo_l3", bx0, by0 + 3.65, 5.4, 0.4, "squared error", size=18, bold=True, color=INK, align="c")
    # right: coin flip
    rx0 = 7.0
    bars = [(0.3, BLUE, "pass"), (0.7, ORANGE, "fail")]
    shapes = []
    for k, (pv, col, lab) in enumerate(bars):
        hgt = 2.0 * pv
        shapes.append(rect(sl, f"lo_bar{k}", rx0 + 1.55 + k * 1.3, by0 + 2.4 - hgt, 0.9, hgt, fill=col, alpha=0.85, radius=0.06))
        shapes.append(text(sl, f"lo_bl{k}", rx0 + 1.35 + k * 1.3, by0 + 2.4 - hgt - 0.4, 1.3, 0.3, f"{lab}  {pv:.1f}",
                           size=14, color=MUTED, align="c"))
    r1 = text(sl, "lo_r1", rx0, by0 + 2.6, 5.4, 0.4, "each outcome is a weighted coin flip", size=20, color=MUTED, align="c")
    r2 = text(sl, "lo_r2", rx0, by0 + 3.05, 5.4, 0.6, [("−log ", {}), ("p", {"italic": True})], size=34, font=MATH,
              color=RED, align="c")
    r3 = text(sl, "lo_r3", rx0, by0 + 3.65, 5.4, 0.4, "cross-entropy", size=18, bold=True, color=INK, align="c")
    tl.add(b, "wipe_u", 1200, 600).add(l1, "fade", 1500, 400).add(l2, "pop", 1900, 450).add(l3, "fade", 2200, 400)
    tl.stagger(shapes[::2], "wipe_u", 2700, 200, 400)
    tl.stagger(shapes[1::2], "fade", 2900, 200, 300)
    tl.add(r1, "fade", 3100, 400).add(r2, "pop", 3500, 450).add(r3, "fade", 3800, 400)


# ============================================================================= CH4 neural networks
FIELD_BOX = (0.9, 0.95, 5.9, 4.95)
T_LIM, PH_LIM = (29.6, 38.4), (6.56, 7.44)


def field_image(name, prob_fn, title=None):
    path = os.path.join(ASSETS, f"{name}.png")
    TT, PP = np.meshgrid(np.linspace(*T_LIM, 300), np.linspace(*PH_LIM, 300))
    Z = prob_fn(D.nn_inputs(TT.ravel(), PP.ravel())).reshape(TT.shape)
    fig = plt.figure(figsize=(FIELD_BOX[2], FIELD_BOX[3]), dpi=170)
    ax = fig.add_axes([0, 0, 1, 1])
    fig.patch.set_facecolor("#" + BG)
    cmap = LinearSegmentedColormap.from_list("pf", ["#16324a", "#1a1f28", "#4a3020"])
    ax.imshow(Z, extent=[*T_LIM, *PH_LIM], origin="lower", aspect="auto", cmap=cmap, vmin=0, vmax=1)
    try:
        ax.contour(TT, PP, Z, levels=[0.5], colors="#" + YELLOW, linewidths=2.5)
    except Exception:
        pass
    ax.scatter(D.nn_T[D.nn_y == 0], D.nn_pH[D.nn_y == 0], s=22, c="#" + BLUE, edgecolors="none")
    ax.scatter(D.nn_T[D.nn_y == 1], D.nn_pH[D.nn_y == 1], s=22, c="#" + ORANGE, edgecolors="none")
    ax.set_xlim(*T_LIM)
    ax.set_ylim(*PH_LIM)
    ax.axis("off")
    fig.savefig(path, facecolor=fig.get_facecolor())
    plt.close(fig)
    return path


def field(sl, path, tl=None, at=0):
    x, y, w, h = FIELD_BOX
    im = picture(sl, "!!field", path, x, y, w, h)
    text(sl, "!!fx_l", x, y + h + 0.08, w, 0.3, "temperature (°C)", size=14, color=MUTED, align="r")
    text(sl, "!!fy_l", x - 0.05, y - 0.38, 3, 0.3, "pH", size=14, color=MUTED)
    if tl:
        tl.add(im, "fade", at, 600)
    return im


def nn_window():
    path = field_image("field_logistic", lambda X: D.sigmoid(X @ D.LOG2_W + D.LOG2_B))
    sl, tl = new(4, f"""Back to the operating window from Lecture 1. Good batches, orange, sit inside a window of temperature and pH; outside it, they fail. Let's try logistic regression: one straight boundary. Gradient descent does its best, and its best is a line that gives up and calls almost everything bad. {D.LOG2_ACC * 100:.0f} percent accuracy sounds fine until you notice that's what you'd get by always saying "bad". One line can't draw a window.""",
                 morph=1100)
    field(sl, path, tl, 100)
    a = text(sl, "win_h", 7.6, 1.6, 5.2, 1.6, "One straight line can't draw a window.", size=36, bold=True, font=HEAD, color=INK)
    k = text(sl, "win_k", 7.6, 3.55, 5, 0.35, "LOGISTIC REGRESSION ACCURACY", size=12, bold=True, font=HEAD, color=MUTED, spacing=2)
    v = text(sl, "win_v", 7.55, 3.85, 5, 1.2, f"{D.LOG2_ACC * 100:.0f}%", size=72, bold=True, font=HEAD, color=RED)
    s = text(sl, "win_s", 7.6, 5.05, 5, 0.8, "its best straight boundary is no boundary: it calls every batch “bad”", size=18, color=MUTED, italic=True)
    tl.add(a, "rise", 600, 500).add(k, "fade", 1100, 400).add(v, "pop", 1200, 450).add(s, "fade", 1600, 400)


def relu_glyph(sl, name, cx, cy, s=0.32, color=INK):
    return poly(sl, name, [(cx - s, cy + s * 0.45), (cx, cy + s * 0.45), (cx + s, cy - s * 0.55)], color=color, width=2.25)


def neuron_scene(stage):
    notes = {
        "one": """Let's look inside logistic regression and call it what it is: a neuron. It takes the inputs, temperature and pH, multiplies each by a weight, adds them up with a bias, then bends the result. The bend here is a ReLU, which just zeroes out anything negative; earlier we used the sigmoid S-curve. One neuron draws one straight boundary.""",
        "net": """So stack them. A layer of hidden neurons, each one drawing its own bent line through the data, and an output neuron that blends them. Every line in this picture is one weight, one knob. Same three questions: the model is this whole network, the loss is cross-entropy, and learning is gradient descent on every weight at once.""",
    }[stage]
    sl, tl = new(4, notes, morph=1300)
    in_xy = [(2.2, 2.55), (2.2, 4.55)] if stage == "one" else [(2.0, 2.45), (2.0, 4.65)]
    labels = ["T", "pH"]
    if stage == "one":
        hid = [(6.65, 3.55)]
        r_h = 0.85
    else:
        hid = [(6.4, 1.35 + k * 1.1) for k in range(5)]
        r_h = 0.4
    out = (10.9, 3.55) if stage == "one" else (10.4, 3.55)
    edges = []
    for a, (ix, iy) in enumerate(in_xy):
        for k, (hx, hy) in enumerate(hid):
            name = f"!!e_i{a}_h{k + 2 if stage == 'one' else k}"
            edges.append(seg(sl, name, ix + 0.42, iy, hx - r_h, hy, MUTED, 1.75, alpha=0.7))
    for k, (hx, hy) in enumerate(hid):
        name = f"!!e_h{k + 2 if stage == 'one' else k}_o"
        edges.append(seg(sl, name, hx + r_h, hy, out[0] - 0.45, out[1], MUTED, 1.75, alpha=0.7))
    ins = []
    for a, ((ix, iy), lab) in enumerate(zip(in_xy, labels)):
        ins.append(dot(sl, f"!!in{a}", ix, iy, 0.42, PANEL, line=BLUE, line_w=2.5))
        ins.append(text(sl, f"!!inl{a}", ix - 0.42, iy - 0.25, 0.84, 0.5, lab, size=22, font=MATH, italic=True, color=INK,
                        align="c", anchor="m"))
    hs = []
    for k, (hx, hy) in enumerate(hid):
        nm = k + 2 if stage == "one" else k
        hs.append(dot(sl, f"!!h{nm}", hx, hy, r_h, PANEL, line=YELLOW, line_w=2.5))
        if stage == "one":
            text(sl, f"!!hs{nm}", hx - 0.55, hy - 0.42, 0.6, 0.6, "Σ", size=30, font=MATH, color=INK, align="c", anchor="m")
            relu_glyph(sl, f"!!hg{nm}", hx + 0.28, hy + 0.0, 0.28, YELLOW)
        else:
            relu_glyph(sl, f"!!hg{nm}", hx, hy, 0.18, YELLOW)
    o = dot(sl, "!!out", out[0], out[1], 0.45, PANEL, line=ORANGE, line_w=2.5)
    text(sl, "!!outl", out[0] - 0.45, out[1] - 0.25, 0.9, 0.5, "p", size=24, font=MATH, italic=True, color=INK, align="c", anchor="m")
    if stage == "one":
        tl.stagger(ins[::2], "pop", 100, 150, 350)
        tl.stagger(edges[:2], "wipe_l", 500, 150, 400)
        tl.add(hs[0], "pop", 900, 450)
        tl.add(edges[2], "wipe_l", 1300, 400).add(o, "pop", 1600, 400)
        lab = text(sl, "neur_lab", 5.0, 5.0, 3.3, 0.9, "weigh  ·  add  ·  bend", size=20, color=YELLOW, align="c", font=MATH, italic=True)
        tl.add(lab, "fade", 1900, 500)
        caption(sl, tl, "A neuron: weigh the inputs, add them up, bend. That's logistic regression.", 2200)
    else:
        lab1 = text(sl, "nl_in", 1.2, 5.6, 1.6, 0.4, "inputs", size=16, color=MUTED, align="c")
        lab2 = text(sl, "nl_h", 5.4, 0.5, 2.0, 0.4, "hidden layer", size=16, color=MUTED, align="c")
        lab3 = text(sl, "nl_o", 9.6, 4.25, 1.6, 0.4, "output", size=16, color=MUTED, align="c")
        tl.add(lab1, "fade", 900, 400).add(lab2, "fade", 1000, 400).add(lab3, "fade", 1100, 400)
        caption(sl, tl, "Stack them. Every line is one weight, one knob to turn.", 1300)


def nonlin_scene(stage):
    notes = {
        "parts": """Why insist on the bend? On the left, three straight lines. On the right, three bent ones, ReLUs, each flat and then rising from a different corner. Now let's do what a network does: add them up.""",
        "sum": """Add up straight lines and you get... another straight line. Stack a hundred linear layers without a bend and the whole network collapses into a single line; all that depth buys nothing. Add up bends and you get corners, and with enough corners you can draw any shape you like, including a window. The nonlinearity is the whole point.""",
    }[stage]
    sl, tl = new(4, notes, morph=1500)
    xs = np.linspace(0, 6, 60)
    lines = [0.5 * xs + 0.2, -0.3 * xs + 1.6, 0.15 * xs + 0.4]
    hinges = [np.maximum(xs - 0.8, 0) * 0.8, np.maximum(xs - 2.4, 0) * -1.3, np.maximum(xs - 4.0, 0) * 1.4]
    YL = (-5.0, 4.6)
    axL = Ax(0.9, 1.55, 5.3, 3.8, (0, 6), YL)
    axR = Ax(7.1, 1.55, 5.3, 3.8, (0, 6), YL)
    cols = [BLUE, TEAL, ORANGE]
    for side, (ax, curves, title) in enumerate([(axL, lines, "without the bend"), (axR, hinges, "with the bend")]):
        seg(sl, f"nl_ax{side}", ax.x0, ax.Y(0), ax.x0 + ax.w, ax.Y(0), FAINT, 1.25)
        t = text(sl, f"nl_t{side}", ax.x0, 0.85, ax.w, 0.5, title, size=26, bold=True, font=HEAD, color=INK)
        total = np.sum(curves, 0)
        for k, c in enumerate(curves):
            y = total if stage == "sum" else c
            col = (INK if side == 0 else YELLOW) if stage == "sum" else cols[k]
            p = poly(sl, f"!!nl{side}_{k}", [ax.P(x, np.clip(v, *YL)) for x, v in zip(xs, y)], color=col,
                     width=3.5 if stage == "sum" else 2.5)
            if stage == "parts":
                tl.add(p, "wipe_l", 400 + side * 900 + k * 250, 500)
        if stage == "sum":
            res = ["still just a line", "corners: any shape you like"][side]
            r = text(sl, f"nl_r{side}", ax.x0, 5.5, ax.w, 0.5, res, size=22, italic=True, font=MATH,
                     color=MUTED if side == 0 else YELLOW)
            tl.add(r, "fade", 300 + side * 300, 500)
    if stage == "parts":
        caption(sl, tl, "Three straight lines. Three bent ones. Now add them up.", 2400)
    else:
        caption(sl, tl, "Lines stacked on lines stay a line. Bends stacked on bends draw anything.", 900)


# ============================================================================= CH5 backprop
NODES = {"x": (1.55, 3.0), "h": (4.95, 3.0), "yh": (8.35, 3.0), "L": (11.75, 3.0)}
R_NODE = 0.55


def bp_scene(stage):
    B = D.BP
    notes = {
        "fwd": f"""Now the big question: with thousands of weights, how do we get the slope for each one? Let's do it by hand on the smallest possible network. One input x equals 2. Weight w1 equals 0.5, then a ReLU, giving h. Weight w2 equals 1.5 gives the prediction y-hat. The target is 3, and the loss is squared error. Forward pass: 0.5 times 2 is 1; ReLU of 1 is 1; 1.5 times 1 is 1.5; and (1.5 minus 3) squared is 2.25.""",
        "bwd": f"""Backward pass. Start at the loss and ask: if y-hat wiggles, how much does the loss wiggle? Two times (1.5 minus 3) is minus 3. Step back one link: y-hat is w2 times h, so the slope for w2 is minus 3 times h, which is minus 3. And the slope passed back to h is minus 3 times w2, minus 4.5. Through the ReLU, which was on, so it passes the slope unchanged, times 1. Finally w1: minus 4.5 times x, minus 9. Each step just multiplies by its local slope. That's the chain rule, and doing it right to left, reusing each product, is backpropagation.""",
        "upd": f"""Now take one gradient-descent step with learning rate 0.05. w1 goes from 0.5 to 0.5 minus 0.05 times minus 9, which is 0.95. w2 goes from 1.5 to 1.65. Run forward again: h is 1.9, y-hat is 3.135, and the loss drops from 2.25 to 0.018. A real network does exactly this, with millions of weights, millions of times.""",
    }[stage]
    sl, tl = new(5, notes, morph=1300)
    w1 = B["w1n"] if stage == "upd" else B["w1"]
    w2 = B["w2n"] if stage == "upd" else B["w2"]
    h = B["hn"] if stage == "upd" else B["h"]
    yh = B["yhatn"] if stage == "upd" else B["yhat"]
    L = B["Ln"] if stage == "upd" else B["L"]
    keys = ["x", "h", "yh", "L"]
    edges = []
    for k in range(3):
        (ax_, ay), (bx, by) = NODES[keys[k]], NODES[keys[k + 1]]
        edges.append(seg(sl, f"!!be{k}", ax_ + R_NODE, ay, bx - R_NODE, by, FAINT, 3))
    nodes = []
    labels = {"x": "x", "h": "h", "yh": "ŷ", "L": "L"}
    for k in keys:
        cx, cy = NODES[k]
        nodes.append(dot(sl, f"!!bn{k}", cx, cy, R_NODE, PANEL, line=INK if k != "L" else RED, line_w=2.25))
        nodes.append(text(sl, f"!!bnl{k}", cx - R_NODE, cy - 0.3, 2 * R_NODE, 0.6, labels[k], size=28, font=MATH,
                          italic=True, color=INK, align="c", anchor="m"))
    # edge labels
    def mid(k):
        (ax_, ay), (bx, by) = NODES[keys[k]], NODES[keys[k + 1]]
        return (ax_ + bx) / 2
    wl = [text(sl, "!!bw1", mid(0) - 1.0, 2.2, 2.0, 0.4, [("w", {"italic": True}), ("1", {"sub": True}), (f" = {w1:g}", {})], size=20,
               font=MATH, color=TEAL if stage == "upd" else INK, align="c"),
          text(sl, "!!bw2", mid(1) - 1.0, 2.2, 2.0, 0.4, [("w", {"italic": True}), ("2", {"sub": True}), (f" = {w2:g}", {})], size=20,
               font=MATH, color=TEAL if stage == "upd" else INK, align="c"),
          text(sl, "!!bw3", mid(2) - 1.1, 2.2, 2.2, 0.4, [("(ŷ − 3)²", {})], size=20, font=MATH, color=MUTED, align="c")]
    tops = [text(sl, "!!bfx", NODES["x"][0] - 1, 1.45, 2, 0.5, "2", size=26, bold=True, font=HEAD, color=YELLOW, align="c"),
            text(sl, "!!bfh", NODES["h"][0] - 1, 1.45, 2, 0.5, f"{h:g}", size=26, bold=True, font=HEAD, color=YELLOW, align="c"),
            text(sl, "!!bfy", NODES["yh"][0] - 1, 1.45, 2, 0.5, f"{yh:g}", size=26, bold=True, font=HEAD, color=YELLOW, align="c"),
            text(sl, "!!bfL", NODES["L"][0] - 1, 1.45, 2, 0.5, f"{L:.3f}" if stage == "upd" else f"{L:g}", size=26, bold=True,
                 font=HEAD, color=TEAL if stage == "upd" else YELLOW, align="c")]
    text(sl, "!!brelu", NODES["h"][0] - 1, 3.65, 2, 0.35, "ReLU", size=14, color=MUTED, align="c")
    text(sl, "!!btarget", NODES["L"][0] - 1, 3.65, 2, 0.35, "target y = 3", size=14, color=MUTED, align="c")
    k1 = text(sl, "!!bk", 0.75, 0.5, 6, 0.35, {"fwd": "FORWARD PASS", "bwd": "BACKWARD PASS", "upd": "ONE STEP DOWNHILL"}[stage],
              size=12, bold=True, font=HEAD, color={"fwd": YELLOW, "bwd": RED, "upd": TEAL}[stage], spacing=2)
    if stage == "fwd":
        tl.stagger([n for n in nodes], "fade", 0, 40, 300)
        fw = []
        for k in range(3):
            (ax_, ay), (bx, by) = NODES[keys[k]], NODES[keys[k + 1]]
            fw.append(seg(sl, f"fw{k}", ax_ + R_NODE, ay, bx - R_NODE, by, YELLOW, 3))
        tl.add(tops[0], "pop", 600, 350)
        t = 900
        for k in range(3):
            tl.add(fw[k], "wipe_l", t, 500).add(tops[k + 1], "pop", t + 450, 350)
            t += 900
        caption(sl, tl, "Forward pass: push the numbers through. Loss = 2.25.", t)
        return
    if stage == "bwd":
        bw = []
        for k in range(3):
            (ax_, ay), (bx, by) = NODES[keys[k]], NODES[keys[k + 1]]
            bw.append(seg(sl, f"bw{k}", ax_ + R_NODE, ay + 0.0, bx - R_NODE, by + 0.0, RED, 3))
        g_y = text(sl, "g_y", NODES["yh"][0] - 1, 3.95, 2, 0.5, fmt(B["dyhat"], 0), size=26, bold=True, font=HEAD, color=RED, align="c")
        g_h = text(sl, "g_h", NODES["h"][0] - 1, 3.95, 2, 0.5, fmt(B["dh"], 1), size=26, bold=True, font=HEAD, color=RED, align="c")
        g_w2 = text(sl, "g_w2", mid(1) - 1.3, 4.4, 2.6, 0.45, [("∂L/∂w", {"italic": True}), ("2", {"sub": True}),
                                                                (f" = {fmt(B['dw2'], 0)}", {})], size=20, font=MATH, color=RED, align="c")
        g_w1 = text(sl, "g_w1", mid(0) - 1.3, 4.4, 2.6, 0.45, [("∂L/∂w", {"italic": True}), ("1", {"sub": True}),
                                                                (f" = {fmt(B['dw1'], 0)}", {})], size=20, font=MATH, color=RED, align="c")
        t = 200
        tl.add(bw[2], "wipe_r", t, 500).add(g_y, "pop", t + 450, 350)
        t += 1000
        tl.add(bw[1], "wipe_r", t, 500).add(g_w2, "fade", t + 450, 400).add(g_h, "pop", t + 650, 350)
        t += 1100
        tl.add(bw[0], "wipe_r", t, 500).add(g_w1, "pop", t + 450, 400)
        t += 1000
        # the chain written out
        parts = [(fmt(B["dyhat"], 0), "∂L/∂ŷ"), ("×", ""), ("1.5", "w₂"), ("×", ""), ("1", "ReLU on"),
                 ("×", ""), ("2", "x"), ("=", ""), (fmt(B["dw1"], 0), "∂L/∂w₁")]
        x = 2.55
        for i, (sym, lab) in enumerate(parts):
            wd = 0.55 if sym in ("×", "=") else 1.25
            s = text(sl, f"ch{i}", x, 5.15, wd, 0.6, sym, size=30, bold=sym not in ("×", "="), font=HEAD,
                     color=RED if i in (0, 8) else INK, align="c")
            tl.add(s, "fade", t + i * 140, 300)
            if lab:
                l = text(sl, f"chl{i}", x - 0.3, 5.72, wd + 0.6, 0.35, lab, size=14, color=MUTED, align="c")
                tl.add(l, "fade", t + i * 140 + 100, 300)
            x += wd + 0.12
        caption(sl, tl, "Backward pass: multiply local slopes, right to left. That's the chain rule.", t + 1500)
        return
    # update
    old = text(sl, "old_L", 9.1, 4.3, 3.6, 0.6, "loss  2.25", size=24, color=MUTED, align="r", font=HEAD)
    new_ = text(sl, "new_L", 8.3, 4.85, 4.4, 1.2, "0.018", size=72, bold=True, font=HEAD, color=TEAL, align="r")
    tl.add(old, "fade", 300, 400).add(new_, "pop", 700, 450)
    upd = text(sl, "upd_eq", 0.9, 4.45, 7.2, 1.2,
               [("w", {"italic": True}), ("1", {"sub": True}), (":  0.5 − 0.05 × (−9) = ", {}), ("0.95", {"color": TEAL, "bold": True}),
                ("\n", {}), ("w", {"italic": True}), ("2", {"sub": True}), (":  1.5 − 0.05 × (−3) = ", {}),
                ("1.65", {"color": TEAL, "bold": True})], size=22, font=MATH, color=INK)
    tl.add(upd, "fade", 200, 500)
    caption(sl, tl, "One step downhill: the loss falls from 2.25 to 0.018.", 1100)


# ============================================================================= CH6 training montage
LAX = Ax(8.05, 2.0, 4.6, 2.55, (0, np.log10(D.NN_EPOCHS + 1)), (np.log10(0.01), np.log10(1.0)))


def train_frame(k):
    ep = D.NN_SNAPS[k]
    P = D.nn_snaps[ep]
    path = field_image(f"field_ep{ep}", lambda X, P=P: D.mlp_forward(P, X)[2])
    last = k == len(D.NN_SNAPS) - 1
    notes = {0: f"""Now let's put everything together and press play. A network with 16 hidden ReLU neurons, randomly initialised. Its starting boundary is just noise. Every epoch: forward pass, cross-entropy loss, backpropagation, one gradient step. Same three steps, repeated. [auto-plays]""",
             len(D.NN_SNAPS) - 1: f"""After {ep} epochs the boundary has wrapped itself around the window, and the loss on the right fell by about a factor of forty. Nobody told the network what an ellipse is. It found the shape by rolling downhill on its own mistakes. [Click for the pause card.] Trap: 100 percent on training data proves nothing by itself; Lecture 1's rule still holds, judge it on batches it has never seen."""}
    sl, tl = new(6, notes.get(k, "[auto-plays]"), morph=900, advance=None if (k == 0 or last) else 700)
    field(sl, path, tl if k == 0 else None, 0)
    # epoch + loss readouts
    text(sl, "!!ep_k", 8.05, 4.95, 2.2, 0.35, "EPOCH", size=12, bold=True, font=HEAD, color=MUTED, spacing=2)
    text(sl, "!!ep_v", 8.0, 5.2, 2.4, 0.9, f"{ep}", size=44, bold=True, font=HEAD, color=INK)
    text(sl, "!!ac_k", 10.5, 4.95, 2.2, 0.35, "ACCURACY", size=12, bold=True, font=HEAD, color=MUTED, spacing=2)
    acc = D.nn_acc(P)
    text(sl, "!!ac_v", 10.45, 5.2, 2.4, 0.9, f"{acc * 100:.0f}%", size=44, bold=True, font=HEAD, color=TEAL if acc > 0.95 else INK)
    seg(sl, "!!lax_x", LAX.x0, LAX.y0 + LAX.h, LAX.x0 + LAX.w, LAX.y0 + LAX.h, FAINT, 1.25)
    seg(sl, "!!lax_y", LAX.x0, LAX.y0, LAX.x0, LAX.y0 + LAX.h, FAINT, 1.25)
    text(sl, "!!lax_l", LAX.x0 + 0.12, LAX.y0 - 0.4, 4, 0.3, "loss (log scale)", size=14, color=MUTED)
    text(sl, "!!lax_e", LAX.x0, LAX.y0 + LAX.h + 0.06, LAX.w, 0.3, "epochs (log scale)", size=14, color=MUTED, align="r")
    # growing loss curve: fixed 120 points, the unrevealed ones parked on the current end
    grid = np.unique(np.concatenate([np.linspace(0, 60, 40), np.geomspace(60, D.NN_EPOCHS, 80)]).astype(int))[:120]
    pts = []
    for e in grid:
        e2 = min(e, ep)
        pts.append(LAX.P(np.log10(e2 + 1), np.log10(max(D.nn_losses[e2], 0.01))))
    if ep == 0:
        pts = [(p[0] + 0.001 * i, p[1]) for i, p in enumerate(pts)]
    poly(sl, "!!lossline", pts, color=YELLOW, width=3)
    bx, by = pts[-1]
    ball(sl, bx, by, 0.12)
    if k == 0:
        caption(sl, tl, "Random start. Forward, loss, backward, step. Repeat.", 500)
    if last:
        caption(sl, tl, "It found the window on its own, by rolling downhill.", 200)
        pause_card(sl, tl, "100% on the training batches. Are we done? What would Lecture 1 say?", x=8.05, y=0.3)


# ============================================================================= WRAP
def quiz():
    sl, tl = new(7, """Pause the video, so to speak. Three questions; try them before next lecture. One: a bigger step is sometimes worse, because you overshoot the valley and land higher up the far wall. Two: one neuron is one straight boundary, and a window needs a closed curve, so you need several bent lines blended together. Three: minus nine says that increasing w1 decreases the loss, steeply, so gradient descent pushes w1 up; it's also the largest gradient, so w1 moves the most.""",
                 morph=1100)
    kicker(sl, "Pause and think", YELLOW)
    qs = ["Why can a bigger learning rate make the loss go up instead of down?",
          "Why can't a single neuron learn the operating window, no matter how long you train?",
          "In the hand-worked example ∂L/∂w₁ = −9. What does that number tell you to do with w₁?"]
    for i, q in enumerate(qs):
        x = 0.75 + i * 4.05
        c = rect(sl, f"qz{i}", x, 1.35, 3.75, 4.3, fill=PANEL, radius=0.12)
        n = text(sl, f"qzn{i}", x + 0.35, 1.65, 2, 0.9, f"{i + 1}", size=54, bold=True, font=HEAD, color=CRIMSON)
        t = text(sl, f"qzt{i}", x + 0.35, 2.75, 3.1, 2.7, q, size=20, color=INK)
        tl.add(c, "pop", 100 + i * 250, 400).add(n, "fade", 300 + i * 250, 300).add(t, "fade", 400 + i * 250, 400)


def end_screen():
    sl, tl = new(7, """[End screen] That's Lecture 2. Next time, Lecture 3: making deep learning actually work. Why plain gradient descent struggles on deep networks, and the fixes: momentum and Adam, normalisation, residual connections, dropout. Then CNNs, RNNs and embeddings. Questions?""",
                 morph=1300, scrub=False)
    # left third: identity
    poly(sl, "!!terrain", mark_pts(1.95, 2.25, 2.4, 0.75), color=MUTED, width=2.5)
    ball(sl, *mark_ball(1.95, 2.25, 2.4, 0.1), 0.1)
    a = text(sl, "es_t", 0.75, 2.55, 3.9, 1.5, "Learning by gradients", size=34, bold=True, font=HEAD, color=INK)
    b = text(sl, "es_s", 0.75, 3.85, 3.9, 0.4, "Lecture 2 of 5", size=16, color=CRIMSON, bold=True, font=HEAD)
    c = text(sl, "es_a", 0.75, 4.25, 3.9, 0.4, "Paras Koundal", size=16, color=MUTED)
    d = text(sl, "es_g", 0.75, 5.4, 3.9, 0.6, "github.com/ParasKoundal/ML-Slides", size=14, color=MUTED)
    tl.add(a, "rise", 200, 500).add(b, "fade", 500, 400).add(c, "fade", 600, 400).add(d, "fade", 700, 400)
    # right two-thirds: "up next" card
    cx, cy, cw, ch = 5.7, 1.35, 6.85, 3.85
    card = rect(sl, "es_card", cx, cy, cw, ch, fill=PANEL, line=FAINT, line_w=1.0, radius=0.08)
    k = text(sl, "es_k", cx, cy - 0.45, 3, 0.35, "UP NEXT", size=12, bold=True, font=HEAD, color=YELLOW, spacing=2)
    # thumbnail motif: a jagged training curve calming into a smooth one
    xs = np.linspace(0, 1, 70)
    rng = np.random.default_rng(1)
    jag = 0.85 * np.exp(-2.2 * xs) + 0.12 + 0.18 * np.exp(-1.2 * xs) * rng.normal(0, 1, 70) * (xs < 0.6)
    jag2 = 0.85 * np.exp(-4.5 * xs) + 0.08
    th = Ax(cx + 0.5, cy + 0.45, cw - 1.0, 2.2, (0, 1), (0, 1.2))
    j1 = poly(sl, "es_j1", [th.P(x, np.clip(y, 0, 1.2)) for x, y in zip(xs, jag)], color=MUTED, width=2)
    j2 = poly(sl, "es_j2", [th.P(x, y) for x, y in zip(xs, jag2)], color=TEAL, width=3.5)
    n3 = text(sl, "es_n", cx + 0.5, cy + 2.75, 1.2, 0.8, "03", size=40, bold=True, font=HEAD, color=CRIMSON)
    t3 = text(sl, "es_nt", cx + 1.65, cy + 2.8, cw - 2.0, 0.5, "Making deep learning work", size=26, bold=True, font=HEAD, color=INK)
    s3 = text(sl, "es_ns", cx + 1.65, cy + 3.3, cw - 2.0, 0.4, "Adam  ·  normalisation  ·  residuals  ·  CNNs", size=15, color=MUTED)
    tl.add(card, "pop", 800, 450).add(k, "fade", 900, 400).add(j1, "wipe_l", 1200, 700).add(j2, "wipe_l", 1700, 900)
    tl.add(n3, "fade", 1400, 400).add(t3, "fade", 1500, 400).add(s3, "fade", 1600, 400)
    q = dot(sl, "es_q", 9.13, 6.15, 0.55, None, line=INK, line_w=1.5)
    ql = text(sl, "es_ql", 9.13 - 0.55, 6.15 - 0.3, 1.1, 0.6, "?", size=30, bold=True, font=HEAD, color=INK, align="c", anchor="m")
    qt = text(sl, "es_qt", 9.85, 5.92, 3, 0.5, "Questions", size=20, color=MUTED)
    tl.add(q, "pop", 2400, 400).add(ql, "fade", 2500, 300).add(qt, "fade", 2550, 300)


