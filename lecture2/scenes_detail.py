"""
scenes_detail.py - the "worked by hand" layer of Lecture 2: formulas symbol by symbol, tables of real
numbers, chemical-engineering examples, animated GIFs, checks of understanding, glossary and reading.

Same visual language as scenes_core.py: dark stage, subtitles in a lower-third pill, Morph between slides,
rows and cards that build in by themselves.
"""
import os
import numpy as np
from PIL import Image

import data as D
from scenes_core import (new, caption, kicker, text, rect, dot, seg, poly, picture, ball, pause_card, fmt, Ax,
                         W, H, BG, PANEL, INK, MUTED, FAINT, ORANGE, BLUE, YELLOW, RED, TEAL, CRIMSON, HEAD, BODY,
                         MATH, HERE)

GIFS = os.path.join(HERE, "assets", "gifs")
MONO = "Courier New"
CH_NAMES = {0: "Intro", 1: "A line that learns", 2: "Rolling downhill", 3: "Yes or no?", 4: "Stacking neurons",
            5: "Blame flows backwards", 6: "Watch it learn", 7: "Summary"}


# ============================================================================= building blocks
_n = [0]


def uid(p):
    _n[0] += 1
    return f"{p}{_n[0]}"


def dslide(chapter, title, notes, morph=1000, kick=None, advance=None):
    """A detail slide: kicker + headline at a fixed position, content below."""
    sl, tl = new(chapter, notes, morph=morph, advance=advance)
    kicker(sl, kick or CH_NAMES[chapter], color=CRIMSON)
    t = text(sl, uid("dt"), 0.75, 0.82, 11.8, 0.7, title, size=30, bold=True, font=HEAD, color=INK)
    tl.add(t, "rise", 0, 450)
    return sl, tl


def table(sl, tl, x, y, cols, header, rows, size=16, row_h=0.46, at=500, step=180, hl_last=False):
    """cols: list of column widths (in). cells: str or (str, color) or list-of-runs."""
    hx = x
    for wd, hd in zip(cols, header):
        h = text(sl, uid("th"), hx, y, wd - 0.1, 0.3, hd.upper(), size=11, bold=True, font=HEAD, color=MUTED, spacing=1.5)
        tl.add(h, "fade", at, 300)
        hx += wd
    seg(sl, uid("tl"), x, y + 0.36, x + sum(cols), y + 0.36, FAINT, 1.0)
    yy = y + 0.46
    for ri, row in enumerate(rows):
        shapes = []
        if hl_last and ri == len(rows) - 1:
            shapes.append(rect(sl, uid("thl"), x - 0.1, yy - 0.06, sum(cols) + 0.2, row_h, fill=PANEL, radius=0.06))
        cx = x
        for wd, cell in zip(cols, row):
            col, runs = INK, None
            if isinstance(cell, tuple):
                cell, col = cell
            if isinstance(cell, list):
                runs = cell
            shapes.append(text(sl, uid("td"), cx, yy, wd - 0.1, row_h, runs or cell, size=size, color=col, font=BODY))
            cx += wd
        for s in shapes:
            tl.add(s, "fade", at + 250 + ri * step, 350)
        yy += row_h
    return yy


def card(sl, tl, x, y, w, h, head, body, color=YELLOW, at=300, size=16, head_size=20):
    c = rect(sl, uid("cd"), x, y, w, h, fill=PANEL, radius=0.1)
    t = text(sl, uid("ch"), x + 0.3, y + 0.25, w - 0.6, 0.45, head, size=head_size, bold=True, font=HEAD, color=color)
    b = text(sl, uid("cb"), x + 0.3, y + 0.8, w - 0.6, h - 0.95, body, size=size, color=INK)
    tl.add(c, "pop", at, 400).add(t, "fade", at + 200, 300).add(b, "fade", at + 300, 400)
    return c


def formula(sl, tl, x, y, w, runs, size=30, at=300, color=INK, align="l", h=0.75, effect="rise"):
    f = text(sl, uid("fm"), x, y, w, h, runs, size=size, font=MATH, color=color, align=align)
    tl.add(f, effect, at, 450)
    return f


def gif(sl, tl, name, x, y, w=None, h=None, at=200):
    path = os.path.join(GIFS, name)
    iw, ih = Image.open(path).size
    if w and not h:
        h = w * ih / iw
    elif h and not w:
        w = h * iw / ih
    p = picture(sl, uid("gif"), path, x, y, w, h)
    tl.add(p, "fade", at, 500)
    return p, w, h


def code(sl, tl, x, y, w, h, lines, size=14, at=300):
    c = rect(sl, uid("cbg"), x, y, w, h, fill="0A0C10", line=FAINT, line_w=0.75, radius=0.06)
    runs = []
    for i, ln in enumerate(lines):
        col = MUTED if ln.strip().startswith("#") else INK
        if "#" in ln and not ln.strip().startswith("#"):
            a, b = ln.split("#", 1)
            runs += [(a, {"color": INK}), ("#" + b, {"color": MUTED})]
        else:
            runs.append((ln, {"color": col}))
        if i < len(lines) - 1:
            runs.append(("\n", {}))
    t = text(sl, uid("code"), x + 0.3, y + 0.22, w - 0.5, h - 0.3, runs, size=size, font=MONO, color=INK)
    tl.add(c, "fade", at, 400).add(t, "wipe_d", at + 200, 900)
    return t


def quiz(chapter, qs, answers):
    sl, tl = dslide(chapter, "Check your understanding",
                    "Suggested answers. " + " ".join(f"Q{i + 1}: {a}" for i, a in enumerate(answers)))
    for i, q in enumerate(qs):
        x = 0.75 + (i % 2) * 6.0
        y = 1.75 + (i // 2) * 2.25
        c = rect(sl, uid("qc"), x, y, 5.8, 2.05, fill=PANEL, radius=0.1)
        n = text(sl, uid("qn"), x + 0.3, y + 0.22, 1, 0.4, f"Q{i + 1}", size=16, bold=True, font=HEAD, color=CRIMSON)
        t = text(sl, uid("qt"), x + 0.3, y + 0.65, 5.2, 1.35, q, size=17, color=INK)
        tl.add(c, "pop", 300 + i * 200, 400).add(n, "fade", 450 + i * 200, 300).add(t, "fade", 500 + i * 200, 400)


def sub(s):
    return {"sub": True, "text": s}


def M(*parts):
    """math runs: plain strings are upright, ('x', 'i') italic, ('2', 'sub'/'sup')"""
    out = []
    for p in parts:
        if isinstance(p, str):
            out.append((p, {}))
        else:
            t, k = p
            out.append((t, {"italic": k == "i", "sub": k == "sub", "sup": k == "sup",
                            "color": {"y": YELLOW, "r": RED, "t": TEAL, "o": ORANGE}.get(k, INK)}))
    return out


# ============================================================================= INTRO
def bridge():
    sl, tl = dslide(0, "Lecture 1 already did gradient descent, on predictions",
                    """Remember the end of Lecture 1. Gradient boosting fitted each new tree to the residuals, and we showed that fitting residuals is exactly a step downhill on the loss, where the thing being moved was the prediction itself. Today we keep the loss and the downhill step, but we move the numbers inside the model, its weights. That one change is what makes neural networks, and eventually LLMs, possible.""")
    card(sl, tl, 0.75, 1.85, 5.75, 3.6, "Lecture 1 · boosting",
         "Each new tree nudges the predictions in the direction that lowers the loss.\n\nWhat moves: the prediction F(x) itself, one small tree at a time.",
         color=MUTED, at=400)
    card(sl, tl, 6.83, 1.85, 5.75, 3.6, "Lecture 2 · gradients",
         "Each step nudges the weights inside the model in the direction that lowers the loss.\n\nWhat moves: the knobs w and b, all of them at once.",
         color=YELLOW, at=900)
    formula(sl, tl, 1.05, 4.6, 5.3, M(("F", "i"), "  ←  ", ("F", "i"), " − η · ∂", ("L", "i"), "/∂", ("F", "i")), size=24, at=1500, color=MUTED)
    formula(sl, tl, 7.13, 4.6, 5.3, M(("w", "y"), "  ←  ", ("w", "i"), " − η · ∂", ("L", "i"), "/∂", ("w", "i")), size=24, at=1800, color=YELLOW)
    caption(sl, tl, "Same loss, same downhill step. Today the step moves the weights.", 2300)


def solver():
    sl, tl = dslide(0, "You have done this before: least squares in a spreadsheet",
                    """Most chemical engineers have already trained a model without calling it that. Fitting kinetic parameters, a heat-transfer correlation, or an Antoine equation: you guess parameters, compute the sum of squared residuals, and let Solver or a nonlinear least-squares routine change the parameters until that number is as small as possible. That is model, loss, learning. What's new today is how the learning works inside, the gradient, and why the same method scales from 2 parameters to 175 billion.""")
    rows = [["model", "k = A·exp(−Ea / RT)", "any function with knobs w"],
            ["loss", "sum of squared residuals", "mean squared error, cross-entropy"],
            ["learning", "Solver / Levenberg–Marquardt", "gradient descent + backprop"],
            ["knobs", "2 (A, Ea)", "2 … 175 000 000 000"]]
    table(sl, tl, 0.85, 1.9, [2.2, 4.4, 4.9], ["", "kinetic fit you've done", "today"], rows, size=18, row_h=0.62, at=500, step=300)
    caption(sl, tl, "Fitting kinetic parameters is training a model. Today: what happens inside.", 2200)


def vocab():
    sl, tl = dslide(0, "New vocabulary for today",
                    """New words we'll use. Weights and biases are the parameters: the knobs training turns. The prediction is y-hat. The residual is what's still wrong. The gradient is the vector of slopes of the loss, one per knob. The learning rate sets the step size. A mini-batch is the handful of examples used for one step, and an epoch is one full pass over the data. An activation is the bend applied inside a neuron, and a logit is the raw score before it's squashed into a probability.""")
    rows = [["weight w, bias b", "the knobs (parameters) training adjusts"],
            ["prediction  ŷ", "what the model outputs for one example"],
            ["residual  y − ŷ", "what is still wrong for that example"],
            ["gradient  ∇L", "slope of the loss with respect to every knob"],
            ["learning rate  η", "how big a step to take downhill"],
            ["mini-batch, epoch", "examples used per step / one full pass over the data"],
            ["activation  g(z)", "the bend inside a neuron: ReLU, sigmoid, tanh"],
            ["logit  z", "raw score before it is squashed into a probability"]]
    table(sl, tl, 0.85, 1.75, [3.6, 8.0], ["term", "meaning"], rows, size=17, row_h=0.53, at=400, step=140)


# ============================================================================= CH1 detail
def mse_formula():
    sl, tl = dslide(1, "The loss, symbol by symbol",
                    """Here's the loss we just watched, written properly. For each of the n batches, take the measured titre y-i, subtract the line's prediction w x-i plus b, square it, and average. It's a function of the knobs w and b only: the data are fixed numbers. Every choice of w and b gives one loss value; training means finding the pair with the smallest.""")
    formula(sl, tl, 0.85, 1.8, 11.5, M(("L", "i"), "(", ("w", "y"), ", ", ("b", "y"), ")  =  ", "1/", ("n", "i"), " Σ ", "(", ("y", "i"), ("i", "sub"), " − (", ("w", "y"), " ", ("x", "i"), ("i", "sub"), " + ", ("b", "y"), "))", ("2", "sup")), size=38, at=300, h=0.9)
    rows = [[("xᵢ", ORANGE), "feature of batch i", "glucose feed rate (g/L/h)"],
            [("yᵢ", ORANGE), "measured label of batch i", "titre (g/L)"],
            [("w, b", YELLOW), "the knobs: slope and intercept", "what training changes"],
            [("yᵢ − (w xᵢ + b)", RED), "residual of batch i", "the vertical miss"],
            [("1/n Σ", INK), "average over all n batches", "mean squared error (MSE)"]]
    table(sl, tl, 0.85, 3.0, [3.2, 4.3, 4.1], ["symbol", "meaning", "here"], rows, size=17, row_h=0.5, at=900)
    caption(sl, tl, "The data are fixed. The loss is a function of the knobs only.", 2500)


def h4_by_hand():
    w, b = D.H4_W0, D.H4_B0
    yh, r, r2, L = D.h4_table(w, b)
    sl, tl = dslide(1, "Worked by hand: 4 batches, one guess",
                    f"""Let's compute a loss by hand with four batches, so the numbers are friendly. Feed rates 1, 2, 3, 4; titres 2.1, 3.9, 6.2, 7.8. Guess w equals {w:g} and b equals {b:g}. Predictions: 2, 3, 4, 5. Misses: {', '.join(f'{v:.1f}' for v in r)}. Squared: {', '.join(f'{v:.2f}' for v in r2)}. Average: {L:.3f}. That's the loss of this guess. Notice the last batch dominates: squaring makes a miss of 2.8 count 784 times more than a miss of 0.1.""")
    rows = []
    for i in range(4):
        rows.append([f"#{i + 1}", f"{D.H4_X[i]:g}", f"{D.H4_Y[i]:g}", (f"{yh[i]:g}", YELLOW), (f"{r[i]:.1f}", RED), (f"{r2[i]:.2f}", RED)])
    rows.append(["", "", "", "", ("mean", MUTED), (f"{L:.3f}", INK)])
    table(sl, tl, 0.85, 1.75, [1.2, 2.0, 1.9, 2.4, 2.3, 2.0], ["batch", "x feed", "y titre", "ŷ = 1·x + 1", "y − ŷ", "(y − ŷ)²"],
          rows, size=19, row_h=0.56, at=400, step=250, hl_last=True)
    caption(sl, tl, f"One guess, one number: loss = {L:.3f}. Big misses dominate.", 2600)


def closed_form():
    sl, tl = dslide(1, "For a straight line there is a shortcut",
                    f"""For a straight line we don't even need to search: set both slopes of the loss to zero and solve. You get the textbook formulas: w-star is the covariance of x and y divided by the variance of x, and b-star makes the line pass through the means. On our four batches: covariance {D.H4_COV:.3f}, variance {D.H4_VAR:.2f}, so w-star is {D.H4_WSTAR:.2f}, and b-star is {D.H4_YBAR:.1f} minus {D.H4_WSTAR:.2f} times {D.H4_XBAR:.1f}, which is {D.H4_BSTAR:.2f}. Loss {D.h4_table(D.H4_WSTAR, D.H4_BSTAR)[3]:.4f}. So why learn gradient descent? Because this trick only works when the model is linear in its knobs. For a neural network there is no formula: we have to walk downhill.""")
    formula(sl, tl, 0.85, 1.85, 6.0, M(("w", "y"), "*  =  cov(", ("x", "i"), ", ", ("y", "i"), ") / var(", ("x", "i"), ")"), size=30, at=300)
    formula(sl, tl, 0.85, 2.7, 6.0, M(("b", "y"), "*  =  ", ("ȳ", "i"), " − ", ("w", "y"), "* ", ("x̄", "i")), size=30, at=600)
    rows = [["x̄, ȳ", f"{D.H4_XBAR:.1f},  {D.H4_YBAR:.1f}"],
            ["cov(x, y)", f"{D.H4_COV:.3f}"],
            ["var(x)", f"{D.H4_VAR:.2f}"],
            [("w*", YELLOW), (f"{D.H4_COV:.3f} / {D.H4_VAR:.2f} = {D.H4_WSTAR:.2f}", YELLOW)],
            [("b*", YELLOW), (f"{D.H4_YBAR:.1f} − {D.H4_WSTAR:.2f}·{D.H4_XBAR:.1f} = {D.H4_BSTAR:.2f}", YELLOW)],
            ["loss at (w*, b*)", (f"{D.h4_table(D.H4_WSTAR, D.H4_BSTAR)[3]:.4f}", TEAL)]]
    table(sl, tl, 0.85, 3.6, [2.6, 3.6], ["quantity", "4 batches"], rows, size=17, row_h=0.42, at=900, step=200)
    card(sl, tl, 7.4, 1.85, 5.15, 3.9, "So why gradient descent?",
         "This shortcut exists only when the model is linear in its knobs.\n\nA neural network has no formula for its best weights. The only way is to walk downhill.",
         color=RED, at=2400)


def arrhenius(stage):
    T, K = D.ARR_T, D.ARR_K
    notes = {
        0: f"""A real chemical-engineering regression. Eight rate constants measured between 300 and 370 kelvin. Plotted against temperature they curve upwards: Arrhenius, k equals A e to the minus Ea over RT. A straight line through these points would be terrible.""",
        1: f"""But take logs. ln k equals ln A minus Ea over R, times 1 over T. Plot ln k against 1000 over T and the same eight points fall on a line. Now it's plain linear regression: the slope is minus Ea over R, and the intercept is ln A.""",
        2: f"""Fit the line and read off the physics. Slope {D.ARR_SLOPE:.2f} thousand kelvin, so Ea equals {D.ARR_SLOPE:.2f} times minus 1000 times 8.314: {D.ARR_EA_FIT / 1000:.1f} kilojoules per mole. The data were simulated with 55, plus 6 percent noise on each k. The lesson: linear regression means linear in the knobs, not in the raw measurements. Transform the features and a line can fit a lot. Neural networks will learn such transformations by themselves.""",
    }[stage]
    sl, tl = dslide(1, ["Arrhenius: a curve that hides a straight line", "Take logs: the curve becomes a line",
                        "The slope is the activation energy"][stage], notes, morph=1500)
    if stage == 0:
        ax = Ax(1.3, 1.9, 6.4, 3.9, (295, 375), (0, 0.17))
        X, Y = T, K
        xl, yl = "temperature T (K)", "rate constant k (1/s)"
    else:
        ax = Ax(1.3, 1.9, 6.4, 3.9, (2.65, 3.37), (-6.2, -1.6))
        X, Y = D.ARR_X, D.ARR_Y
        xl, yl = "1000 / T  (1/K)", "ln k"
    seg(sl, "!!arx", ax.x0, ax.y0 + ax.h, ax.x0 + ax.w, ax.y0 + ax.h, FAINT, 1.25)
    seg(sl, "!!ary", ax.x0, ax.y0, ax.x0, ax.y0 + ax.h, FAINT, 1.25)
    text(sl, uid("arxl"), ax.x0 + ax.w - 4, ax.y0 + ax.h + 0.1, 4, 0.3, xl, size=14, color=MUTED, align="r")
    text(sl, uid("aryl"), ax.x0 + 0.12, ax.y0 - 0.05, 4, 0.3, yl, size=14, color=MUTED)
    pts = [dot(sl, f"!!ar{i}", ax.X(x), ax.Y(y), 0.09, ORANGE) for i, (x, y) in enumerate(zip(X, Y))]
    if stage == 0:
        tl.stagger(pts, "pop", 400, 120, 350)
        formula(sl, tl, 8.3, 2.2, 4.5, M(("k", "i"), " = ", ("A", "i"), " e", ("−Ea/RT", "sup")), size=36, at=1600)
        t = text(sl, uid("arn"), 8.3, 3.2, 4.4, 1.5, "8 rate constants, 300–370 K\n(simulated: Ea = 55 kJ/mol, 6% noise)", size=16, color=MUTED)
        tl.add(t, "fade", 1900, 400)
        caption(sl, tl, "Rate constants curve upward with temperature. No straight line fits.", 2200)
        return
    formula(sl, tl, 8.3, 2.0, 4.6, M("ln ", ("k", "i"), " = ln ", ("A", "i"), " − (", ("Ea", "i"), "/", ("R", "i"), ") · 1/", ("T", "i")), size=26, at=300)
    lab = text(sl, uid("arm"), 8.3, 2.8, 4.6, 1.4, [("y", {"italic": True, "color": ORANGE}), ("  =  ", {}), ("b", {"italic": True, "color": YELLOW}),
                                               ("  +  ", {}), ("w", {"italic": True, "color": YELLOW}), ("  ·  ", {}), ("x", {"italic": True, "color": ORANGE})],
               size=30, font=MATH, color=INK)
    tl.add(lab, "fade", 700, 400)
    xs = np.array([2.68, 3.34])
    ln = poly(sl, "!!arline", [ax.P(x, D.ARR_SLOPE * x + D.ARR_ICPT) for x in xs], color=YELLOW, width=3)
    if stage == 1:
        tl.add(ln, "wipe_l", 1300, 700)
        caption(sl, tl, "Plot ln k against 1/T and the curve becomes a straight line.", 1800)
        return
    rows = [["slope w", (f"{D.ARR_SLOPE:.2f} × 1000 K", YELLOW)],
            ["Ea = −w · 1000 · R", (f"{D.ARR_EA_FIT / 1000:.1f} kJ/mol", TEAL)],
            ["A = e^b", (f"{D.ARR_A_FIT:.1e} 1/s".replace("e+06", " × 10⁶").replace("e+07", " × 10⁷"), TEAL)],
            ["true values", "55 kJ/mol,  1.0 × 10⁷ 1/s"]]
    table(sl, tl, 8.3, 3.85, [2.4, 2.4], ["", "fitted"], rows, size=15, row_h=0.45, at=500, step=250)
    caption(sl, tl, "Linear regression means linear in the knobs, not in the raw measurements.", 1800)


def plane():
    w = D.PL_W
    sl, tl = dslide(1, "More features: the line becomes a plane",
                    f"""Real processes have more than one input. With feed rate and temperature, the model is y-hat equals w1 x1 plus w2 x2 plus b: a flat plane through a cloud of points in 3-D. With 20 process variables it's a 20-dimensional plane we can't draw, but the loss, the formulas and gradient descent are exactly the same. Fitted here: {w[0]:.2f} grams per litre per unit of feed rate, and {w[1]:.2f} per degree.""")
    gif(sl, tl, "plane3d.gif", 0.7, 1.6, h=5.0, at=200)
    formula(sl, tl, 7.55, 1.9, 5.2, M(("ŷ", "i"), " = ", ("w", "y"), ("1", "sub"), ("x", "i"), ("1", "sub"), " + ", ("w", "y"), ("2", "sub"), ("x", "i"), ("2", "sub"), " + ", ("b", "y")), size=32, at=500)
    rows = [["x₁  feed rate", (f"w₁ = {w[0]:.2f}", YELLOW)],
            ["x₂  temperature", (f"w₂ = {w[1]:.2f}", YELLOW)],
            ["intercept", (f"b = {w[2]:.2f}", YELLOW)]]
    table(sl, tl, 7.55, 2.9, [2.6, 2.4], ["feature", "fitted knob"], rows, size=17, row_h=0.48, at=900)
    t = text(sl, uid("pln"), 7.55, 4.85, 5.0, 1.0, "With d features: ŷ = w·x + b, a plane in d dimensions. Same loss, same gradient descent.", size=16, color=MUTED)
    tl.add(t, "fade", 1800, 400)
    caption(sl, tl, "Two inputs: a plane. Twenty inputs: the same maths in 20-D.", 2000)


def quiz1():
    quiz(1, ["Three batches have residuals 0.5, −1.0 and 2.0. What is their mean squared error?",
             "Why does squaring the residuals make one bad batch dominate the loss?",
             "Arrhenius is curved in T. Why can we still fit it with linear regression?",
             "When can we skip gradient descent and solve for the best knobs directly?"],
            ["(0.25 + 1 + 4)/3 = 1.75.",
             "The penalty grows with the square of the miss: a miss twice as big costs four times as much. Good when big errors are dangerous; bad if a single outlier is a measurement error.",
             "ln k is linear in 1/T, and linear regression only needs the model to be linear in its knobs (ln A and Ea/R), not in the raw measurement.",
             "When the model is linear in its parameters and the loss is squared error: set the gradient to zero and solve the normal equations. Not possible for neural networks."])


# ============================================================================= CH2 detail
def grad_formula():
    sl, tl = dslide(2, "The slope, written out",
                    """Here's the slope the ball was feeling, computed exactly. Differentiate the mean squared error with respect to w: you get minus 2 over n times the sum of x-i times the residual. With respect to b: minus 2 over n times the sum of the residuals. Read it in words: the slope is a weighted vote of the misses. Batches the line under-predicts pull w up; batches it over-predicts pull it down; big feed rates get a bigger vote for the slope. When the votes cancel, the slope is zero and we're at the bottom.""")
    formula(sl, tl, 0.85, 1.85, 11.6, M("∂", ("L", "i"), "/∂", ("w", "y"), "  =  − 2/", ("n", "i"), " Σ ", ("x", "i"), ("i", "sub"), " (", ("y", "i"), ("i", "sub"), " − ", ("ŷ", "i"), ("i", "sub"), ")"), size=34, at=300)
    formula(sl, tl, 0.85, 2.75, 11.6, M("∂", ("L", "i"), "/∂", ("b", "y"), "  =  − 2/", ("n", "i"), " Σ (", ("y", "i"), ("i", "sub"), " − ", ("ŷ", "i"), ("i", "sub"), ")"), size=34, at=700)
    card(sl, tl, 0.85, 3.95, 3.7, 1.9, "under-predicted", "y > ŷ: residual positive, pushes w and b up", color=ORANGE, at=1300, size=16)
    card(sl, tl, 4.8, 3.95, 3.7, 1.9, "over-predicted", "y < ŷ: residual negative, pushes w and b down", color=BLUE, at=1500, size=16)
    card(sl, tl, 8.75, 3.95, 3.75, 1.9, "balanced", "votes cancel: slope = 0, the bottom of the valley", color=TEAL, at=1700, size=16)
    caption(sl, tl, "The slope is a vote of the misses. At the bottom the votes cancel.", 2200)


def gd_by_hand():
    rows = []
    for k, (w, b) in enumerate(D.H4_ITERS):
        gw, gb = D.h4_grad(w, b)
        L = D.h4_table(w, b)[3]
        rows.append([f"{k}", (f"{w:.3f}", YELLOW), (f"{b:.3f}", YELLOW), (fmt(gw, 2), RED), (fmt(gb, 2), RED), f"{L:.3f}"])
    w0, b0 = D.H4_ITERS[0]
    gw0, gb0 = D.h4_grad(w0, b0)
    w1, b1 = D.H4_ITERS[1]
    sl, tl = dslide(2, "Gradient descent by hand on the 4 batches",
                    f"""Back to our four batches, starting from w equals 1, b equals 1, with learning rate 0.05. Step 0: the slopes are {gw0:.2f} for w and {gb0:.2f} for b. Both negative, so both knobs go up: w becomes 1 minus 0.05 times {gw0:.2f}, which is {w1:.4f}; b becomes {b1:.3f}. The loss falls from {D.h4_table(w0, b0)[3]:.3f} to {D.h4_table(w1, b1)[3]:.3f} in one step. Keep going and it creeps toward the exact answer, 1.94 and 0.15, that the shortcut gave us.""")
    table(sl, tl, 0.85, 1.8, [1.3, 1.9, 1.9, 2.2, 2.2, 1.9], ["step", "w", "b", "∂L/∂w", "∂L/∂b", "loss"], rows, size=19, row_h=0.56, at=400, step=450)
    formula(sl, tl, 0.85, 4.75, 11.5, M("step 0 → 1:   ", ("w", "y"), " = 1 − 0.05 × (", fmt(gw0, 2), ") = ", f"{w1:.4f}", "     ", ("b", "y"), " = 1 − 0.05 × (", fmt(gb0, 2), ") = ", f"{b1:.3f}"), size=20, at=2500, color=MUTED, h=0.5)
    caption(sl, tl, f"One step: loss {D.h4_table(w0, b0)[3]:.3f} → {D.h4_table(w1, b1)[3]:.3f}. It heads for w* = 1.94, b* = 0.15.", 2900)


def gd_algorithm():
    sl, tl = dslide(2, "The whole algorithm on one slide",
                    """Everything so far, as an algorithm. Pick a learning rate and a number of steps. Start the knobs somewhere, usually small random numbers. Then repeat: compute predictions, compute the loss, compute its gradient, and step every knob against its own slope. Stop when the loss stops improving on validation data. This loop is unchanged from linear regression all the way to training GPT; only how we compute the gradient gets clever.""")
    code(sl, tl, 0.85, 1.8, 7.0, 4.1, [
        "w, b = small random numbers",
        "for step in range(n_steps):",
        "    y_hat = w * x + b                # predict",
        "    L = mean((y - y_hat) ** 2)       # loss",
        "    dw = -2 * mean(x * (y - y_hat))  # gradient",
        "    db = -2 * mean(y - y_hat)",
        "    w = w - lr * dw                  # step",
        "    b = b - lr * db",
        "    if validation loss stops improving:",
        "        break"], size=15, at=300)
    card(sl, tl, 8.2, 1.8, 4.35, 1.25, "hyper-parameters", "learning rate lr, number of steps", color=MUTED, at=1600, size=15, head_size=17)
    card(sl, tl, 8.2, 3.2, 4.35, 1.25, "parameters", "w, b: set by the loop", color=YELLOW, at=1800, size=15, head_size=17)
    card(sl, tl, 8.2, 4.6, 4.35, 1.3, "never changes", "the same loop trains GPT; only the gradient step gets clever", color=TEAL, at=2000, size=15, head_size=17)


def lr_response():
    sl, tl = dslide(2, "Learning rate is a controller gain",
                    f"""Chemical engineers have seen this picture before: it's a step response. Plot the knob w against step number for four learning rates. Small eta: sluggish, overdamped. Medium: fast, no overshoot. Larger: it overshoots and rings, like a controller with too much gain. Beyond a limit it's unstable and the oscillation grows. For a bowl-shaped loss you can compute the limit exactly: eta must stay below 2 divided by the curvature. Here the curvature is {D.CURV:.2f}, so the limit is {D.LR_ETA_MAX:.2f}, which is why 0.30 blew up. Tuning a learning rate is tuning a loop gain.""")
    gif(sl, tl, "lr_response.gif", 0.7, 1.6, w=7.4, at=200)
    formula(sl, tl, 8.4, 1.9, 4.3, M(("η", "t"), "  <  2 / curvature"), size=32, at=900)
    rows = [["curvature  ∂²L/∂w²", f"{D.CURV:.2f}"], ["stability limit", (f"η < {D.LR_ETA_MAX:.2f}", TEAL)],
            ["η = 0.03", ("sluggish", MUTED)], ["η = 0.12", ("fast", TEAL)], ["η = 0.25", ("rings", YELLOW)], ["η = 0.30", ("unstable", RED)]]
    table(sl, tl, 8.4, 2.85, [2.6, 1.7], ["", ""], rows, size=16, row_h=0.45, at=1300, step=200)
    caption(sl, tl, "Too much gain and the loop oscillates. Too much learning rate, same story.", 2600)


def bumpy():
    sl, tl = dslide(2, "One valley, or many?",
                    """For linear and logistic regression the loss is a single bowl: convex. Wherever you start, downhill leads to the same bottom. Neural networks are not convex: the landscape has many valleys, and where you start can decide where you end, as these three balls show. Two pieces of good news from practice: in high dimensions most valleys turn out to be nearly as good as each other, and the noise of mini-batches helps the ball hop out of small dips. This is also why we start networks from random weights and sometimes train several times.""")
    gif(sl, tl, "bumpy.gif", 0.7, 1.7, w=8.0, at=200)
    card(sl, tl, 9.0, 1.8, 3.6, 1.85, "convex", "linear and logistic regression: one bowl, one answer", color=TEAL, at=900, size=15, head_size=18)
    card(sl, tl, 9.0, 3.85, 3.6, 2.0, "non-convex", "neural networks: many valleys; the start matters", color=ORANGE, at=1200, size=15, head_size=18)
    caption(sl, tl, "Three starts, three different valleys. Downhill is local.", 1800)


def scaling():
    sl, tl = dslide(2, "Mixed units make a bad landscape: standardise",
                    f"""This one bites chemical engineers constantly, because our variables come in wildly different units. Here titre depends on temperature, spread over a few degrees, and pH, spread over a few tenths. In raw units the valley is a long thin trench: the step size that's stable across the steep direction is far too small along the shallow one, so after 60 steps the pH weight has barely moved. Standardise each feature, subtract its mean and divide by its standard deviation, and the valley becomes round: a handful of steps. Always scale inputs before gradient descent. Trees in Lecture 1 didn't care; gradients do.""")
    gif(sl, tl, "scaling.gif", 0.6, 1.6, w=8.4, at=200)
    formula(sl, tl, 9.25, 1.85, 3.6, M(("x", "i"), "′ = (", ("x", "i"), " − mean) / sd"), size=24, at=900)
    rows = [["T spread", f"{D.SC_RAW[:, 0].std():.2f} °C"], ["pH spread", f"{D.SC_RAW[:, 1].std():.2f}"],
            ["curvature ratio", (f"{D.SC_CURV_RAW[0] / D.SC_CURV_RAW[1]:.0f} : 1", RED)], ["after scaling", ("1 : 1", TEAL)]]
    table(sl, tl, 9.25, 2.7, [1.9, 1.5], ["", ""], rows, size=15, row_h=0.45, at=1300, step=200)
    caption(sl, tl, "Raw units: a long trench and a crawl. Standardised: a round bowl.", 2400)


def sgd_detail():
    steps = D.EB_STEPS
    sl, tl = dslide(2, "Epochs, batches and steps",
                    f"""The vocabulary of mini-batch training, with numbers. Say you have {D.EB_N:,} historical batch records. With a mini-batch size of {D.EB_BATCH}, one pass over all the data, one epoch, is {steps} gradient steps. Training for 50 epochs is about {50 * steps:,} steps. The animation compares the exact full-batch path with mini-batches of four on our 14 batches: the yellow loss curve is noisier, but each step costs a fraction of the work.""")
    gif(sl, tl, "sgd.gif", 0.6, 1.6, w=8.3, at=200)
    rows = [["records", f"{D.EB_N:,}"], ["batch size", f"{D.EB_BATCH}"], ["steps per epoch", (f"{D.EB_N:,} / {D.EB_BATCH} ≈ {steps}", YELLOW)],
            ["50 epochs", f"≈ {50 * steps:,} steps"]]
    table(sl, tl, 9.15, 1.9, [1.9, 2.0], ["", ""], rows, size=15, row_h=0.5, at=900, step=200)
    t = text(sl, uid("ebn"), 9.15, 4.2, 3.6, 1.6, "epoch: one full pass over the data\nstep: one gradient update, from one mini-batch", size=15, color=MUTED)
    tl.add(t, "fade", 1800, 400)
    caption(sl, tl, "One epoch = every example seen once = many small, cheap steps.", 2000)


def quiz2():
    quiz(2, [f"The slope dL/dw is −4 and η = 0.1. Does w go up or down, and by how much?",
             "Your training loss jumps up and down and then becomes NaN. What is the first thing you change?",
             "You fit titre from temperature in kelvin (~300) and pressure in bar (~1). Why will plain gradient descent struggle?",
             "With 50,000 records and a batch size of 100, how many steps are in one epoch?"],
            ["Up, by 0.1 × 4 = 0.4.",
             "Lower the learning rate: the steps overshoot the valley (above the 2 / curvature limit).",
             "The landscape is a long thin trench: one direction is far steeper. Standardise both features first.",
             "500 steps."])


# ============================================================================= CH3 detail
def sigmoid_table():
    zs = [-4, -2, 0, 2, 4]
    rows = [[(f"{z:+d}".replace("-", "−"), YELLOW), f"{D.sigmoid(z):.3f}", f"{np.exp(z):.3g}" if z else "1"] for z in zs]
    sl, tl = dslide(3, "Inside the sigmoid: scores, odds and probabilities",
                    f"""What the sigmoid actually does. The line's output z is a raw score, called the logit. Z of zero means p of one half: a coin toss. Every increase of z by 1 multiplies the odds, p over 1 minus p, by e, about 2.7. So z is the log of the odds. The decision boundary is where p equals one half, which is where z equals zero: w x plus b equals zero, x equals minus b over w. For our batches that's a lactate of {D.LG_THRESH:.2f} grams per litre: above it, the model bets on failure.""")
    formula(sl, tl, 0.85, 1.8, 6.0, M(("p", "i"), " = σ(", ("z", "y"), ") = 1 / (1 + ", ("e", "i"), ("−z", "sup"), ")"), size=30, at=300)
    formula(sl, tl, 0.85, 2.65, 6.0, M(("z", "y"), " = ln( ", ("p", "i"), " / (1 − ", ("p", "i"), ") )   the log-odds"), size=24, at=700, color=MUTED)
    table(sl, tl, 0.85, 3.5, [1.5, 2.0, 2.4], ["logit z", "p = σ(z)", "odds p/(1−p)"], rows, size=17, row_h=0.42, at=1000, step=150)
    card(sl, tl, 7.4, 1.8, 5.15, 4.05, "The decision boundary",
         f"p = 0.5  ⇔  z = 0  ⇔  x = −b / w\n\nFor the lactate model:\nx = {-D.LG_B:.2f} / {D.LG_W:.2f} = {D.LG_THRESH:.2f} g/L\n\nAbove it the model bets “fail”. In 2-D the boundary is a line, in d dimensions a plane.",
         color=YELLOW, at=2000, size=16)
    caption(sl, tl, f"p = 0.5 exactly where z = 0: at {D.LG_THRESH:.2f} g/L lactate.", 2600)


def ce_formula():
    sl, tl = dslide(3, "Cross-entropy, symbol by symbol",
                    """The full cross-entropy loss. For each batch, if it failed, y equals 1 and we pay minus log p. If it passed, y equals 0 and we pay minus log of 1 minus p, the probability the model gave to passing. The y and 1 minus y factors just switch between the two cases. Average over the batches. It's the average surprise at what actually happened.""")
    formula(sl, tl, 0.85, 1.85, 11.6, M(("L", "i"), " = − 1/", ("n", "i"), " Σ [ ", ("y", "o"), ("i", "sub"), " ln ", ("p", "y"), ("i", "sub"), "  +  (1 − ", ("y", "o"), ("i", "sub"), ") ln(1 − ", ("p", "y"), ("i", "sub"), ") ]"), size=34, at=300, h=0.9)
    rows = [[("yᵢ", ORANGE), "true outcome of batch i", "1 = failed spec, 0 = passed"],
            [("pᵢ", YELLOW), "model's probability of failure", "σ(w xᵢ + b)"],
            [("−ln pᵢ", RED), "surprise if it failed", "0 when p = 1, ∞ when p → 0"],
            [("−ln(1 − pᵢ)", RED), "surprise if it passed", "0 when p = 0, ∞ when p → 1"],
            [("y, 1 − y", INK), "switches: only one term is on", "pick the case that happened"]]
    table(sl, tl, 0.85, 3.05, [2.9, 4.4, 4.3], ["symbol", "meaning", "behaviour"], rows, size=17, row_h=0.5, at=900)
    caption(sl, tl, "Cross-entropy = average surprise at what actually happened.", 2500)


def ce_by_hand():
    rows = []
    for i in range(4):
        rows.append([f"{D.LH_X[i]:g}", ("fail" if D.LH_Y[i] else "pass", ORANGE if D.LH_Y[i] else BLUE), (fmt(D.LH_Z[i], 1), YELLOW),
                     f"{D.LH_P[i]:.3f}", ("−ln p" if D.LH_Y[i] else "−ln(1−p)", MUTED), (f"{D.LH_LOSS[i]:.3f}", RED)])
    rows.append(["", "", "", "", ("mean", MUTED), (f"{D.LH_MEAN:.3f}", INK)])
    sl, tl = dslide(3, "Worked by hand: 4 batches, w = 1.6, b = −4.8",
                    f"""Four batches by hand. Lactate 1.5 and 2.5 passed; 3.5 and 4.5 failed. With w equals 1.6 and b equals minus 4.8, the logits are minus 2.4, minus 0.8, 0.8 and 2.4. Through the sigmoid: {', '.join(f'{p:.3f}' for p in D.LH_P)}. The first batch passed and the model gave failure only 8 percent, so the surprise is minus ln of 0.917, just {D.LH_LOSS[0]:.3f}. The second passed but the model said 31 percent fail: surprise {D.LH_LOSS[1]:.3f}. Average: {D.LH_MEAN:.3f}.""")
    table(sl, tl, 0.85, 1.75, [1.8, 1.6, 2.1, 2.1, 2.3, 1.8], ["lactate x", "outcome", "z = 1.6x − 4.8", "p (fail)", "pay", "loss"], rows, size=19, row_h=0.56, at=400, step=250, hl_last=True)
    caption(sl, tl, f"Confident and right costs little. Average surprise = {D.LH_MEAN:.3f}.", 2600)


def why_not_mse():
    sl, tl = dslide(3, "Why not just use squared error for yes/no?",
                    """Why a new loss? Try squared error on a probability. When the batch failed and the model said almost zero, squared error caps out at 1: it barely cares how confidently wrong the model was. Cross-entropy shoots to infinity. Worse, with the sigmoid in the way, the squared-error gradient gets multiplied by p times 1 minus p, which is tiny exactly when the model is confidently wrong, so learning stalls. With cross-entropy that factor cancels and the gradient is simply p minus y: big when wrong, small when right.""")
    ax = Ax(1.3, 1.95, 5.6, 3.8, (0, 1), (0, 4))
    seg(sl, uid("wx"), ax.x0, ax.y0 + ax.h, ax.x0 + ax.w, ax.y0 + ax.h, FAINT, 1.25)
    seg(sl, uid("wy"), ax.x0, ax.y0, ax.x0, ax.y0 + ax.h, FAINT, 1.25)
    text(sl, uid("wxl"), ax.x0 + ax.w - 5, ax.y0 + ax.h + 0.08, 5, 0.3, "p given to the truth (y = 1)", size=14, color=MUTED, align="r")
    text(sl, uid("wyl"), ax.x0 + 0.12, ax.y0 - 0.05, 3, 0.3, "loss", size=14, color=MUTED)
    ps = D.SQ_VS_CE_P
    ce = poly(sl, uid("wce"), [ax.P(p, min(-np.log(p), 4)) for p in ps], color=RED, width=3)
    sq = poly(sl, uid("wsq"), [ax.P(p, (1 - p) ** 2) for p in ps], color=BLUE, width=3)
    l1 = text(sl, uid("wl1"), ax.X(0.08), ax.Y(3.5), 2.5, 0.35, "cross-entropy  −ln p", size=15, color=RED)
    l2 = text(sl, uid("wl2"), ax.X(0.03), ax.Y(1.0) - 0.45, 2.6, 0.35, "squared  (1 − p)²", size=15, color=BLUE)
    tl.add(sq, "wipe_l", 400, 700).add(l2, "fade", 900, 300).add(ce, "wipe_l", 1300, 700).add(l1, "fade", 1800, 300)
    rows = [["squared error + σ", ("(p − y) · p(1 − p) · x", BLUE)], ["cross-entropy + σ", ("(p − y) · x", RED)]]
    table(sl, tl, 7.5, 1.95, [2.4, 2.8], ["loss", "gradient for w"], rows, size=16, row_h=0.55, at=2200, step=300)
    card(sl, tl, 7.5, 3.7, 5.1, 2.1, "confidently wrong, p = 0.01",
         "squared error: gradient × 0.0099, learning stalls\ncross-entropy: gradient ≈ −1 · x, full push", color=YELLOW, at=3000, size=15, head_size=17)
    caption(sl, tl, "Cross-entropy punishes confident mistakes and keeps the gradient alive.", 3400)


def same_gradient():
    sl, tl = dslide(3, "A neat surprise: the two gradients look the same",
                    """Put the two models side by side. Linear regression with squared error: the gradient for w is the average of prediction minus truth, times x. Logistic regression with cross-entropy: the average of p minus y, times x. Same shape. That's not a coincidence: each loss is the natural partner of its output function, and both come from maximum likelihood, as we'll see next. It also means the update rule you already know works unchanged.""")
    card(sl, tl, 0.85, 1.85, 5.6, 3.9, "linear regression + squared error", "", color=YELLOW, at=300)
    card(sl, tl, 6.9, 1.85, 5.6, 3.9, "logistic regression + cross-entropy", "", color=RED, at=600)
    formula(sl, tl, 1.15, 3.1, 5.2, M("∂", ("L", "i"), "/∂", ("w", "y"), " = 2/", ("n", "i"), " Σ (", ("ŷ", "y"), ("i", "sub"), " − ", ("y", "o"), ("i", "sub"), ") ", ("x", "i"), ("i", "sub")), size=26, at=1000)
    formula(sl, tl, 7.2, 3.1, 5.2, M("∂", ("L", "i"), "/∂", ("w", "y"), " = 1/", ("n", "i"), " Σ (", ("p", "y"), ("i", "sub"), " − ", ("y", "o"), ("i", "sub"), ") ", ("x", "i"), ("i", "sub")), size=26, at=1300)
    t1 = text(sl, uid("sg1"), 1.15, 4.2, 5.0, 1.2, "prediction minus truth, weighted by the input", size=17, color=MUTED)
    t2 = text(sl, uid("sg2"), 7.2, 4.2, 5.0, 1.2, "probability minus truth, weighted by the input", size=17, color=MUTED)
    tl.add(t1, "fade", 1600, 400).add(t2, "fade", 1800, 400)
    caption(sl, tl, "Both: (prediction − truth) × input. The update rule carries over unchanged.", 2200)


def mle():
    sl, tl = dslide(3, "Where both losses come from: maximum likelihood",
                    """The derivation in four lines. If each miss is Gaussian noise with spread sigma, the probability of one measured y is proportional to e to the minus residual squared over 2 sigma squared. Batches are independent, so the likelihood of all the data is the product. Logs turn the product into a sum, and maximising the likelihood is minimising minus the log: a constant times the sum of squared residuals. For yes/no outcomes the model of one batch is p to the y times 1 minus p to the 1 minus y; minus the log is exactly cross-entropy. Pick the noise model; the loss follows.""")
    lines = [
        ("Gaussian misses", M(("P", "i"), "(", ("y", "i"), ("i", "sub"), ") ∝ exp( −(", ("y", "i"), ("i", "sub"), " − ", ("ŷ", "i"), ("i", "sub"), ")² / 2σ² )"), ORANGE),
        ("independent batches", M("likelihood = ∏ ", ("P", "i"), "(", ("y", "i"), ("i", "sub"), ")"), INK),
        ("take −log", M("−log likelihood = Σ (", ("y", "i"), ("i", "sub"), " − ", ("ŷ", "i"), ("i", "sub"), ")² / 2σ² + const   ⇒  squared error"), ORANGE),
        ("coin-flip outcomes", M(("P", "i"), "(", ("y", "i"), ("i", "sub"), ") = ", ("p", "i"), ("y", "sup"), " (1 − ", ("p", "i"), ")", ("1−y", "sup"), "   ⇒  −log = cross-entropy"), RED),
    ]
    y = 1.85
    for i, (lab, runs, col) in enumerate(lines):
        l = text(sl, uid("ml"), 0.85, y + 0.12, 3.0, 0.4, lab, size=15, color=col, bold=True)
        f = text(sl, uid("mf"), 3.9, y, 8.7, 0.7, runs, size=24, font=MATH, color=INK)
        tl.add(l, "fade", 400 + i * 700, 300).add(f, "rise", 500 + i * 700, 450)
        y += 0.98
    caption(sl, tl, "Choose a noise model, take minus the log of the likelihood: the loss falls out.", 3400)


def logistic2d():
    sl, tl = dslide(3, "Two inputs: the boundary is a straight line",
                    """Logistic regression with two process variables: reactor temperature and dissolved oxygen, predicting a runaway or fail. Watch training: the yellow boundary, where p equals one half, rotates and shifts until it separates the two groups as well as a straight line can. The shading is the probability. Because z is w1 x1 plus w2 x2 plus b, the boundary z equals zero is always a straight line: remember that, it's the limitation the next chapter breaks.""")
    gif(sl, tl, "logistic2d.gif", 0.7, 1.55, h=5.1, at=200)
    formula(sl, tl, 7.85, 1.9, 4.8, M(("z", "y"), " = ", ("w", "i"), ("1", "sub"), ("T", "i"), " + ", ("w", "i"), ("2", "sub"), ("DO", "i"), " + ", ("b", "i")), size=28, at=600)
    t = text(sl, uid("l2n"), 7.85, 2.8, 4.7, 2.5, "boundary:  z = 0, a straight line\nshading:  p = σ(z)\nupdate:  same gradient, (p − y)·x", size=17, color=MUTED, line_spacing=1.3)
    tl.add(t, "fade", 1000, 400)
    caption(sl, tl, "However long it trains, a logistic boundary is a straight line.", 1600)


def softmax():
    z = np.array([2.0, 0.5, -1.0])
    p = np.exp(z) / np.exp(z).sum()
    labels = ["on spec", "low titre", "contaminated"]
    cols = [TEAL, YELLOW, RED]
    sl, tl = dslide(3, "More than two outcomes: softmax",
                    f"""Often there are more than two outcomes: on spec, low titre, contaminated. The model produces one score per class, then softmax turns scores into probabilities: exponentiate each score and divide by the total. Scores 2, 0.5 and minus 1 become {p[0]:.2f}, {p[1]:.2f} and {p[2]:.2f}. The loss is still minus log of the probability given to what happened. Keep this slide in mind: an LLM does exactly this, with one score for each of about 100,000 possible next tokens.""")
    formula(sl, tl, 0.85, 1.8, 6.4, M(("p", "i"), ("k", "sub"), " = ", ("e", "i"), ("zₖ", "sup"), " / Σ", ("j", "sub"), " ", ("e", "i"), ("zⱼ", "sup")), size=34, at=300)
    rows = [[(labels[k], cols[k]), (fmt(z[k], 1), YELLOW), f"{np.exp(z[k]):.2f}", (f"{p[k]:.2f}", cols[k])] for k in range(3)]
    rows.append(["total", "", f"{np.exp(z).sum():.2f}", "1.00"])
    table(sl, tl, 0.85, 2.85, [2.4, 1.3, 1.4, 1.4], ["class", "score z", "e^z", "p"], rows, size=17, row_h=0.48, at=800, step=250)
    base = 5.65
    for k in range(3):
        hh = 3.4 * p[k]
        b = rect(sl, uid("smb"), 8.2 + k * 1.5, base - hh, 1.0, hh, fill=cols[k], alpha=0.85, radius=0.05)
        l = text(sl, uid("sml"), 7.9 + k * 1.5, base + 0.08, 1.6, 0.35, labels[k], size=13, color=MUTED, align="c")
        v = text(sl, uid("smv"), 7.9 + k * 1.5, base - hh - 0.4, 1.6, 0.35, f"{p[k]:.2f}", size=16, bold=True, color=cols[k], align="c")
        tl.add(b, "wipe_u", 1800 + k * 250, 500).add(l, "fade", 1800 + k * 250, 300).add(v, "fade", 2200 + k * 250, 300)
    caption(sl, tl, "One score per class, exponentiate, normalise. LLMs do this over every word.", 2700)


def quiz3():
    quiz(3, ["A model gives p = 0.2 to a batch that then fails. What is its cross-entropy loss? (ln 0.2 ≈ −1.61)",
             "w = 2 and b = −6 for lactate. At what lactate is the model undecided (p = 0.5)?",
             "Why does squared error train a sigmoid classifier slowly when it is confidently wrong?",
             "Logistic regression is trained for 10,000 steps on the operating-window data. Why can it still never draw the window?"],
            ["−ln 0.2 ≈ 1.61.", "z = 0 ⇒ x = 6/2 = 3 g/L.",
             "Its gradient carries a factor p(1 − p), which is near zero when p is near 0 or 1, so a confident mistake produces almost no push.",
             "Its boundary z = 0 is always a straight line; a window needs a closed curve. Training longer cannot change the model family."])


# ============================================================================= CH4 detail
def activations():
    sl, tl = dslide(4, "The neuron, and its choice of bend",
                    """The neuron written out. First a weighted sum, z equals w-transpose x plus b: exactly the linear model from chapter one. Then an activation function g bends it. Three common choices. Sigmoid: squashes into 0 to 1, used at the output for probabilities. Tanh: the same S-shape centred on zero. ReLU, max of zero and z: flat, then straight. ReLU is the default inside modern networks because it's cheap and its slope is exactly 1 wherever it's on, so gradients pass through without shrinking.""")
    formula(sl, tl, 0.85, 1.8, 11.5, M(("z", "y"), " = ", ("w", "i"), ("1", "sub"), ("x", "i"), ("1", "sub"), " + ", ("w", "i"), ("2", "sub"), ("x", "i"), ("2", "sub"), " + … + ", ("b", "i"), "        ", ("a", "t"), " = ", ("g", "i"), "(", ("z", "y"), ")"), size=32, at=300)
    zs = np.linspace(-4, 4, 60)
    fns = [("sigmoid", D.sigmoid(zs), (-0.2, 1.2), "1 / (1 + e⁻ᶻ)", "output probabilities"),
           ("tanh", np.tanh(zs), (-1.2, 1.2), "(eᶻ − e⁻ᶻ)/(eᶻ + e⁻ᶻ)", "centred S-curve"),
           ("ReLU", np.maximum(zs, 0), (-0.5, 4.2), "max(0, z)", "default for hidden layers")]
    for i, (nm, ys, yl, f, use) in enumerate(fns):
        x0 = 0.85 + i * 4.05
        c = rect(sl, uid("acb"), x0, 2.75, 3.75, 3.2, fill=PANEL, radius=0.1)
        ax = Ax(x0 + 0.35, 3.3, 3.05, 1.6, (-4, 4), yl)
        seg(sl, uid("ac0"), ax.x0, ax.Y(0), ax.x0 + ax.w, ax.Y(0), FAINT, 1)
        seg(sl, uid("ac1"), ax.X(0), ax.y0, ax.X(0), ax.y0 + ax.h, FAINT, 1)
        p = poly(sl, uid("acp"), [ax.P(a, b) for a, b in zip(zs, ys)], color=[RED, BLUE, YELLOW][i], width=3)
        n = text(sl, uid("acn"), x0 + 0.3, 2.85, 3.2, 0.4, nm, size=20, bold=True, font=HEAD, color=[RED, BLUE, YELLOW][i])
        t = text(sl, uid("act"), x0 + 0.3, 5.0, 3.3, 0.9, [(f, {"font": MATH}), ("\n" + use, {"color": MUTED, "size": 14})], size=15, color=INK)
        tl.add(c, "pop", 700 + i * 300, 400).add(n, "fade", 850 + i * 300, 300).add(p, "wipe_l", 1000 + i * 300, 600).add(t, "fade", 1200 + i * 300, 300)
    caption(sl, tl, "Weighted sum, then a bend. ReLU is the default bend inside networks.", 2300)


def matrix_form():
    sl, tl = dslide(4, "A whole layer at once: matrices, and counting knobs",
                    f"""With 16 hidden neurons we don't write 16 equations. Stack each neuron's weights as a column of a matrix W1, two rows by 16 columns. Then the whole hidden layer is one line: h equals ReLU of x times W1 plus b1. The output neuron takes those 16 numbers and squashes them: p equals sigma of h dot w2 plus b2. Count the knobs: 32 weights and 16 biases in the hidden layer, 16 weights and one bias at the output: {D.NN_PARAMS} in total. GPT-3 has 175 billion, arranged the same way.""")
    formula(sl, tl, 0.85, 1.8, 11.5, M(("h", "y"), " = ReLU( ", ("x", "i"), " ", ("W", "i"), ("1", "sub"), " + ", ("b", "i"), ("1", "sub"), " )          ", ("p", "o"), " = σ( ", ("h", "y"), " · ", ("w", "i"), ("2", "sub"), " + ", ("b", "i"), ("2", "sub"), " )"), size=30, at=300)
    rows = [["x", "1 × 2", "inputs (scaled T, pH)", ""],
            ["W₁, b₁", "2 × 16,  16", "hidden layer", ("32 + 16", YELLOW)],
            ["h", "1 × 16", "hidden activations", ""],
            ["w₂, b₂", "16,  1", "output neuron", ("16 + 1", YELLOW)],
            [("total", INK), "", "", (f"{D.NN_PARAMS} knobs", TEAL)]]
    table(sl, tl, 0.85, 2.85, [2.2, 2.4, 4.4, 2.6], ["symbol", "shape", "role", "parameters"], rows, size=18, row_h=0.52, at=900, step=250, hl_last=True)
    caption(sl, tl, f"A 2–16–1 network has {D.NN_PARAMS} knobs. GPT-3 has 175 billion, same idea.", 2600)


def forward_by_hand():
    x, W1, b1, z1, h = D.NH_X, D.NH_W1, D.NH_B1, D.NH_Z1, D.NH_H
    sl, tl = dslide(4, "Forward pass by hand: a 2–2–1 network",
                    f"""A forward pass with real numbers, on the smallest network that has a hidden layer. One batch with scaled temperature 0.5 and scaled pH minus 1. Hidden neuron 1: 2 times 0.5 plus minus 1 times minus 1 plus 0.1: z equals {z1[0]:.1f}; ReLU keeps it: {h[0]:.1f}. Hidden neuron 2: 1 times 0.5 plus 1.5 times minus 1 plus 0.2: {z1[1]:.1f}; ReLU switches it off: 0. Output: 1.2 times {h[0]:.1f} minus 0.7 times 0 minus 1: z equals {D.NH_Z2:.2f}, and the sigmoid gives p equals {D.NH_P:.2f}. An 82 percent chance this batch is in the window. Notice neuron 2 contributed nothing for this batch: each ReLU is on for some region of the inputs and off elsewhere.""")
    rows = [["hidden 1", f"2·(0.5) + (−1)·(−1) + 0.1", (f"{z1[0]:.1f}", YELLOW), (f"{h[0]:.1f}", TEAL)],
            ["hidden 2", f"1·(0.5) + 1.5·(−1) + 0.2", (fmt(z1[1], 1), YELLOW), ("0  (off)", RED)],
            ["output", f"1.2·({h[0]:.1f}) + (−0.7)·(0) − 1.0", (f"{D.NH_Z2:.2f}", YELLOW), (f"p = σ = {D.NH_P:.2f}", TEAL)]]
    t = text(sl, uid("fhx"), 0.85, 1.75, 11.5, 0.45, [("input: ", {"color": MUTED}), ("x₁ = 0.5 (scaled T),  x₂ = −1.0 (scaled pH)", {})], size=19, color=INK)
    tl.add(t, "fade", 300, 400)
    table(sl, tl, 0.85, 2.45, [1.9, 5.4, 1.8, 2.6], ["neuron", "weighted sum", "z", "after bend"], rows, size=19, row_h=0.65, at=700, step=600)
    card(sl, tl, 0.85, 4.75, 11.7, 1.05, "", "", color=MUTED, at=2600)
    n = text(sl, uid("fhn"), 1.15, 4.95, 11.2, 0.7, "Each ReLU is on in one region of the inputs and off elsewhere: that's how the network carves up the space.", size=17, color=INK)
    tl.add(n, "fade", 2800, 400)
    caption(sl, tl, f"Two inputs in, one probability out: p = {D.NH_P:.2f}.", 3000)


def hidden_lines():
    sl, tl = dslide(4, "What each hidden neuron draws",
                    """Open up the trained network from chapter six and look at what each of the 16 hidden neurons does. Each one is a weighted sum followed by a ReLU, so each switches on along one straight line in the temperature-pH plane: that's what you see being drawn, orange lines for neurons that vote 'inside', blue for 'outside'. No single line is a window. The output neuron adds up the 16 bent planes, and the 0.5 contour of that sum is the yellow polygon. A curved boundary, built from straight pieces.""")
    gif(sl, tl, "hidden_lines.gif", 0.7, 1.55, h=5.15, at=200)
    t = text(sl, uid("hln"), 7.5, 1.9, 5.1, 3.0, "16 neurons → 16 straight lines\n\nOrange: votes “inside the window”\nBlue: votes “outside”\n\nThe output adds them up and squashes: a polygon wraps the window.", size=17, color=INK, line_spacing=1.15)
    tl.add(t, "fade", 800, 500)
    caption(sl, tl, "Straight pieces, bent and added, make a curved boundary.", 1500)


def haldane(stage):
    notes = {0: f"""Networks don't just classify. Here's a regression problem every bioprocess engineer knows: substrate-inhibited growth, the Haldane equation, where growth rises with substrate and then falls as substrate becomes toxic. We give a 1-16-1 ReLU network the 26 measured growth rates and nothing else: no equation. Watch it train: the fit is a chain of straight segments, one kink per active neuron, and gradient descent moves the kinks until the curve hugs the data.""",
             1: """How many neurons do you need? One neuron gives one bend: a hockey stick. Two, four, eight: more kinks, better fit. This is the universal approximation idea: with enough hidden neurons, a one-hidden-layer network can approximate any smooth curve as closely as you like. It says nothing about how much data you need, or what happens outside the data. We'll come back to both."""}[stage]
    sl, tl = dslide(4, ["A network learns Haldane kinetics from data alone", "More neurons, more bends, a closer fit"][stage], notes, morph=1100)
    gif(sl, tl, "haldane_fit.gif" if stage == 0 else "widths.gif", 0.7, 1.6, w=7.6, at=200)
    if stage == 0:
        formula(sl, tl, 8.6, 1.9, 4.2, M(("μ", "i"), " = ", ("μ", "i"), ("max", "sub"), ("S", "i"), " / (", ("K", "i"), ("s", "sub"), " + ", ("S", "i"), " + ", ("S", "i"), "²/", ("K", "i"), ("i", "sub"), ")"), size=22, at=600)
        t = text(sl, uid("hd"), 8.6, 2.7, 4.1, 3.0, "true curve used to simulate the data (dashed)\n\nThe network never sees this equation: only 26 noisy points.\n\n1–16–1 ReLU network, 49 knobs, plain gradient descent on squared error", size=15, color=MUTED, line_spacing=1.1)
        tl.add(t, "fade", 900, 400)
        caption(sl, tl, "No kinetic model given. The network finds the shape from 26 points.", 1400)
    else:
        t = text(sl, uid("wd"), 8.6, 1.9, 4.1, 3.6, "1 neuron: one bend\n2 neurons: two bends\n…\n16 neurons: a smooth-looking curve\n\nUniversal approximation: enough neurons can fit any smooth curve. It says nothing about extrapolation.", size=16, color=INK, line_spacing=1.15)
        tl.add(t, "fade", 600, 400)
        caption(sl, tl, "Each hidden neuron adds one kink. Enough kinks fit any smooth curve.", 1200)


def quiz4():
    quiz(4, ["How many parameters does a 3–8–1 network have (one hidden layer, biases included)?",
             "If you remove every activation function, what does a 10-layer network become?",
             "A hidden ReLU neuron receives z = −0.7. What does it output, and what gradient does it pass back?",
             "Why is ReLU, not sigmoid, the usual choice inside hidden layers?"],
            ["3·8 + 8 + 8 + 1 = 41.", "A single linear model: a product of matrices is one matrix.",
             "Output 0, and gradient 0: it is off for this example, so its incoming weights are not updated by it.",
             "Cheap to compute, and its slope is 1 wherever it is on, so gradients pass through many layers without shrinking (sigmoid's slope is at most 0.25)."])


# ============================================================================= CH5 detail
def transfer_functions():
    sl, tl = dslide(5, "You already know backprop: gains in series",
                    """Here's the analogy for anyone who's done process control. Three units in series with steady-state gains K1, K2, K3. Nudge the input by delta u: the output moves by K1 times K2 times K3 times delta u. The overall sensitivity is the product of the local sensitivities. A neural network is a flowsheet of simple units, and the gradient of the loss with respect to a weight deep inside is the product of the local gains between that weight and the loss. That product is the chain rule, and computing all of them efficiently is backpropagation.""")
    xs = [1.35, 4.65, 7.95]
    ks = ["K₁ = 2", "K₂ = 0.5", "K₃ = 3"]
    names = ["heater", "reactor", "separator"]
    y0 = 2.4
    l0 = seg(sl, uid("tfl"), 0.6, y0 + 0.6, xs[0], y0 + 0.6, MUTED, 2.5)
    tl.add(l0, "wipe_l", 300, 300)
    for i, (x, k, n) in enumerate(zip(xs, ks, names)):
        b = rect(sl, uid("tfb"), x, y0, 2.3, 1.2, fill=PANEL, line=YELLOW, line_w=2, radius=0.1)
        t = text(sl, uid("tft"), x, y0 + 0.15, 2.3, 0.5, k, size=22, font=MATH, color=YELLOW, align="c")
        n_ = text(sl, uid("tfn"), x, y0 + 0.68, 2.3, 0.4, n, size=14, color=MUTED, align="c")
        nxt = xs[i + 1] if i < 2 else 11.6
        l = seg(sl, uid("tfl"), x + 2.3, y0 + 0.6, nxt, y0 + 0.6, MUTED, 2.5)
        tl.add(b, "pop", 500 + i * 450, 400).add(t, "fade", 650 + i * 450, 300).add(n_, "fade", 650 + i * 450, 300).add(l, "wipe_l", 800 + i * 450, 300)
    a = text(sl, uid("tfa"), 0.4, y0 + 1.35, 1.8, 0.4, "Δu", size=20, font=MATH, color=INK)
    b_ = text(sl, uid("tfb"), 10.7, y0 + 1.35, 2.2, 0.4, "Δy = 3 Δu", size=20, font=MATH, color=TEAL, align="r")
    tl.add(a, "fade", 300, 300).add(b_, "fade", 2000, 400)
    formula(sl, tl, 0.85, 4.6, 11.6, M("∂", ("y", "i"), "/∂", ("u", "i"), " = ", ("K", "y"), ("1", "sub"), " · ", ("K", "y"), ("2", "sub"), " · ", ("K", "y"), ("3", "sub"), " = 2 × 0.5 × 3 = 3"), size=32, at=2300, align="c")
    caption(sl, tl, "Overall sensitivity = product of local gains. That's the chain rule.", 2800)


def chain_table():
    B = D.BP
    sl, tl = dslide(5, "Every local gain in the worked example",
                    """The backward pass from the worked example, as a table of local gains. Each row is one link: how much the next quantity moves per unit change of the previous one. Multiply along the path from the loss back to a weight and you get that weight's gradient. Notice the reuse: the gradient arriving at h, minus 4.5, is needed for w1 and would be needed for anything further left. Backpropagation computes each of these once, right to left, and reuses them.""")
    rows = [[("∂L/∂ŷ", RED), "L = (ŷ − y)²", "2(ŷ − y)", f"2(1.5 − 3) = {fmt(B['dyhat'], 0)}"],
            [("∂ŷ/∂w₂", RED), "ŷ = w₂ h", "h", f"{B['h']:g}"],
            [("∂ŷ/∂h", RED), "ŷ = w₂ h", "w₂", f"{B['w2']:g}"],
            [("∂h/∂z", RED), "h = ReLU(z)", "1 if z > 0, else 0", "1  (z = 1 > 0)"],
            [("∂z/∂w₁", RED), "z = w₁ x", "x", f"{B['x']:g}"]]
    table(sl, tl, 0.85, 1.75, [2.4, 2.9, 3.2, 3.2], ["link", "unit", "local gain", "value"], rows, size=18, row_h=0.52, at=400, step=300)
    formula(sl, tl, 0.85, 4.75, 11.6, M("∂", ("L", "i"), "/∂", ("w", "i"), ("2", "sub"), " = (−3)(1) = ", ("−3", "r"), "          ∂", ("L", "i"), "/∂", ("w", "i"), ("1", "sub"), " = (−3)(1.5)(1)(2) = ", ("−9", "r")), size=26, at=2300, h=0.6)
    caption(sl, tl, "A gradient = the product of local gains along the path back to the loss.", 2800)


def why_backwards():
    sl, tl = dslide(5, "Why go backwards? Because it's cheap",
                    f"""Why not just wiggle each weight and see what happens to the loss? That's the finite-difference method: two forward passes per weight. For our {D.NN_PARAMS}-knob network that's {2 * D.NN_PARAMS} passes per step, fine. For GPT-3 it's 350 billion forward passes for a single step: impossible. Backpropagation gets every gradient from one forward pass and one backward pass, roughly the cost of two or three forward passes, no matter how many weights there are. That's the whole reason deep learning is practical.""")
    rows = [["finite differences", f"2 per weight", f"{2 * D.NN_PARAMS}", ("350 000 000 000", RED)],
            ["backpropagation", "1 forward + 1 backward", "≈ 3", ("≈ 3", TEAL)]]
    table(sl, tl, 0.85, 1.9, [3.1, 3.3, 2.4, 3.0], ["method", "forward passes needed", f"{D.NN_PARAMS}-knob net", "GPT-3 (175 B)"], rows, size=19, row_h=0.62, at=400, step=600)
    card(sl, tl, 0.85, 3.85, 11.7, 1.9, "The trick", "Going right to left, each local gain is computed once and the products are reused for every weight upstream. Cost grows with the size of the network, not with the number of gradients you want.", color=YELLOW, at=1800, size=17)
    caption(sl, tl, "Backprop: every gradient for about the price of three forward passes.", 2400)


def grad_check():
    sl, tl = dslide(5, "Trust, but verify: a gradient check",
                    f"""How do you know your backprop is right? Check one gradient numerically. Nudge w1 up by a thousandth, compute the loss; nudge it down, compute again; the difference divided by 0.002 is the slope. Here: {D.GC_W1:.4f}, matching the minus 9 from backprop. Same for w2: {D.GC_W2:.4f}, versus minus 3. Every deep-learning library is tested this way. Too slow to train with, perfect for checking.""")
    formula(sl, tl, 0.85, 1.85, 11.6, M("∂", ("L", "i"), "/∂", ("w", "i"), " ≈ [ ", ("L", "i"), "(", ("w", "i"), " + ε) − ", ("L", "i"), "(", ("w", "i"), " − ε) ] / 2ε"), size=32, at=300)
    L_p = D.bp_loss(0.5 + D.GC_EPS, 1.5); L_m = D.bp_loss(0.5 - D.GC_EPS, 1.5)
    rows = [["L(w₁ = 0.501)", f"{L_p:.6f}"], ["L(w₁ = 0.499)", f"{L_m:.6f}"],
            ["difference / 0.002", (f"{D.GC_W1:.4f}", YELLOW)], ["backprop said", ("−9", RED)],
            ["same check for w₂", (f"{D.GC_W2:.4f}  vs  −3", YELLOW)]]
    table(sl, tl, 0.85, 3.0, [4.0, 3.6], ["", "value"], rows, size=18, row_h=0.5, at=800, step=300)
    card(sl, tl, 8.9, 3.0, 3.65, 2.8, "use it for", "testing code, never for training: it costs 2 forward passes per weight", color=TEAL, at=2400, size=16, head_size=18)
    caption(sl, tl, "Numerical slope ≈ −9.000. Backprop was right.", 2800)


def backprop_code():
    sl, tl = dslide(5, "Backprop for the 2–16–1 network, in ten lines",
                    """This is the actual code that trained the network you'll see in the next chapter: it's in data.py. Forward: weighted sum, ReLU, weighted sum, sigmoid. Backward: for sigmoid with cross-entropy the gradient at the output score is simply p minus y. Then each line is one link of the chain rule: multiply by the hidden activations to get the output weights' gradient, by the output weights to send the gradient back into the hidden layer, by the ReLU's on-off mask, and finally by the inputs. Then step every weight downhill. PyTorch does exactly this for you, automatically.""")
    code(sl, tl, 0.85, 1.75, 11.7, 4.15, [
        "z1 = X @ W1 + b1;  h = relu(z1)              # forward: hidden layer",
        "p  = sigmoid(h @ W2 + b2)                    # forward: output probability",
        "d_out = (p - y) / n                          # dL/dz2 for sigmoid + cross-entropy",
        "dW2 = h.T @ d_out;  db2 = d_out.sum(0)       # local gain: h",
        "dh  = d_out @ W2.T                           # local gain: W2 (send blame back)",
        "dz1 = dh * (z1 > 0)                          # local gain: ReLU on/off mask",
        "dW1 = X.T @ dz1;  db1 = dz1.sum(0)           # local gain: inputs x",
        "for P, G in [(W1, dW1), (b1, db1), (W2, dW2), (b2, db2)]:",
        "    P -= lr * G                              # one step downhill"], size=15, at=300)
    caption(sl, tl, "Each backward line is one link of the chain rule. This trained chapter 6.", 1800)


def quiz5():
    quiz(5, ["Three units in series have local gains 4, −0.5 and 2. What is the overall sensitivity?",
             "In the worked example, why is ∂L/∂w₁ larger in size than ∂L/∂w₂?",
             "A ReLU neuron was off (z < 0) for an example. What gradient flows to its incoming weights from that example?",
             "Your backprop gives −2.40 but a finite-difference check gives −1.20. What do you conclude?"],
            ["4 × (−0.5) × 2 = −4.",
             "Its path to the loss multiplies by w₂ = 1.5 and by x = 2, so the blame is amplified on the way back.",
             "Zero: the ReLU's local gain is 0, so the chain is cut for that example.",
             "There is a bug in the backprop code (often a missing factor of 2 or a wrong transpose). The numerical check is the reference."])


# ============================================================================= CH6 detail
def training_loop():
    sl, tl = dslide(6, "The full training loop",
                    """Everything in one loop, the way it's really done. Split the data into training and validation, as in Lecture 1. Standardise the inputs using the training set's means and spreads. Then for each epoch, shuffle, and for each mini-batch: forward pass, loss, backward pass, update. After each epoch, measure the validation loss, keep the weights from the best epoch, and stop when it hasn't improved for a while. That's it. Lecture 3 is about making this loop work for networks with dozens of layers.""")
    code(sl, tl, 0.85, 1.75, 7.6, 4.15, [
        "split data: train / validation / test",
        "standardise inputs (train mean, train sd)",
        "initialise weights: small random numbers",
        "for epoch in range(max_epochs):",
        "    shuffle training data",
        "    for batch in mini_batches(train):",
        "        p = forward(batch)              # predict",
        "        L = loss(p, y)                  # how wrong",
        "        grads = backward(L)             # backprop",
        "        weights -= lr * grads           # step",
        "    track validation loss; keep best weights",
        "    stop if no improvement for N epochs"], size=14, at=300)
    card(sl, tl, 8.8, 1.75, 3.75, 1.2, "model", "the network", color=YELLOW, at=1600, size=15, head_size=17)
    card(sl, tl, 8.8, 3.1, 3.75, 1.2, "loss", "squared error or cross-entropy", color=RED, at=1800, size=15, head_size=17)
    card(sl, tl, 8.8, 4.45, 3.75, 1.45, "learning", "mini-batch gradient descent, gradients from backprop", color=TEAL, at=2000, size=15, head_size=17)


def overfit():
    import gifs
    tr, va = gifs.overfit_numbers()
    best = int(np.argmin(va)) + 1
    sl, tl = dslide(6, "Too flexible, too few batches: overfitting returns",
                    f"""Lecture 1's warning applies with full force. Fourteen expensive batches, and a model that gets more flexible: here a polynomial whose degree grows from 1 to 12, so the number of knobs grows from 2 to 13. Training error, orange, falls every time. Validation error on new batches, green, is lowest at degree {best} and then climbs as the curve starts threading every noisy point. Neural networks behave the same way: more neurons or more epochs keep lowering training loss. The cures: more data, a smaller model, regularisation in Lecture 3, and early stopping: watch validation loss during training and keep the best epoch.""")
    gif(sl, tl, "overfit.gif", 0.6, 1.6, w=8.6, at=200)
    rows = [["training error", ("always falls", ORANGE)], ["validation error", (f"best at degree {best}", TEAL)], ["then", ("climbs: overfitting", RED)]]
    table(sl, tl, 9.4, 1.9, [1.9, 1.8], ["", ""], rows, size=15, row_h=0.55, at=900, step=250)
    t = text(sl, uid("ofn"), 9.4, 3.85, 3.3, 2.0, "Cures: more batches, a smaller model, regularisation (Lecture 3), early stopping.", size=15, color=MUTED)
    tl.add(t, "fade", 1800, 400)
    caption(sl, tl, "Judge by validation error. Stop where it bottoms out.", 2100)


def extrapolation():
    sl, tl = dslide(6, "Process engineers: never trust it outside the data",
                    """The most important warning for using these models in a plant. The network learned growth kinetics between 0 and 20 grams per litre, and it's good there. Ask it about 30 or 40 and it just continues its last straight segment: here it predicts growth rates that keep falling past zero, nonsense, while the true kinetics level off. A mechanistic model carries physics outside the data; a neural network carries nothing. Keep predictions inside the operating range you trained on, and flag inputs that fall outside it.""")
    p = picture(sl, uid("ex"), os.path.join(GIFS, "extrapolate.png"), 0.7, 1.6, 7.7, 7.7 * 4.2 / 7.0)
    tl.add(p, "fade", 200, 500)
    card(sl, tl, 8.9, 1.8, 3.7, 1.9, "inside the data", "interpolation: usually fine", color=TEAL, at=900, size=16, head_size=18)
    card(sl, tl, 8.9, 3.9, 3.7, 1.95, "outside the data", "extrapolation: no physics, no guarantee", color=RED, at=1200, size=16, head_size=18)
    caption(sl, tl, "Inside the training range: useful. Outside it: a guess with no physics.", 1700)


def troubleshoot():
    sl, tl = dslide(6, "When training goes wrong: symptom, cause, fix",
                    """A practical table to keep. Loss explodes or turns into NaN: learning rate too high, lower it. Loss falls painfully slowly with zigzags: features on different scales, standardise them. Training loss low but validation high: overfitting, get more data, shrink the model, stop early. Loss stuck from the start: learning rate too low, or dead ReLUs from a bad start. Great validation score but nonsense in the plant: you're extrapolating outside the training range, or the process has drifted since the data were collected.""")
    rows = [["loss explodes / NaN", "learning rate too high", ("lower η (try ÷3)", TEAL)],
            ["slow, zig-zag progress", "features on different scales", ("standardise inputs", TEAL)],
            ["train ↓, validation ↑", "overfitting", ("more data, smaller net, early stop", TEAL)],
            ["loss stuck from step 1", "η too low, or dead ReLUs", ("raise η, re-initialise", TEAL)],
            ["fine on paper, wrong in the plant", "extrapolation or process drift", ("check input ranges, retrain", TEAL)]]
    table(sl, tl, 0.85, 1.8, [3.9, 3.9, 3.9], ["symptom", "likely cause", "fix"], rows, size=17, row_h=0.62, at=400, step=300)


# ============================================================================= WRAP detail
def compare_methods():
    sl, tl = dslide(7, "Today's three models, side by side",
                    """The whole lecture in one table. Linear regression: a straight line or plane, squared error, solvable directly or by gradient descent. Logistic regression: a line squashed into a probability, cross-entropy, gradient descent; its boundary is always straight. A neural network: stacked weighted sums with bends, either loss, gradient descent with backprop computing the gradients; its boundary can take any shape. Read the last column top to bottom: the learning rule never changed.""")
    rows = [["linear regression", "ŷ = w·x + b", "squared error", "closed form or gradient descent"],
            ["logistic regression", "p = σ(w·x + b)", "cross-entropy", "gradient descent"],
            ["neural network", "stacked sums + bends", "either", ("gradient descent + backprop", YELLOW)],
            [("Lecture 1: trees", MUTED), ("boxes", MUTED), ("impurity, log-loss", MUTED), ("search, averaging, boosting", MUTED)]]
    table(sl, tl, 0.85, 1.8, [2.9, 3.0, 2.4, 3.4], ["method", "model", "loss", "learning"], rows, size=17, row_h=0.62, at=400, step=300)
    caption(sl, tl, "Read the last column: the learning rule never changed.", 2000)


def glossary():
    sl, tl = dslide(7, "Glossary of today's terms",
                    """A one-page glossary to keep. Worth leaving on screen during questions.""")
    left = [("Weight, bias", "the knobs training adjusts"), ("Residual", "y − ŷ, what is still wrong"),
            ("MSE", "mean squared error, regression loss"), ("Cross-entropy", "average surprise −ln p (classifiers)"),
            ("Logit", "raw score z before the sigmoid"), ("Softmax", "scores → probabilities over many classes"),
            ("Gradient", "slopes of the loss, one per knob")]
    right = [("Learning rate η", "step size; too big oscillates"), ("Mini-batch, epoch", "examples per step / one pass"),
             ("Standardise", "(x − mean)/sd before training"), ("Activation", "the bend: ReLU, sigmoid, tanh"),
             ("Backpropagation", "chain rule, right to left, reused"), ("Early stopping", "keep the best validation epoch"),
             ("Extrapolation", "predicting outside the training range")]
    for col, items in enumerate([left, right]):
        y = 1.8
        for i, (a, b) in enumerate(items):
            t = text(sl, uid("gl"), 0.85 + col * 6.1, y, 5.8, 0.6, [(a, {"bold": True, "color": YELLOW}), ("  —  " + b, {})], size=18, color=INK)
            tl.add(t, "fade", 300 + (col * 7 + i) * 90, 300)
            y += 0.7


def hands_on():
    sl, tl = dslide(7, "Hands-on: today in a few lines of Python",
                    """Everything today in a dozen lines of PyTorch. A 2-16-1 network, the binary cross-entropy loss, plain stochastic gradient descent. The loop is the one from the training slide: zero the old gradients, forward, loss, backward, step. loss.backward() is backpropagation: PyTorch records the forward pass and applies the chain rule for you. Try the exercises on the right with the operating-window data.""")
    code(sl, tl, 0.85, 1.75, 7.6, 4.15, [
        "import torch, torch.nn as nn",
        "X = torch.tensor(X_scaled, dtype=torch.float32)",
        "y = torch.tensor(y, dtype=torch.float32)[:, None]",
        "net = nn.Sequential(nn.Linear(2, 16), nn.ReLU(),",
        "                    nn.Linear(16, 1))",
        "loss_fn = nn.BCEWithLogitsLoss()   # sigmoid + CE",
        "opt = torch.optim.SGD(net.parameters(), lr=0.3)",
        "for epoch in range(3000):",
        "    opt.zero_grad()",
        "    loss = loss_fn(net(X), y)      # forward + loss",
        "    loss.backward()                # backprop",
        "    opt.step()                     # w -= lr * grad"], size=14, at=300)
    ex = ["Change lr to 3.0. What happens to the loss?", "Skip the scaling of T and pH. How many more epochs?",
          "Use 2 hidden neurons instead of 16. Can it draw the window?", "Remove nn.ReLU(). What boundary do you get?"]
    for i, e in enumerate(ex):
        n = text(sl, uid("hon"), 8.8, 1.8 + i * 1.05, 0.5, 0.4, f"({i + 1})", size=16, bold=True, color=CRIMSON)
        t = text(sl, uid("hot"), 9.35, 1.8 + i * 1.05, 3.3, 1.0, e, size=15, color=INK)
        tl.add(n, "fade", 1600 + i * 250, 300).add(t, "fade", 1650 + i * 250, 300)


def reading():
    sl, tl = dslide(7, "Further reading",
                    """Where to go deeper. Goodfellow, Bengio and Courville chapters 4 to 6 cover exactly today's material, free online. Nielsen's online book is the gentlest derivation of backprop. 3Blue1Brown's neural-network videos are the visual companion to this lecture. Karpathy's micrograd builds backprop from scratch in about 100 lines of Python. And Rumelhart, Hinton and Williams 1986 is the paper that made backprop famous. For the chemical-engineering angle, Venkatasubramanian's 2019 perspective is a good read on where these methods fit in our field.""")
    rows = [["Goodfellow, Bengio & Courville (2016)", "Deep Learning, ch. 4–6: optimisation, regression, networks (free online)"],
            ["Nielsen (2015)", "Neural Networks and Deep Learning, ch. 1–2: the gentlest backprop derivation"],
            ["3Blue1Brown", "Neural networks video series: the visual companion to today"],
            ["Karpathy, micrograd", "backprop from scratch in ~100 lines of Python"],
            ["Rumelhart, Hinton & Williams (1986)", "Learning representations by back-propagating errors. Nature 323"],
            ["Venkatasubramanian (2019)", "The promise of artificial intelligence in chemical engineering. AIChE J. 65"]]
    table(sl, tl, 0.85, 1.75, [4.4, 7.3], ["source", "what it is"], rows, size=16, row_h=0.62, at=300, step=150)
