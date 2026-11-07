"""Lecture 2: Learning by gradients. The storyboard.

Run:  python build_lecture2.py   (needs python-pptx, numpy, matplotlib, lxml, pillow)

scenes_core.py   the cinematic "story" scenes (Morph-driven: ball, lines, neurons, backprop)
scenes_detail.py the worked-detail scenes (tables, formulas, ChemE examples, quizzes, glossary)
gifs.py          the matplotlib animations embedded in the detail scenes (run it first if assets/gifs is empty)
"""
import os
import data as D
import scenes_core as C
import scenes_detail as S
from scenes_core import SCENES, CHAPTERS, W, YT_RED, MUTED, FAINT, HEAD, prs, OUT
from motion import rect, dot, text, set_transition


def build():
    for s in range(4):
        C.cold_open(s)
    C.title_sting()
    C.three_questions(0)
    S.bridge()
    C.three_questions(1)
    S.solver()
    S.vocab()

    C.chapter_card(1, "A line that learns", "linear regression and squared error",
                 """Chapter one. The simplest model there is, and the place where the idea of a loss becomes something you can see.""")
    for st in ["data", "guess", "miss", "square", "fit0", "fit1", "fit2"]:
        C.ch1_scene(st)
    S.mse_formula()
    S.h4_by_hand()
    S.closed_form()
    for st in range(3):
        S.arrhenius(st)
    S.plane()
    S.quiz1()

    C.chapter_card(2, "Rolling downhill", "gradient descent, learning rates, mini-batches",
                 """Chapter two. We got that best line by nudging. Now: exactly how does the computer know which way to nudge?""")
    C.ch2_split("split")
    C.ch2_split("slope")
    S.grad_formula()
    C.ch2_split("rule")
    for k in range(1, len(D.CH2_WS)):
        C.ch2_split("step", k)
    S.gd_by_hand()
    S.gd_algorithm()
    C.lr_trio()
    S.lr_response()
    S.bumpy()
    S.scaling()
    C.ch2_landscape("full")
    C.ch2_landscape("sgd")
    S.sgd_detail()
    S.quiz2()

    C.chapter_card(3, "Yes or no?", "logistic regression and cross-entropy",
                 """Chapter three. Not every answer is a number. Sometimes it's yes or no.""")
    for st in ["line", "sig", "wrong", "right"]:
        C.log_scene(st)
    S.sigmoid_table()
    S.ce_formula()
    S.ce_by_hand()
    S.why_not_mse()
    S.same_gradient()
    C.loss_origins()
    S.mle()
    S.logistic2d()
    S.softmax()
    S.quiz3()

    C.chapter_card(4, "Stacking neurons", "from one neuron to a network, and why the bend matters",
                 """Chapter four. What happens when one straight boundary is not enough.""")
    C.nn_window()
    S.activations()
    C.neuron_scene("one")
    C.neuron_scene("net")
    S.matrix_form()
    S.forward_by_hand()
    S.hidden_lines()
    C.nonlin_scene("parts")
    C.nonlin_scene("sum")
    S.haldane(0)
    S.haldane(1)
    S.quiz4()

    C.chapter_card(5, "Blame flows backwards", "backpropagation, worked by hand",
                 """Chapter five. Gradient descent needs the slope for every single weight. Backpropagation is how we get all of them in one sweep.""")
    S.transfer_functions()
    for st in ["fwd", "bwd", "upd"]:
        C.bp_scene(st)
    S.chain_table()
    S.why_backwards()
    S.grad_check()
    S.backprop_code()
    S.quiz5()

    C.chapter_card(6, "Watch it learn", "a small network finds the operating window",
                 """Chapter six. Everything at once, on real numbers.""")
    for k in range(len(D.NN_SNAPS)):
        C.train_frame(k)
    S.training_loop()
    S.overfit()
    S.extrapolation()
    S.troubleshoot()

    C.three_questions(2)
    S.compare_methods()
    S.glossary()
    C.quiz()
    S.hands_on()
    S.reading()
    C.end_screen()

    # ---------------------------------------------------------------- overlays: scrubber + watermark
    content = [i for i, s in enumerate(SCENES) if s["scrub"]]
    counts = {c: sum(1 for i in content if SCENES[i]["chapter"] == c) for c in range(len(CHAPTERS))}
    x0, x1, gap = 0.55, W - 0.55, 0.06
    total = sum(counts.values())
    seg_x = {}
    x = x0
    for c in range(len(CHAPTERS)):
        wseg = (x1 - x0 - gap * (len(CHAPTERS) - 1)) * counts[c] / total
        seg_x[c] = (x, x + wseg)
        x += wseg + gap
    pos = 0
    for i, s in enumerate(SCENES):
        sl = s["slide"]
        if s["scrub"]:
            pos += 1
            frac_x = x0 + (x1 - x0) * pos / total
            yb = 7.2
            for c in range(len(CHAPTERS)):
                a, b = seg_x[c]
                rect(sl, f"!!trk{c}", a, yb, b - a, 0.05, fill="FFFFFF", alpha=0.18)
                played = max(0.0, min(b, frac_x) - a)
                rect(sl, f"!!ply{c}", a, yb, max(played, 0.001), 0.05, fill=YT_RED, alpha=1.0 if played > 0.002 else 0.0)
            dot(sl, "!!head", frac_x, yb + 0.025, 0.075, YT_RED)
            ch = s["chapter"]
            label = "Intro" if ch == 0 else (f"{ch:02d}  ·  {CHAPTERS[ch]}" if ch < 7 else CHAPTERS[ch])
            text(sl, "!!chap", 0.55, 6.86, 6, 0.3, label, size=12, color=MUTED, bold=True, font=HEAD, spacing=1)
            text(sl, "!!bug", W - 4.55, 6.86, 4.0, 0.3, "LECTURE 2  ·  LEARNING BY GRADIENTS", size=10, color=FAINT,
                 bold=True, font=HEAD, spacing=2, align="r")
        set_transition(sl, "morph" if i > 0 else "fade", s["morph"], s["advance"])
        s["tl"].apply()
    prs.core_properties.title = "Lecture 2 - Learning by gradients"
    prs.core_properties.author = "Paras Koundal"
    prs.save(OUT)
    print("wrote", OUT, "slides:", len(SCENES))





if __name__ == "__main__":
    build()
