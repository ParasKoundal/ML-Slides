"""
gifs.py - the animated GIFs embedded in Lecture 2 (they play by themselves in PowerPoint's slideshow).

    python gifs.py          # writes assets/gifs/*.gif

Every frame is computed from the real algorithm in data.py. Dark theme to match the deck.
Each function is independent, so you can re-render one animation at a time:
    python -c "import gifs; gifs.g_scaling()"
"""
import os
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.colors import LinearSegmentedColormap

import data as D

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "assets", "gifs")
os.makedirs(OUT, exist_ok=True)

BG, PANEL, INK, MUTED, FAINT = "#0E1116", "#171B23", "#F2F0EA", "#8B93A1", "#3A414D"
ORANGE, BLUE, YELLOW, RED, TEAL = "#F4A259", "#5BA4E6", "#FFD166", "#EF476F", "#3DDC97"
DPI = 110

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 12, "text.color": INK, "axes.labelcolor": MUTED,
    "axes.edgecolor": FAINT, "xtick.color": MUTED, "ytick.color": MUTED, "axes.facecolor": BG,
    "figure.facecolor": BG, "savefig.facecolor": BG, "axes.spines.top": False, "axes.spines.right": False,
})


def _save(anim, name, fps=15):
    path = os.path.join(OUT, name + ".gif")
    anim.save(path, writer=PillowWriter(fps=fps), dpi=DPI)
    plt.close("all")
    print("wrote", path)
    return path


def _hold(n_frames, hold_end=20):
    """frame index sequence that holds on the last frame (so the loop pauses before restarting)."""
    return list(range(n_frames)) + [n_frames - 1] * hold_end


# ----------------------------------------------------------------------------- 1. regression plane in 3-D
def g_plane():
    fig = plt.figure(figsize=(6.0, 4.6))
    ax = fig.add_axes([0, 0, 1, 1], projection="3d")
    ax.set_facecolor(BG)
    for a in (ax.xaxis, ax.yaxis, ax.zaxis):
        a.set_pane_color((0.09, 0.105, 0.135, 1.0))
        a._axinfo["grid"]["color"] = (0.23, 0.25, 0.3, 1)
    ax.scatter(D.PL_F, D.PL_T, D.PL_Y, c=ORANGE, s=22, depthshade=False)
    F, T = np.meshgrid(np.linspace(0.5, 3, 12), np.linspace(30, 38, 12))
    Z = D.PL_W[0] * F + D.PL_W[1] * T + D.PL_W[2]
    ax.plot_surface(F, T, Z, color=YELLOW, alpha=0.28, linewidth=0, shade=False)
    ax.plot_wireframe(F, T, Z, color=YELLOW, linewidth=0.5, alpha=0.6, rstride=2, cstride=2)
    ax.set_xlabel("feed rate (g/L/h)", labelpad=6)
    ax.set_ylabel("temperature (°C)", labelpad=6)
    ax.set_zlabel("titre (g/L)", labelpad=4)
    ax.tick_params(labelsize=9)

    def upd(i):
        ax.view_init(elev=18 + 6 * np.sin(i / 90 * 2 * np.pi), azim=-60 + i * 4)
        return []

    return _save(FuncAnimation(fig, upd, frames=90), "plane3d", fps=15)


# ----------------------------------------------------------------------------- 2. feature scaling: zig-zag vs straight
def _contour_panel(ax, X, wstar, half, title, path, col):
    w1 = np.linspace(wstar[0] - half, wstar[0] + half, 220)
    w2 = np.linspace(wstar[1] - half, wstar[1] + half, 220)
    A, B = np.meshgrid(w1, w2)
    L = np.array([[D.sc_loss(X, np.array([a, b])) for a in w1] for b in w2])
    lv = np.geomspace(L.min() * 1.3 + 1e-4, L.max(), 16)
    ax.contour(A, B, L, levels=lv, colors="#5d6878", linewidths=0.8)
    ax.plot(*wstar, marker="x", color=TEAL, ms=10, mew=2.5)
    ax.set_xlim(w1[0], w1[-1]); ax.set_ylim(w2[0], w2[-1])
    ax.set_aspect("equal")
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(title, color=INK, fontsize=13, loc="left")
    line, = ax.plot([], [], "-", color=col, lw=1.6)
    dots, = ax.plot([], [], "o", color=col, ms=4)
    return line, dots


def g_scaling():
    fig, axs = plt.subplots(1, 2, figsize=(9.0, 4.4))
    fig.subplots_adjust(left=0.03, right=0.97, top=0.88, bottom=0.12, wspace=0.08)
    l1, d1 = _contour_panel(axs[0], D.SC_RAW, D.SC_WRAW, 3.2, "raw units: °C and pH", D.SC_PATH_RAW, RED)
    l2, d2 = _contour_panel(axs[1], D.SC_STD, D.SC_WSTD, 3.2, "standardised: both in std units", D.SC_PATH_STD, TEAL)
    axs[0].set_xlabel("weight on temperature"); axs[0].set_ylabel("weight on pH")
    axs[1].set_xlabel("weight on temperature"); axs[1].set_ylabel("weight on pH")
    txt = fig.text(0.5, 0.02, "", ha="center", color=MUTED, fontsize=12)
    P1, P2 = D.SC_PATH_RAW, D.SC_PATH_STD

    def upd(i):
        k = min(i, len(P1) - 1)
        l1.set_data(P1[:k + 1, 0], P1[:k + 1, 1]); d1.set_data(P1[:k + 1, 0], P1[:k + 1, 1])
        l2.set_data(P2[:k + 1, 0], P2[:k + 1, 1]); d2.set_data(P2[:k + 1, 0], P2[:k + 1, 1])
        txt.set_text(f"step {k}")
        return []

    return _save(FuncAnimation(fig, upd, frames=_hold(len(P1), 25)), "scaling", fps=10)


# ----------------------------------------------------------------------------- 3. learning rate as a step response
def g_lr_response():
    fig, ax = plt.subplots(figsize=(7.2, 4.3))
    fig.subplots_adjust(left=0.1, right=0.78, top=0.95, bottom=0.14)
    K = 30
    cols = [MUTED, TEAL, YELLOW, RED]
    names = ["sluggish", "fast, no overshoot", "overshoots, rings", "unstable"]
    ax.axhline(D.W_STAR, color=FAINT, ls="--", lw=1.2)
    ax.text(K - 0.3, D.W_STAR + 0.15, "best w", color=MUTED, ha="right", fontsize=11)
    ax.set_xlim(0, K); ax.set_ylim(-2.6, 5.0)
    ax.set_xlabel("gradient-descent step"); ax.set_ylabel("slope of the line, w")
    lines, labels = [], []
    paths = [D.lr_response(lr, K) for lr in D.LR_SET]
    for lr, c, n in zip(D.LR_SET, cols, names):
        ln, = ax.plot([], [], "-o", color=c, ms=3.5, lw=1.8)
        lines.append(ln)
        labels.append(fig.text(0.8, 0, f"η = {lr}\n{n}", color=c, fontsize=11, va="center"))
    for j, lab in enumerate(labels):
        lab.set_y(0.82 - j * 0.2)

    def upd(i):
        k = min(i, K)
        for ln, p in zip(lines, paths):
            ln.set_data(np.arange(k + 1), np.clip(p[:k + 1], -5, 5))
        return []

    return _save(FuncAnimation(fig, upd, frames=_hold(K + 1, 20)), "lr_response", fps=8)


# ----------------------------------------------------------------------------- 4. non-convex landscape
def g_bumpy():
    fig, ax = plt.subplots(figsize=(7.6, 3.9))
    fig.subplots_adjust(left=0.04, right=0.98, top=0.95, bottom=0.12)
    ws = np.linspace(-5, 5.2, 500)
    ax.plot(ws, D.bumpy(ws), color=INK, lw=2.2)
    ax.set_yticks([]); ax.set_xlabel("a weight")
    ax.spines["left"].set_visible(False)
    cols = [ORANGE, YELLOW, BLUE]
    paths = [D.bumpy_gd(s) for s in D.BUMPY_STARTS]
    balls = [ax.plot([], [], "o", color=c, ms=13, mec=BG, mew=1.5)[0] for c in cols]
    trails = [ax.plot([], [], "-", color=c, lw=1.2, alpha=0.6)[0] for c in cols]
    wmin = ws[np.argmin(D.bumpy(ws))]
    ax.annotate("deepest valley", (wmin, D.bumpy(wmin)), (wmin - 0.9, D.bumpy(wmin) + 1.6), color=TEAL, fontsize=12,
                arrowprops=dict(arrowstyle="-", color=TEAL, lw=1))

    def upd(i):
        k = min(i, len(paths[0]) - 1)
        for b, t, p in zip(balls, trails, paths):
            b.set_data([p[k]], [D.bumpy(p[k]) + 0.09])
            t.set_data(p[:k + 1], D.bumpy(p[:k + 1]) + 0.09)
        return []

    return _save(FuncAnimation(fig, upd, frames=_hold(len(paths[0]), 25)), "bumpy", fps=12)


# ----------------------------------------------------------------------------- 5. full batch vs mini-batch
def g_sgd():
    wl, bl = (-0.8, 2.6), (-1.0, 3.8)
    ww, bb = np.meshgrid(np.linspace(*wl, 300), np.linspace(*bl, 300))
    L = np.mean((ww[..., None] * D.lin_x + bb[..., None] - D.lin_y) ** 2, -1)
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(9.2, 4.3), gridspec_kw=dict(width_ratios=[1, 1.15]))
    fig.subplots_adjust(left=0.05, right=0.97, top=0.9, bottom=0.14, wspace=0.25)
    ax.contour(ww, bb, L, levels=np.geomspace(L.min() * 1.05, L.max(), 20), colors="#5d6878", linewidths=0.8)
    ax.plot(D.W_STAR, D.B_STAR, "x", color=TEAL, ms=10, mew=2.5)
    ax.set_xlabel("slope w"); ax.set_ylabel("height b"); ax.set_xticks([]); ax.set_yticks([])
    ax.set_title("path through the landscape", loc="left", fontsize=13)
    full, sgd = D.C_FULL, D.C_SGD[::2]
    lf, = ax.plot([], [], "-o", color=INK, ms=3, lw=1.2)
    ls, = ax.plot([], [], "-o", color=YELLOW, ms=3, lw=1.0)
    lossf = [D.mse(*p) for p in full]
    losss = [D.mse(*p) for p in sgd]
    n = min(len(full), len(sgd))
    ax2.set_yscale("log"); ax2.set_xlim(0, n); ax2.set_ylim(0.08, 20)
    ax2.set_xlabel("step"); ax2.set_title("loss on all 14 batches", loc="left", fontsize=13)
    mf, = ax2.plot([], [], color=INK, lw=2, label="full batch: exact slope")
    ms, = ax2.plot([], [], color=YELLOW, lw=1.5, label="mini-batch of 4: noisy slope")
    ax2.legend(frameon=False, fontsize=11, labelcolor="linecolor")

    def upd(i):
        k = min(i, n - 1)
        lf.set_data(full[:k + 1, 0], full[:k + 1, 1]); ls.set_data(sgd[:k + 1, 0], sgd[:k + 1, 1])
        mf.set_data(np.arange(k + 1), lossf[:k + 1]); ms.set_data(np.arange(k + 1), losss[:k + 1])
        return []

    return _save(FuncAnimation(fig, upd, frames=_hold(n, 25)), "sgd", fps=12)


# ----------------------------------------------------------------------------- 6. logistic regression boundary in 2-D
def g_logistic2d():
    snaps = sorted(set([0, 1, 2, 3, 5, 8, 12, 18, 25, 35, 50, 70, 100, 140, 200, 280, 400]))
    S = D.l2_train(400, snaps=snaps)
    T = np.linspace(29.5, 40.5, 200); O = np.linspace(8, 62, 200)
    TT, OO = np.meshgrid(T, O)
    XX = np.stack([(TT - 35) / 3, (OO - 35) / 15], -1)
    cmap = LinearSegmentedColormap.from_list("pf", ["#16324a", "#1a1f28", "#4a3020"])
    fig, ax = plt.subplots(figsize=(6.4, 4.8))
    fig.subplots_adjust(left=0.12, right=0.97, top=0.92, bottom=0.13)
    im = ax.imshow(np.zeros_like(TT), extent=[T[0], T[-1], O[0], O[-1]], origin="lower", aspect="auto", cmap=cmap, vmin=0, vmax=1)
    ax.scatter(D.L2_T[D.L2_Y == 0], D.L2_DO[D.L2_Y == 0], c=BLUE, s=24, zorder=3)
    ax.scatter(D.L2_T[D.L2_Y == 1], D.L2_DO[D.L2_Y == 1], c=ORANGE, s=24, zorder=3)
    ln, = ax.plot([], [], color=YELLOW, lw=2.5, zorder=4)
    ttl = ax.set_title("", loc="left", fontsize=13)
    ax.set_xlabel("temperature (°C)"); ax.set_ylabel("dissolved oxygen (% sat.)")
    ax.set_xlim(T[0], T[-1]); ax.set_ylim(O[0], O[-1])

    def upd(i):
        s = snaps[min(i, len(snaps) - 1)]
        w, b = S[s]
        p = D.sigmoid(XX @ w + b)
        im.set_data(p)
        # boundary: w0*(T-35)/3 + w1*(O-35)/15 + b = 0
        Ob = 35 + 15 * (-(b + w[0] * (T - 35) / 3) / w[1])
        ln.set_data(T, Ob)
        acc = np.mean((D.sigmoid(D.L2_X @ w + b) > 0.5) == D.L2_Y)
        ttl.set_text(f"step {s}    accuracy {acc * 100:.0f}%")
        return []

    frames = [i for i in range(len(snaps)) for _ in range(3)] + [len(snaps) - 1] * 20
    return _save(FuncAnimation(fig, upd, frames=frames), "logistic2d", fps=10)


# ----------------------------------------------------------------------------- 7. what each hidden neuron draws
def g_hidden_lines():
    P = D.nn_snaps[D.NN_EPOCHS]
    T_LIM, PH_LIM = (29.6, 38.4), (6.56, 7.44)
    TT, PP = np.meshgrid(np.linspace(*T_LIM, 250), np.linspace(*PH_LIM, 250))
    X = D.nn_inputs(TT.ravel(), PP.ravel())
    Z = D.mlp_forward(P, X)[2].reshape(TT.shape)
    cmap = LinearSegmentedColormap.from_list("pf", ["#16324a", "#1a1f28", "#4a3020"])
    fig, ax = plt.subplots(figsize=(6.0, 4.9))
    fig.subplots_adjust(left=0.12, right=0.97, top=0.92, bottom=0.12)
    ax.scatter(D.nn_T[D.nn_y == 0], D.nn_pH[D.nn_y == 0], c=BLUE, s=16, zorder=3)
    ax.scatter(D.nn_T[D.nn_y == 1], D.nn_pH[D.nn_y == 1], c=ORANGE, s=16, zorder=3)
    ax.set_xlim(*T_LIM); ax.set_ylim(*PH_LIM)
    ax.set_xlabel("temperature (°C)"); ax.set_ylabel("pH")
    ttl = ax.set_title("", loc="left", fontsize=13)
    W1, b1, W2 = P["W1"], P["b1"], P["W2"].ravel()
    order = np.argsort(-np.abs(W2))
    T = np.linspace(*T_LIM, 200)
    lines = []
    for k in order:
        # unit k switches on where W1[0,k]*t + W1[1,k]*p + b1[k] = 0 (scaled units)
        tt = (T - D.T_MU) / D.T_SD
        pp = -(W1[0, k] * tt + b1[k]) / (W1[1, k] + 1e-9)
        ph = D.PH_MU + D.PH_SD * pp
        col = ORANGE if W2[k] > 0 else BLUE
        ln, = ax.plot([], [], color=col, lw=1.3, alpha=0.85)
        lines.append((ln, T, ph))
    holder = {"im": None, "cs": None}
    n = len(lines)

    def upd(i):
        k = min(i, n)
        for j, (ln, x, y) in enumerate(lines):
            if j < k:
                ln.set_data(x, y)
        if i == n + 2 and holder["im"] is None:
            holder["im"] = ax.imshow(Z, extent=[*T_LIM, *PH_LIM], origin="lower", aspect="auto", cmap=cmap, vmin=0, vmax=1, zorder=0)
            holder["cs"] = ax.contour(TT, PP, Z, levels=[0.5], colors=YELLOW, linewidths=3, zorder=5)
        ttl.set_text(f"hidden neuron lines: {k} of {n}" if i <= n + 1 else "blended by the output neuron")
        return []

    frames = [i for i in range(n + 1) for _ in range(2)] + [n + 1] * 4 + [n + 2] * 30
    return _save(FuncAnimation(fig, upd, frames=frames), "hidden_lines", fps=8)


# ----------------------------------------------------------------------------- 8. a ReLU net learns Haldane kinetics
HAL_SNAPS = [0, 5, 10, 20, 40, 70, 120, 200, 350, 600, 1000, 1700, 3000, 5000, 9000, 15000, 30000]


def g_haldane_fit():
    out = D.reg_net_train(16, HAL_SNAPS[-1], lr=0.05, snaps=HAL_SNAPS)
    S = np.linspace(0, 20, 400)
    fig, ax = plt.subplots(figsize=(7.0, 4.3))
    fig.subplots_adjust(left=0.11, right=0.97, top=0.9, bottom=0.14)
    ax.plot(S, D.haldane(S), "--", color=FAINT, lw=1.5, label="true Haldane curve")
    ax.scatter(D.HAL_S, D.HAL_Y, c=ORANGE, s=26, zorder=3, label="measured growth rates")
    fit, = ax.plot([], [], color=YELLOW, lw=2.8, label="16-neuron network")
    ax.set_xlim(0, 20); ax.set_ylim(-0.05, 0.42)
    ax.set_xlabel("substrate concentration S (g/L)"); ax.set_ylabel("specific growth rate μ (1/h)")
    ax.legend(frameon=False, loc="upper right", fontsize=10, labelcolor="linecolor")
    ttl = ax.set_title("", loc="left", fontsize=13)

    def upd(i):
        s = HAL_SNAPS[min(i, len(HAL_SNAPS) - 1)]
        prm = out[s]
        fit.set_data(S, D.reg_net_predict(prm, S))
        ttl.set_text(f"step {s:>5}    loss {prm[4]:.5f}")
        return []

    frames = [i for i in range(len(HAL_SNAPS)) for _ in range(3)] + [len(HAL_SNAPS) - 1] * 20
    return _save(FuncAnimation(fig, upd, frames=frames), "haldane_fit", fps=8)


def g_widths():
    widths = [1, 2, 4, 8, 16]
    S = np.linspace(0, 20, 400)
    fits = {h: D.reg_net_train(h, 30000, lr=0.05, snaps=[30000], seed=3)[30000] for h in widths}
    fig, ax = plt.subplots(figsize=(7.0, 4.3))
    fig.subplots_adjust(left=0.11, right=0.97, top=0.9, bottom=0.14)
    ax.plot(S, D.haldane(S), "--", color=FAINT, lw=1.5)
    ax.scatter(D.HAL_S, D.HAL_Y, c=ORANGE, s=26, zorder=3)
    fit, = ax.plot([], [], color=YELLOW, lw=2.8)
    ax.set_xlim(0, 20); ax.set_ylim(-0.05, 0.42)
    ax.set_xlabel("substrate concentration S (g/L)"); ax.set_ylabel("μ (1/h)")
    ttl = ax.set_title("", loc="left", fontsize=13)

    def upd(i):
        h = widths[min(i // 14, len(widths) - 1)]
        fit.set_data(S, D.reg_net_predict(fits[h], S))
        ttl.set_text(f"{h} hidden neuron{'s' if h > 1 else ''}  =  {h} bend{'s' if h > 1 else ''} available    loss {fits[h][4]:.5f}")
        return []

    return _save(FuncAnimation(fig, upd, frames=len(widths) * 14 + 10), "widths", fps=7)


# ----------------------------------------------------------------------------- 9. overfitting: polynomial degree
def poly_fit(deg):
    x = (D.OF_STR - 10) / 10
    A = np.vander(x, deg + 1)
    c = np.linalg.lstsq(A, D.OF_YTR, rcond=None)[0]
    return c


def poly_eval(c, S):
    return np.vander((np.asarray(S) - 10) / 10, len(c)) @ c


DEGREES = list(range(1, 13))


def overfit_numbers():
    tr, va = [], []
    for d in DEGREES:
        c = poly_fit(d)
        tr.append(float(np.mean((poly_eval(c, D.OF_STR) - D.OF_YTR) ** 2)))
        va.append(float(np.mean((poly_eval(c, D.OF_SVA) - D.OF_YVA) ** 2)))
    return np.array(tr), np.array(va)


def g_overfit():
    tr, va = overfit_numbers()
    S = np.linspace(0, 20, 400)
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(9.4, 4.2), gridspec_kw=dict(width_ratios=[1.3, 1]))
    fig.subplots_adjust(left=0.08, right=0.98, top=0.9, bottom=0.15, wspace=0.28)
    ax.scatter(D.OF_SVA, D.OF_YVA, c=FAINT, s=8, label="new batches (validation)")
    ax.scatter(D.OF_STR, D.OF_YTR, c=ORANGE, s=34, zorder=3, label="14 training batches")
    fit, = ax.plot([], [], color=YELLOW, lw=2.5)
    ax.set_xlim(0, 20); ax.set_ylim(-0.15, 0.5)
    ax.set_xlabel("substrate S (g/L)"); ax.set_ylabel("μ (1/h)")
    ax.legend(frameon=False, fontsize=10, loc="upper right", labelcolor="linecolor")
    ttl = ax.set_title("", loc="left", fontsize=13)
    tr = np.maximum(tr, 2e-5)
    ax2.set_yscale("log"); ax2.set_xlim(0.5, 12.5); ax2.set_ylim(1e-5, max(va.max(), 0.05) * 1.5)
    ax2.set_xlabel("polynomial degree (model complexity)"); ax2.set_title("error", loc="left", fontsize=13)
    a, = ax2.plot([], [], "-o", color=ORANGE, label="training")
    b, = ax2.plot([], [], "-o", color=TEAL, label="validation")
    ax2.legend(frameon=False, fontsize=11, labelcolor="linecolor")

    def upd(i):
        k = min(i // 6, len(DEGREES) - 1)
        d = DEGREES[k]
        fit.set_data(S, np.clip(poly_eval(poly_fit(d), S), -1, 1))
        ttl.set_text(f"degree {d}: {d + 1} parameters")
        a.set_data(DEGREES[:k + 1], tr[:k + 1]); b.set_data(DEGREES[:k + 1], va[:k + 1])
        return []

    return _save(FuncAnimation(fig, upd, frames=len(DEGREES) * 6 + 18), "overfit", fps=6)


# ----------------------------------------------------------------------------- static: extrapolation
def p_extrapolate():
    out = D.reg_net_train(16, 30000, lr=0.05, snaps=[30000])[30000]
    S = np.linspace(0, 40, 600)
    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    fig.subplots_adjust(left=0.11, right=0.97, top=0.92, bottom=0.14)
    ax.axvspan(0, 20, color="#1b2633", zorder=0)
    ax.text(10, 0.43, "trained here", color=MUTED, ha="center", fontsize=12)
    ax.text(30, 0.43, "asked to predict here", color=RED, ha="center", fontsize=12)
    ax.plot(S, D.haldane(S), "--", color=MUTED, lw=1.6, label="true kinetics")
    ax.scatter(D.HAL_S, D.HAL_Y, c=ORANGE, s=22, zorder=3)
    ax.plot(S, D.reg_net_predict(out, S), color=YELLOW, lw=2.8, label="network")
    ax.set_xlim(0, 40); ax.set_ylim(-0.25, 0.47)
    ax.set_xlabel("substrate concentration S (g/L)"); ax.set_ylabel("μ (1/h)")
    ax.legend(frameon=False, fontsize=11, loc="lower left", labelcolor="linecolor")
    path = os.path.join(OUT, "extrapolate.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print("wrote", path)
    return path


ALL = [g_plane, g_scaling, g_lr_response, g_bumpy, g_sgd, g_logistic2d, g_hidden_lines, g_haldane_fit, g_widths,
       g_overfit, p_extrapolate]

if __name__ == "__main__":
    for f in ALL:
        f()
