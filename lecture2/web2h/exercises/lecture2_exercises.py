"""Lecture 2 - Learning by gradients: hands-on exercises.

Run everything:            python lecture2_exercises.py
Run one exercise:          python lecture2_exercises.py 1      (or 2 ... 6)

Only numpy and matplotlib are needed. All data are simulated, with the same recipes as the lecture
animations: an operating window in (temperature, pH), Arrhenius rate constants, Haldane growth kinetics,
an A -> B -> C series reaction, and fermentation batches for cross-validation.
Each exercise prints what to look for. The places marked TODO are where you should experiment.
"""
import sys

import matplotlib.pyplot as plt
import numpy as np

rng = np.random.default_rng(0)


# --------------------------------------------------------------------------- data
def window_data(n=240, seed=42):
    """Batches at random temperature (30-38 C) and pH (6.6-7.4); good (1) inside an elliptical window."""
    r = np.random.default_rng(seed)
    T, pH = r.uniform(30, 38, n), r.uniform(6.6, 7.4, n)
    y = (((T - 34.2) / 2.0) ** 2 + ((pH - 7.02) / 0.19) ** 2 < 1).astype(float)
    X = np.c_[(T - 34.0) / 2.3, (pH - 7.0) / 0.23]          # standardised inputs
    return X, y


def haldane(S, mu_max=0.45, Ks=1.0, Ki=6.0):
    return mu_max * S / (Ks + S + S ** 2 / Ki)


# --------------------------------------------------------------------------- the 2-16-1 network from chapter 5
def init(hidden=16, seed=1):
    r = np.random.default_rng(seed)
    return {"W1": r.normal(0, np.sqrt(2 / 2), (2, hidden)), "b1": r.normal(0, .5, hidden),
            "w2": r.normal(0, np.sqrt(2 / hidden), hidden), "b2": 0.0}


def forward(p, X):
    z1 = X @ p["W1"] + p["b1"]
    h = np.maximum(0, z1)
    out = 1 / (1 + np.exp(-(h @ p["w2"] + p["b2"])))
    return z1, h, out


def loss(p, X, y):
    _, _, out = forward(p, X)
    out = np.clip(out, 1e-12, 1 - 1e-12)
    return -np.mean(y * np.log(out) + (1 - y) * np.log(1 - out))


def backward(p, X, y, relu_mask=True):
    z1, h, out = forward(p, X)
    dz2 = (out - y) / len(y)                       # sigmoid + cross-entropy: p - y
    g = {"w2": h.T @ dz2, "b2": dz2.sum()}
    dh = np.outer(dz2, p["w2"])
    dz1 = dh * (z1 > 0) if relu_mask else dh       # the ReLU slope (break it on purpose in ex. 2)
    g["W1"], g["b1"] = X.T @ dz1, dz1.sum(0)
    return g


def train(lr, epochs=500, seed=1):
    X, y = window_data()
    p, hist = init(seed=seed), []
    for _ in range(epochs):
        hist.append(loss(p, X, y))
        g = backward(p, X, y)
        for k in p:
            p[k] = p[k] - lr * g[k]
    return p, np.array(hist)


# --------------------------------------------------------------------------- exercise 1
def ex1():
    """Backprop from scratch: four learning rates, four loss curves."""
    plt.figure(figsize=(7, 4))
    for lr in [0.01, 0.1, 1.0, 10.0]:                      # TODO: try others, add momentum
        _, h = train(lr)
        plt.semilogy(h, label=f"η = {lr}")
    plt.xlabel("epoch"); plt.ylabel("cross-entropy"); plt.legend(); plt.title("Exercise 1: which curve is which diagnosis?")
    plt.tight_layout(); plt.show()
    print("Ex 1: tiny η crawls, a good η drops fast, a huge η stalls or jumps. Match them to the loss-curve doctor.")


# --------------------------------------------------------------------------- exercise 2
def ex2():
    """Gradient check by central differences."""
    X, y = window_data()
    p = init()
    for mask in (True, False):
        g = backward(p, X, y, relu_mask=mask)
        worst = 0.0
        for _ in range(10):                                  # ten random knobs in W1
            i, j = rng.integers(2), rng.integers(16)
            eps = 1e-5
            pp = {k: np.copy(v) for k, v in p.items()}; pp["W1"][i, j] += eps
            pm = {k: np.copy(v) for k, v in p.items()}; pm["W1"][i, j] -= eps
            num = (loss(pp, X, y) - loss(pm, X, y)) / (2 * eps)
            rel = abs(num - g["W1"][i, j]) / max(1e-12, abs(num) + abs(g["W1"][i, j]))
            worst = max(worst, rel)
        print(f"Ex 2: ReLU mask {'on ' if mask else 'OFF'} -> worst relative error {worst:.1e}")
    print("      Expect ~1e-7 or smaller with the correct backward pass, and a large error once the mask is broken.")


# --------------------------------------------------------------------------- exercise 3
def ex3():
    """Conditioning: fit ln k = ln A - Ea/(R T) by gradient descent, raw vs standardised input."""
    R, Ea, A = 8.314, 55000.0, 1.0e7
    T = np.linspace(300, 370, 8)
    lnk = np.log(A) - Ea / (R * T) + 0.06 * rng.standard_normal(8)
    for name, x in [("raw 1000/T", 1000 / T), ("standardised", (1000 / T - (1000 / T).mean()) / (1000 / T).std())]:
        X = np.c_[x, np.ones_like(x)]
        H = 2 * X.T @ X / len(x)                             # Hessian of the mean squared error
        lam = np.linalg.eigvalsh(H); lr = 2 / (lam.max() + lam.min())   # the best fixed step for this bowl
        w, steps = np.zeros(2), 0
        w_star = np.linalg.lstsq(X, lnk, rcond=None)[0]
        while np.linalg.norm(w - w_star) > 1e-3 * np.linalg.norm(w_star) and steps < 2_000_000:
            w -= lr * 2 * X.T @ (X @ w - lnk) / len(x)
            steps += 1
        print(f"Ex 3: {name:13s} condition number {np.linalg.cond(H):10.0f}   steps to converge {steps:,}")
    slope = np.polyfit(1000 / T, lnk, 1)[0]
    print(f"      fitted Ea = {-slope * R:.1f} kJ/mol  (true 55.0)")


# --------------------------------------------------------------------------- exercise 4
def ex4():
    """Network vs physics: who extrapolates to S = 40 g/L?"""
    S = np.linspace(0.2, 20, 26)
    Y = haldane(S) + 0.011 * rng.standard_normal(26)
    # (a) a 1-16-1 ReLU network trained by full-batch gradient descent with momentum
    r = np.random.default_rng(3)
    W1, b1, w2, b2 = r.normal(0, 1, 16), r.uniform(-20, 0, 16) * .05, r.normal(0, .1, 16), 0.0
    mW = [0, 0, 0, 0]
    x = S / 20                                               # scale input to [0, 1]
    for _ in range(20000):
        z = np.outer(x, W1) + b1; h = np.maximum(0, z); out = h @ w2 + b2
        d = 2 * (out - Y) / len(Y)
        gw2, gb2 = h.T @ d, d.sum(); dz = np.outer(d, w2) * (z > 0)
        gW1, gb1 = x @ dz, dz.sum(0)
        grads = [gW1, gb1, gw2, gb2]
        mW = [0.9 * m + g for m, g in zip(mW, grads)]
        W1, b1, w2, b2 = W1 - .05 * mW[0], b1 - .05 * mW[1], w2 - .05 * mW[2], b2 - .05 * mW[3]
    net = lambda s: np.maximum(0, np.outer(np.atleast_1d(s) / 20, W1) + b1) @ w2 + b2
    # (b) the Haldane form with three knobs, fitted by the same kind of loop (Adam on log-parameters)
    th, m, v = np.log([0.2, 3.0, 30.0]), np.zeros(3), np.zeros(3)
    model = lambda t, s: np.exp(t[0]) * s / (np.exp(t[1]) + s + s ** 2 / np.exp(t[2]))
    L = lambda t: np.mean((model(t, S) - Y) ** 2)
    for k in range(1, 601):
        g = np.array([(L(th + e) - L(th - e)) / 2e-5 for e in np.eye(3) * 1e-5])
        m, v = .9 * m + .1 * g, .999 * v + .001 * g * g
        th -= .05 * (m / (1 - .9 ** k)) / (np.sqrt(v / (1 - .999 ** k)) + 1e-8)
    s = np.linspace(0, 40, 400)
    plt.figure(figsize=(7, 4))
    plt.axvspan(20, 40, color="0.92"); plt.plot(S, Y, "o", label="data")
    plt.plot(s, haldane(s), "k--", lw=1, label="true law"); plt.plot(s, net(s), label="1-16-1 network")
    plt.plot(s, model(th, s), label="Haldane form, 3 knobs"); plt.axhline(0, color="0.6", lw=.8)
    plt.xlabel("S (g/L)"); plt.ylabel("μ (1/h)"); plt.legend(); plt.title("Exercise 4: extrapolation"); plt.tight_layout(); plt.show()
    print(f"Ex 4: at S = 40: true {haldane(40.0):.3f}, network {net(40.0)[0]:.3f}, hybrid {model(th, 40.0):.3f}")
    print("      fitted mu_max, Ks, Ki =", np.round(np.exp(th), 2))


# --------------------------------------------------------------------------- exercise 5
def abc(k1, k2, t):
    """A -> B -> C, first order, C_A0 = 1."""
    A = np.exp(-k1 * t)
    B = k1 / (k2 - k1) * (np.exp(-k1 * t) - np.exp(-k2 * t)) if abs(k2 - k1) > 1e-9 else k1 * t * np.exp(-k1 * t)
    return np.stack([A, B, 1 - A - B], axis=-1)


def ex5():
    """Rate constants by gradient descent: Adam on (ln k1, ln k2), and why the log matters."""
    t = np.linspace(1, 20, 11)
    data = abc(0.45, 0.12, t) + 0.02 * rng.standard_normal((11, 3))
    def run(log_space, lr, steps=400):
        p = np.log([0.06, 0.9]) if log_space else np.array([0.06, 0.9])
        m, v = np.zeros(2), np.zeros(2)
        k_of = (lambda q: np.exp(q)) if log_space else (lambda q: np.abs(q) + 1e-9)
        L = lambda q: np.mean((abc(*k_of(q), t) - data) ** 2)
        hist = []
        for i in range(1, steps + 1):
            g = np.array([(L(p + e) - L(p - e)) / 2e-6 for e in np.eye(2) * 1e-6])   # TODO: replace with autodiff (jax.grad)
            m, v = .9 * m + .1 * g, .999 * v + .001 * g * g
            p = p - lr * (m / (1 - .9 ** i)) / (np.sqrt(v / (1 - .999 ** i)) + 1e-8)
            hist.append(L(p))
        return k_of(p), hist
    (k_log, h_log), (k_lin, h_lin) = run(True, 0.06), run(False, 0.01)
    plt.figure(figsize=(7, 4)); plt.semilogy(h_log, label="Adam on ln k"); plt.semilogy(h_lin, label="Adam on k")
    plt.xlabel("step"); plt.ylabel("loss"); plt.legend(); plt.title("Exercise 5: fitting k1, k2"); plt.tight_layout(); plt.show()
    print(f"Ex 5: ln-space fit k1, k2 = {k_log.round(3)}   linear-space fit = {k_lin.round(3)}   (true 0.45, 0.12)")
    print("      TODO: start from k1 = 0.9, k2 = 0.06 (the order swapped). Does it still find the right answer? Why?")


# --------------------------------------------------------------------------- exercise 6
def ex6():
    """Cross-validation and leakage: random folds vs leave-one-batch-out on fermentation batches."""
    r = np.random.default_rng(57)
    ts = np.linspace(0, 12, 16)
    curve = 1 / (1 + np.exp(-(ts - 5) / 1.3))
    def batch():
        x, off = r.uniform(), 0.09 * r.standard_normal()
        return x, curve * (0.6 + 0.8 * x) + off + 0.02 * r.standard_normal(16)
    rows = []
    for j in range(6):
        x, ys = batch()
        rows += [(ts[i] / 12, x, ys[i], j) for i in range(16)]
    D_ = np.array(rows)
    F, y, grp = D_[:, :2], D_[:, 2], D_[:, 3]
    def knn(Ftr, ytr, Fte, k=3):
        d = ((Fte[:, None, :] - Ftr[None, :, :]) ** 2).sum(-1)
        return ytr[np.argsort(d, 1)[:, :k]].mean(1)
    rmse = lambda a, b: np.sqrt(np.mean((a - b) ** 2))
    fold = r.permutation(len(y)) % 5
    rand = np.sqrt(np.mean([rmse(knn(F[fold != f], y[fold != f], F[fold == f]), y[fold == f]) ** 2 for f in range(5)]))
    group = np.sqrt(np.mean([rmse(knn(F[grp != g], y[grp != g], F[grp == g]), y[grp == g]) ** 2 for g in range(6)]))
    new = [batch() for _ in range(30)]
    Fn = np.array([(ts[i] / 12, x) for x, _ in new for i in range(16)]); yn = np.concatenate([ys for _, ys in new])
    print(f"Ex 6: random 5-fold CV {rand:.3f}   leave-one-batch-out CV {group:.3f}   30 new batches {rmse(knn(F, y, Fn), yn):.3f}")
    print("      TODO: redo it with sklearn: KFold vs GroupKFold(n_splits=6) and KNeighborsRegressor(3).")


if __name__ == "__main__":
    todo = sys.argv[1:] or ["1", "2", "3", "4", "5", "6"]
    for t in todo:
        {"1": ex1, "2": ex2, "3": ex3, "4": ex4, "5": ex5, "6": ex6}[t]()
