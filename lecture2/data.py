"""
data.py - every number shown in Lecture 2 is computed here from the real algorithms.
All data sets are simulated (bioprocess flavour) and seeded, so the deck is reproducible.
"""
import numpy as np

# ----------------------------------------------------------------------------- 1. linear regression
rng = np.random.default_rng(7)
N_LIN = 14
lin_x = np.sort(rng.uniform(0.4, 3.0, N_LIN))           # glucose feed rate, g/L/h
lin_y = 0.9 + 1.2 * lin_x + rng.normal(0, 0.35, N_LIN)   # titre, g/L

def mse(w, b, x=lin_x, y=lin_y):
    return float(np.mean((w * x + b - y) ** 2))

def grad(w, b, x=lin_x, y=lin_y):
    r = w * x + b - y
    return float(2 * np.mean(r * x)), float(2 * np.mean(r))

W_STAR, B_STAR = np.polyfit(lin_x, lin_y, 1)

# full gradient descent on (w, b)
def gd_path(w, b, lr, steps, batch=None, seed=0):
    r = np.random.default_rng(seed)
    path = [(w, b)]
    for _ in range(steps):
        if batch:
            idx = r.choice(N_LIN, batch, replace=False)
            gw, gb = grad(w, b, lin_x[idx], lin_y[idx])
        else:
            gw, gb = grad(w, b)
        w, b = w - lr * gw, b - lr * gb
        path.append((w, b))
    return np.array(path)

CH1_START = (0.15, 1.30)
CH1_PATH = gd_path(*CH1_START, lr=0.05, steps=3000)
CH1_FRAMES = [0, 2, 6, 3000]                 # iterations shown on the "shrinking squares" slides

# 1-D view: b frozen at its best value, only the slope w moves
def L1(w):
    return mse(w, B_STAR)

def dL1(w):
    return grad(w, B_STAR)[0]

CURV = 2 * np.mean(lin_x ** 2)                # second derivative of L1, sets the stable step size
def gd1(w0, lr, steps):
    ws = [w0]
    for _ in range(steps):
        ws.append(ws[-1] - lr * dL1(ws[-1]))
    return np.array(ws)

CH2_W0 = -0.35
CH2_LR = 0.08
CH2_WS = gd1(CH2_W0, CH2_LR, 4)
LR_SMALL, LR_GOOD, LR_BIG = 0.01, 0.08, round(2.15 / CURV, 2)
LR_PATHS = {
    "small": gd1(CH2_W0, LR_SMALL, 7),
    "good": gd1(CH2_W0, LR_GOOD, 5),
    "big": gd1(W_STAR - 0.9, LR_BIG, 5),
}

# 2-D landscape
C_START = (-0.4, 3.4)
C_FULL = gd_path(*C_START, lr=0.1, steps=120)
C_SGD = gd_path(*C_START, lr=0.05, steps=240, batch=4, seed=3)

# ----------------------------------------------------------------------------- 2. logistic regression
rng2 = np.random.default_rng(11)
N_LOG = 24
log_x = np.sort(rng2.uniform(0.3, 5.7, N_LOG))           # lactate at 48 h, g/L
_p_true = 1 / (1 + np.exp(-1.8 * (log_x - 3.0)))
log_y = (rng2.uniform(size=N_LOG) < _p_true).astype(float)  # 1 = batch failed spec

LS_W, LS_B = np.polyfit(log_x, log_y, 1)                 # straight line through 0/1 labels

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def fit_logistic(x, y, lr=0.5, steps=20000):
    w = b = 0.0
    xm = x.mean()
    for _ in range(steps):
        p = sigmoid(w * (x - xm) + b)
        w -= lr * np.mean((p - y) * (x - xm))
        b -= lr * np.mean(p - y)
    return w, b - w * xm

LG_W, LG_B = fit_logistic(log_x, log_y)
log_p = sigmoid(LG_W * log_x + LG_B)
fails = np.where(log_y == 1)[0]
CE_WRONG = int(fails[np.argmin(log_p[fails])])   # a failed batch the model was sure would pass
CE_RIGHT = int(fails[np.argmax(log_p[fails])])   # a failed batch the model saw coming

# ----------------------------------------------------------------------------- 3. neural network on an operating window
rng3 = np.random.default_rng(5)
N_NN = 150
nn_T = rng3.uniform(30, 38, N_NN)
nn_pH = rng3.uniform(6.6, 7.4, N_NN)
_inside = ((nn_T - 34.2) / 2.1) ** 2 + ((nn_pH - 6.98) / 0.21) ** 2 < 1
nn_y = _inside.astype(float)
T_MU, T_SD, PH_MU, PH_SD = 34, 2.3, 7.0, 0.23
def nn_inputs(T, pH):
    return np.stack([(T - T_MU) / T_SD, (pH - PH_MU) / PH_SD], -1)
NN_X = nn_inputs(nn_T, nn_pH)

def relu(z):
    return np.maximum(z, 0)

def mlp_init(hidden=16, seed=2):
    r = np.random.default_rng(seed)
    return dict(W1=r.normal(0, 1.0, (2, hidden)), b1=r.normal(0, 0.5, hidden),
                W2=r.normal(0, 1 / np.sqrt(hidden), (hidden, 1)), b2=np.zeros(1))

def mlp_forward(P, X):
    z1 = X @ P["W1"] + P["b1"]
    h = relu(z1)
    p = sigmoid(h @ P["W2"] + P["b2"]).ravel()
    return z1, h, p

def bce(p, y):
    p = np.clip(p, 1e-7, 1 - 1e-7)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))

def mlp_train(epochs, lr=0.5, snapshots=()):
    """Plain full-batch gradient descent, gradients by hand-written backprop."""
    P = mlp_init()
    snaps, losses = {}, []
    n = len(nn_y)
    for ep in range(epochs + 1):
        z1, h, p = mlp_forward(P, NN_X)
        losses.append(bce(p, nn_y))
        if ep in snapshots:
            snaps[ep] = {k: v.copy() for k, v in P.items()}
        if ep == epochs:
            break
        dlogit = (p - nn_y)[:, None] / n           # dL/d(output logit) for sigmoid + cross-entropy
        dW2 = h.T @ dlogit
        db2 = dlogit.sum(0)
        dh = dlogit @ P["W2"].T                   # chain rule into the hidden layer
        dz1 = dh * (z1 > 0)                       # ReLU passes gradient only where it was "on"
        dW1 = NN_X.T @ dz1
        db1 = dz1.sum(0)
        for k, g in (("W1", dW1), ("b1", db1), ("W2", dW2), ("b2", db2)):
            P[k] -= lr * g
    return snaps, np.array(losses)

NN_EPOCHS = 3000
NN_SNAPS = [0, 10, 30, 80, 250, 800, 3000]
nn_snaps, nn_losses = mlp_train(NN_EPOCHS, lr=0.3, snapshots=NN_SNAPS)

def nn_acc(P):
    return float(np.mean((mlp_forward(P, NN_X)[2] > 0.5) == nn_y))

# logistic regression on the same window data (a single straight boundary)
def fit_logistic2d(steps=6000, lr=0.5):
    w = np.zeros(2); b = 0.0
    for _ in range(steps):
        p = sigmoid(NN_X @ w + b)
        w -= lr * NN_X.T @ (p - nn_y) / len(nn_y)
        b -= lr * np.mean(p - nn_y)
    return w, b
LOG2_W, LOG2_B = fit_logistic2d()
LOG2_ACC = float(np.mean((sigmoid(NN_X @ LOG2_W + LOG2_B) > 0.5) == nn_y))

# ----------------------------------------------------------------------------- 4. backprop worked by hand
BP = dict(x=2.0, w1=0.5, w2=1.5, y=3.0, lr=0.05)
BP["h"] = max(BP["w1"] * BP["x"], 0)
BP["yhat"] = BP["w2"] * BP["h"]
BP["L"] = (BP["yhat"] - BP["y"]) ** 2
BP["dyhat"] = 2 * (BP["yhat"] - BP["y"])
BP["dw2"] = BP["dyhat"] * BP["h"]
BP["dh"] = BP["dyhat"] * BP["w2"]
BP["dw1"] = BP["dh"] * (1.0 if BP["w1"] * BP["x"] > 0 else 0.0) * BP["x"]
BP["w1n"] = BP["w1"] - BP["lr"] * BP["dw1"]
BP["w2n"] = BP["w2"] - BP["lr"] * BP["dw2"]
BP["hn"] = max(BP["w1n"] * BP["x"], 0)
BP["yhatn"] = BP["w2n"] * BP["hn"]
BP["Ln"] = (BP["yhatn"] - BP["y"]) ** 2


if __name__ == "__main__":
    print("w*, b*", W_STAR, B_STAR, "L*", mse(W_STAR, B_STAR))
    for k in CH1_FRAMES:
        print("ch1 it", k, CH1_PATH[k], mse(*CH1_PATH[k]))
    print("curv", CURV, "lr big", LR_BIG)
    print("ch2 ws", CH2_WS, [L1(w) for w in CH2_WS])
    for k, v in LR_PATHS.items():
        print(k, v)
    print("C_full end", C_FULL[-1], "sgd end", C_SGD[-1])
    print("LS", LS_W, LS_B, "at 6:", LS_W * 6 + LS_B, "at 0:", LS_B)
    print("LG", LG_W, LG_B, "wrong", log_x[CE_WRONG], log_p[CE_WRONG], "right", log_x[CE_RIGHT], log_p[CE_RIGHT])
    print("nn inside frac", nn_y.mean(), "log2 acc", LOG2_ACC)
    for e in NN_SNAPS:
        print("epoch", e, "loss", nn_losses[e], "acc", nn_acc(nn_snaps[e]))
    print(BP)


# =============================================================================
# Detail material added for the expanded deck
# =============================================================================
R_GAS = 8.314

# ----------------------------------------------------------------------------- Arrhenius: linear regression in disguise
rngA = np.random.default_rng(21)
ARR_T = np.array([300., 310., 320., 330., 340., 350., 360., 370.])     # K
ARR_EA_TRUE, ARR_A_TRUE = 55_000.0, 1.0e7                              # J/mol, 1/s
ARR_K = ARR_A_TRUE * np.exp(-ARR_EA_TRUE / (R_GAS * ARR_T)) * np.exp(rngA.normal(0, 0.06, ARR_T.size))
ARR_X = 1000.0 / ARR_T                                                  # 1000/T, 1/K (x1000 for readable numbers)
ARR_Y = np.log(ARR_K)
ARR_SLOPE, ARR_ICPT = np.polyfit(ARR_X, ARR_Y, 1)
ARR_EA_FIT = -ARR_SLOPE * 1000.0 * R_GAS                                # J/mol
ARR_A_FIT = np.exp(ARR_ICPT)

# ----------------------------------------------------------------------------- 4 points, by hand
H4_X = np.array([1.0, 2.0, 3.0, 4.0])
H4_Y = np.array([2.1, 3.9, 6.2, 7.8])
H4_W0, H4_B0, H4_LR = 1.0, 1.0, 0.05

def h4_table(w, b):
    yh = w * H4_X + b
    r = H4_Y - yh
    return yh, r, r ** 2, float(np.mean(r ** 2))

def h4_grad(w, b):
    r = H4_Y - (w * H4_X + b)
    return float(-2 * np.mean(H4_X * r)), float(-2 * np.mean(r))

H4_XBAR, H4_YBAR = H4_X.mean(), H4_Y.mean()
H4_COV = float(np.mean((H4_X - H4_XBAR) * (H4_Y - H4_YBAR)))
H4_VAR = float(np.mean((H4_X - H4_XBAR) ** 2))
H4_WSTAR = H4_COV / H4_VAR
H4_BSTAR = H4_YBAR - H4_WSTAR * H4_XBAR
H4_ITERS = [(H4_W0, H4_B0)]
for _ in range(3):
    gw, gb = h4_grad(*H4_ITERS[-1])
    H4_ITERS.append((H4_ITERS[-1][0] - H4_LR * gw, H4_ITERS[-1][1] - H4_LR * gb))

# ----------------------------------------------------------------------------- two features: a regression plane
rngP = np.random.default_rng(4)
PL_N = 40
PL_F = rngP.uniform(0.5, 3.0, PL_N)            # feed rate g/L/h
PL_T = rngP.uniform(30, 38, PL_N)              # temperature C
PL_Y = 0.8 + 1.1 * PL_F + 0.18 * (PL_T - 30) + rngP.normal(0, 0.25, PL_N)
_A = np.c_[PL_F, PL_T, np.ones(PL_N)]
PL_W = np.linalg.lstsq(_A, PL_Y, rcond=None)[0]

# ----------------------------------------------------------------------------- feature scaling
rngS = np.random.default_rng(9)
SC_N = 60
SC_T = rngS.uniform(30, 38, SC_N)
SC_PH = rngS.uniform(6.6, 7.4, SC_N)
SC_Y = 0.35 * (SC_T - 34) + 3.0 * (SC_PH - 7.0) + rngS.normal(0, 0.15, SC_N)
SC_RAW = np.c_[SC_T - SC_T.mean(), SC_PH - SC_PH.mean()]              # centred, NOT scaled
SC_STD = SC_RAW / SC_RAW.std(0)                                       # standardised
def sc_fit(X):
    return np.linalg.lstsq(X, SC_Y - SC_Y.mean(), rcond=None)[0]
def sc_loss(X, w):
    return float(np.mean((X @ w - (SC_Y - SC_Y.mean())) ** 2))
def sc_gd(X, w0, lr, steps):
    ws = [np.array(w0, float)]
    yc = SC_Y - SC_Y.mean()
    for _ in range(steps):
        w = ws[-1]
        g = 2 * X.T @ (X @ w - yc) / len(yc)
        ws.append(w - lr * g)
    return np.array(ws)
SC_WRAW = sc_fit(SC_RAW)
SC_WSTD = sc_fit(SC_STD)
SC_CURV_RAW = 2 * np.var(SC_RAW, 0)                     # diagonal curvatures
SC_LR_RAW = round(1.8 / SC_CURV_RAW.max(), 3)
SC_PATH_RAW = sc_gd(SC_RAW, SC_WRAW + np.array([0.9, -2.6]), SC_LR_RAW, 60)
SC_PATH_STD = sc_gd(SC_STD, SC_WSTD + np.array([0.9 * SC_RAW[:, 0].std(), -2.6 * SC_RAW[:, 1].std()]), 0.4, 60)

# ----------------------------------------------------------------------------- learning rate as a step response
LR_SET = [0.03, 0.12, 0.25, 0.30]                       # on the 1-D titre loss, curvature = CURV
def lr_response(lr, steps=30, w0=-0.35):
    return gd1(w0, lr, steps)
LR_ETA_MAX = 2 / CURV

# ----------------------------------------------------------------------------- a bumpy (non-convex) 1-D loss
def bumpy(w):
    return 0.12 * (w - 0.5) ** 2 + 0.55 * np.sin(2.6 * w) + 0.9
def dbumpy(w):
    return 0.24 * (w - 0.5) + 0.55 * 2.6 * np.cos(2.6 * w)
def bumpy_gd(w0, lr=0.08, steps=60):
    ws = [w0]
    for _ in range(steps):
        ws.append(ws[-1] - lr * dbumpy(ws[-1]))
    return np.array(ws)
BUMPY_STARTS = [-4.2, -0.6, 3.9]

# ----------------------------------------------------------------------------- epochs and batches
EB_N, EB_BATCH = 10_000, 32
EB_STEPS = int(np.ceil(EB_N / EB_BATCH))

# ----------------------------------------------------------------------------- logistic regression, 4 batches by hand
LH_X = np.array([1.5, 2.5, 3.5, 4.5])
LH_Y = np.array([0., 0., 1., 1.])
LH_W, LH_B = 1.6, -4.8
LH_Z = LH_W * LH_X + LH_B
LH_P = sigmoid(LH_Z)
LH_LOSS = -(LH_Y * np.log(LH_P) + (1 - LH_Y) * np.log(1 - LH_P))
LH_MEAN = float(LH_LOSS.mean())
LG_THRESH = -LG_B / LG_W                                 # lactate at which p = 0.5
# squared error vs cross-entropy for a sigmoid output when the truth is y=1
SQ_VS_CE_P = np.linspace(0.005, 0.995, 100)

# 2-feature logistic regression on linearly separable data (T, DO)
rngL = np.random.default_rng(13)
L2_N = 80
L2_T = rngL.uniform(30, 40, L2_N)
L2_DO = rngL.uniform(10, 60, L2_N)                      # dissolved oxygen, % saturation
L2_Y = ((0.55 * (L2_T - 35) - 0.09 * (L2_DO - 35) + rngL.normal(0, 0.6, L2_N)) > 0).astype(float)  # 1 = runaway / fail
L2_X = np.c_[(L2_T - 35) / 3, (L2_DO - 35) / 15]
def l2_train(steps=400, lr=0.5, snaps=()):
    w = np.array([-1.0, 1.5]); b = 1.0
    out = {}
    for s in range(steps + 1):
        if s in snaps:
            out[s] = (w.copy(), b)
        p = sigmoid(L2_X @ w + b)
        w = w - lr * L2_X.T @ (p - L2_Y) / L2_N
        b = b - lr * np.mean(p - L2_Y)
    return out

# ----------------------------------------------------------------------------- a 2-2-1 network forward pass, by hand
NH_X = np.array([0.5, -1.0])                            # scaled T, scaled pH
NH_W1 = np.array([[2.0, 1.0], [-1.0, 1.5]])             # rows = inputs, cols = hidden units
NH_B1 = np.array([0.1, 0.2])
NH_W2 = np.array([1.2, -0.7])
NH_B2 = -1.0
NH_Z1 = NH_X @ NH_W1 + NH_B1
NH_H = relu(NH_Z1)
NH_Z2 = float(NH_H @ NH_W2 + NH_B2)
NH_P = float(sigmoid(NH_Z2))
NN_PARAMS = 2 * 16 + 16 + 16 + 1

# ----------------------------------------------------------------------------- backprop: gradient check by finite differences
def bp_loss(w1, w2, x=2.0, y=3.0):
    return (w2 * max(w1 * x, 0) - y) ** 2
GC_EPS = 1e-3
GC_W1 = (bp_loss(0.5 + GC_EPS, 1.5) - bp_loss(0.5 - GC_EPS, 1.5)) / (2 * GC_EPS)
GC_W2 = (bp_loss(0.5, 1.5 + GC_EPS) - bp_loss(0.5, 1.5 - GC_EPS)) / (2 * GC_EPS)

# ----------------------------------------------------------------------------- 1-D regression with a small ReLU net: Haldane kinetics
rngH = np.random.default_rng(17)
HAL_MU, HAL_KS, HAL_KI = 0.5, 1.0, 8.0
def haldane(S):
    return HAL_MU * S / (HAL_KS + S + S ** 2 / HAL_KI)
HAL_S = np.sort(rngH.uniform(0.2, 20, 26))
HAL_Y = haldane(HAL_S) + rngH.normal(0, 0.012, HAL_S.size)
HAL_SMU, HAL_SSD = 10.0, 5.8

def reg_net_train(hidden, steps, lr=0.05, snaps=(), seed=1, X=None, Y=None):
    """1-hidden-layer ReLU regression net, plain gradient descent on mean squared error."""
    X = ((HAL_S if X is None else X) - HAL_SMU) / HAL_SSD
    Y = HAL_Y if Y is None else Y
    r = np.random.default_rng(seed)
    W1 = r.normal(0, 1.5, hidden); b1 = r.uniform(-1.5, 1.5, hidden)
    W2 = r.normal(0, 0.3, hidden); b2 = 0.0
    out = {}
    n = len(X)
    for s in range(steps + 1):
        z = np.outer(X, W1) + b1
        h = np.maximum(z, 0)
        yh = h @ W2 + b2
        if s in snaps:
            out[s] = (W1.copy(), b1.copy(), W2.copy(), b2, float(np.mean((yh - Y) ** 2)))
        if s == steps:
            break
        d = 2 * (yh - Y) / n
        gW2 = h.T @ d; gb2 = d.sum()
        dz = np.outer(d, W2) * (z > 0)
        gW1 = (dz * X[:, None]).sum(0); gb1 = dz.sum(0)
        W1 -= lr * gW1; b1 -= lr * gb1; W2 -= lr * gW2; b2 -= lr * gb2
    return out

def reg_net_predict(params, S):
    W1, b1, W2, b2 = params[:4]
    x = (np.asarray(S) - HAL_SMU) / HAL_SSD
    return np.maximum(np.outer(x, W1) + b1, 0) @ W2 + b2

# ----------------------------------------------------------------------------- overfitting: few, noisy batches, big network
rngO = np.random.default_rng(31)
OF_NTR, OF_NVA = 14, 200
def _of_draw(n):
    S = rngO.uniform(0.2, 20, n)
    return S, haldane(S) + rngO.normal(0, 0.035, n)
OF_STR, OF_YTR = _of_draw(OF_NTR)
OF_SVA, OF_YVA = _of_draw(OF_NVA)
