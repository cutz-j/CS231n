import numpy as np


def logits(w, x):
    dot = w[0] * x[0] + w[1] * x[1] + w[2]
    return dot

def sigmoid(W, X):
    return 1. / (1. + np.exp(-logits(W, X)))
                 
w = [2, -3, -3]
x = [-1, -2]

## forward pass ##

f = sigmoid(w, x)

ddot = (1 - f) * f # chain rule
dx = [w[0] * ddot, w[1] * ddot] # chain rule 적용
dw = [x[0] * ddot, x[1] * ddot, 1.0 * ddot]

x = 3
y = -4

sigy = 1.0 / (1 + np.exp(-y))
num = x + sigy
sigx = 1.0 / (1 + np.exp(-x))
xpy = x + y
xpy_sq = np.square(xpy)
denom = sigx + xpy_sq
inv_denom = 1. / denom
f = num * inv_denom

# df / dnum #
dnum = num
# df / dinv #
dinv_denom = inv_denom
# dinv_denom / ddenom #
ddenom = (-1.0 / np.square(denom)) * dinv_denom
# ddenom / dsigx #
dsigx = 1 * ddenom
# ddenome / dxpy_Sq #
dxpy_sq = 1 * ddenom
# dxpy_sq / dxpy #
dxpy = (2 * xpy) * dxpy_sq
# dxpy / dx #
dx = (1) * dxpy
dy = (1) * dxpy

# dsigx #
dx += ((1 - sigx) * sigx) * dsigx
# dnum #
dx += (1) * dnum
dsigy = (1) * dnum
# dsigy #
dy += ((1-sigy) * sigy) * dsigy































