import numpy as np

X = np.random.randint(0, 255, size=(11, 11, 4))

x = 11
P = 0 # padding 0
F = 5 # filter size
S = 2 # stride 2

v = int((x + P*2 - F) / S + 1) # feature map size
W0 = np.random.normal(size=(5,5,4))
b0 = np.array(1)

V = np.zeros(shape=[v, v, 1], dtype=np.float32)


V[0, 0, 0] = np.sum(X[:5, :5, :] * W0) + b0
V[1, 0, 0] = np.sum(X[2:7, :5, :] * W0) + b0
V[2, 0, 0] = np.sum(X[4:9, :5, :] * W0) + b0
V[3, 0, 0] = np.sum(X[6:11, :5, :] * W0) + b0
