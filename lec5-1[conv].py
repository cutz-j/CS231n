import numpy as np

X = np.random.randint(0, 255, size=(11, 11, 4))

x = 11
P = 0 # padding 0
F = 5 # filter size
S = 2 # stride 2

v = int((x + P*2 - F) / S + 1) # feature map size

V = np.zeros(shape=[v, v, 1], dtype=np.float32)

