import numpy as np
import pandas as pd
import data_utils as ut

np.random.seed(777)

def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict1 = pickle.load(fo, encoding='bytes')
    return dict1


Xtr, Ytr, Xte, Yte = ut.load_CIFAR10('e:/CS231n/data/')
Xtr_rows = Xtr.reshape(Xtr.shape[0], 32*32*3)
Xte_rows = Xte.reshape(Xte.shape[0], 32*32*3)

Xte_cv = Xte_rows[:5000]
Yte_cv = Yte[:5000]
Xte_test = Xte_rows[5000:]
Yte_test = Yte[5000:]

def L(X, y, W, delta=1.0, ld=0.9):
    ## all vectorized computation ##
    ## no for loops ##
    ## l2 regularization ##
    ## bias --> x로 추가 ##
    # X = flatten x (3073, 50000) #
    # y = array of int --> corect (1, 50000) #
    # W = weight (10, 3073) #
    scores = np.dot(W, X) # (10, 3073)(3073, 50000)=(10, 50000)
#    y_one_hot = np.array(pd.get_dummies(y[0,:])).T # (10, 50000)
#    y_score = np.sum(y * y_one_hot, axis=0) # (1, 50000)
    margins = scores - y + delta #broadcasting
#    margins[margins==delta] == 0 # 중복 제거
#    margins = np.max(margins, axis=0) # 행별 max
#    loss_i = np.sum(margins)
    for i in range(X.shape[1]): # 중복 제거 ! (대체 코딩 찾기)
        margins[y[i],i] = 0
    margins = np.max(margins, axis=0)
    loss_i = np.sum(margins) / X.shape[1]
    reg = ld * np.sum(np.square(W))
    return loss_i + reg

X_train = Xtr_rows.T
Y_train = Ytr
X_train = np.insert(X_train, 0, 1.0, axis=0) # X에 bias항 추가

### random search ###
bestloss = float("inf") ## assigns very high
for num in range(1000):
    W = np.random.randn(10, 3073) * 0.0001 # random generator
    loss = L(X_train, Y_train, W, ld=0.0)
    if loss < bestloss:
        bestloss = loss  ## randomized loss test
        bestW = W
    if num % 50 == 0:
        print("in attempt %d the loss was %f, best %f" % (num, loss, bestloss))

X_test = Xte_test.T # shape(3072, 5000)
X_test = np.insert(X_test, 0, 1.0, axis=0) # add bias
y_test = Yte_test # shape(10, 5000)

scores = np.dot(bestW, X_test) # (10, 3073) (3073, 5000)
y_hat = np.argmax(scores, axis=0) # value가 가장 높은 값 index return
acc = np.mean(y_hat == y_test)
print(acc) # 0.1314 --> 13%

## Random Local Search ##
## Loss가 변화된다면, W에 step_size를 곱해 진행 ##
W = np.random.randn(10, 3073) * 0.001
bestloss = float("inf")
for i in range(1000):
    step_size = 0.0001
    Wtry = W + np.random.randn(10, 3073) * step_size
    loss = L(X_train, Y_train, Wtry, delta=1.0, ld=0.0)
    if loss < bestloss:
        W = Wtry
        bestloss = loss
    if i % 100 == 0:
        print("iter %d loss is %f" %(i ,bestloss))


def f(x):
    ## 임의의 함수 ##
    return x^2

def eval_numerical_gradient(f, x):
    ## f --> input 1 ##
    ## x --> np.array ##
    ## 수치적 미분 ##
    fx = f(x) # 임의의 함숫값
    grad = np.zeros(x.shape) ## 수치 넣을 공간 생성
    h = 0.00001 # 매우 작은 수
    it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
    while not it.finished:
        ix = it.multi_index
        old_value = x[ix]
        x[ix] = old_value + h # 미분 분자
        fxh = f(x) # evaluate f(x+h) 함숫값
        x[ix] = old_value # 이전값을 가져오기
        
        grad[ix] = (fxh - fx) / h # 분자분모
        # 실제에서는 f(x+h) - f(x-h)를 더 많이 활용
        it.iternext()
    return grad

def CIFAR10_loss_fun(W):
    return L(X_train, Y_train, W)

W = np.random.rand(10, 3073) * 0.001
df = eval_numerical_gradient(CIFAR10_loss_fun, W)

loss_original = CIFAR10_loss_fun(W) # loss value
print("original loss: %f" % (loss_original, ))

for step_size_log in range(-10, 0, 1):
    step_size = 10 ** step_size_log # step_size 변화
    W_new = W - step_size * df # 수치그라디언트 값
    loss_new = CIFAR10_loss_fun(W_new)
    print("for step size %f new loss: %f" % (step_size, loss_new))


























