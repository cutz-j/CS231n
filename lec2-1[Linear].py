import numpy as np
import pandas as pd

### loss function ###
def loss_i(x, y, W):
    ## unvectorized version ##
    ## x = image 1개 --> faltten (1 x 3073).T ##
    ## y = answer classes "index"! (10) (1, 10) ##
    ## w = weight (3073 x 10).T ##
    
    delta =  1.0 # SVM Loss needs delta
    scores = np.dot(W, x) # (10 x 3073)(3703 x 1) ## 전체 ##
    correct_class_score = scores[y] ## 뺄셈을 할 정답 클래스 ##
    D = W.shape[0] # 3073 --> image dimension
    loss_i = 0.0 # summation
    for j in range(D):
        if j == y: # 정답클래스 중복 계산 제외 #
            continue
        loss_i += np.max(0, scores[j] - correct_class_score + delta) # 수식
    return loss_i

def loss_i_vectorized(x, y, W):
    ## vectorized computation ##
    ## numpy ##
    delta = 1.0
    scores = np.dot(W, x)
    margins = np.max(0, scores - scores[y] + delta)
    margins[y] = 0 # 중복연산 방지
    loss_i = np.sum(margins) # sigma --> 정답 클래스는 0이기 때문에 무관
    return loss_i

def L(X, y, W, delta=1.0, ld=0.9):
    ## all vectorized computation ##
    ## no for loops ##
    ## l2 regularization ##
    ## bias --> x로 추가 ##
    # X = flatten x (3073, 50000) #
    # y = array of int --> corect (1, 50000) #
    # W = weight (10, 3073) #
    scores = np.dot(W, X) # (10, 3073)(3073, 50000)=(10, 50000)
    y_one_hot = np.array(pd.get_dummies(y[0,:])).T # (10, 50000)
    y_score = np.sum(y * y_one_hot, axis=0) # (1, 50000)
    margins = scores - y_score + delta #broadcasting
    margins[margins==delta] == 0 # 중복 제거
    margins = np.max(margins, axis=0) # 행별 max
    loss_i = np.sum(margins)
    reg = ld * np.sum(np.square(W))
    return loss_i + reg

