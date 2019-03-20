### transfer learning and CNN visualizing ###

from __future__ import print_function
import time, os, json
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from utils.classifiers.squeezenet import SqueezeNet
from utils.data_utils import load_tiny_imagenet
from utils.image_utils import preprocess_image, deprocess_image
from utils.image_utils import SQUEEZENET_MEAN, SQUEEZENET_STD

plt.rcParams['figure.figsize'] = (10.0, 8.0)
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

def get_session():
    '''
    세션과 메모리 할당
    '''
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = False
    session = tf.Session(config=config)
    return session

tf.reset_default_graph() # 저장된 노드 그래프 삭제
sess = get_session() # 그래프 불러오기 (설정) // session
SAVE_PATH = 'utils/datasets/squeezenet.ckpt'
model = SqueezeNet(save_path=SAVE_PATH, sess=sess)




























