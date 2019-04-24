from keras.models import Sequential, Model
from keras import layers

input_shape = (224, 224, 3)
input_data = layers.Input(shape=input_shape)
x = layers.Conv2D(32, 5, strides=1, padding='same')(input_data)
x = layers.Conv2D(32, 3, strides=1, padding='same')(x)
x = layers.MaxPooling2D(pool_size=(2, 2))(x)
x = layers.Conv2D(64, 3, strides=1, padding='same')(x)
x = layers.MaxPooling2D(pool_size=(2, 2))(x)
x = layers.Conv2D(128, 5, strides=1, padding='same')(x)
x = layers.MaxPooling2D(pool_size=(2, 2))(x)
x = layers.Flatten()(x)
x = layers.Dense(units=1000, bias=None)(x)
output = layers.Activation('softmax')(x)

net = Model(input_data, output)
