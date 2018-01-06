from keras import layers
from keras import Input

# 假定x是一个四维张量
x = Input(shape=(None,), dtype='int32')
y = layers.Conv2D(128, 3, activation='relu')(x)
y = layers.Conv2D(128, 3, activation='relu')(y)
y = layers.Conv2D(128, 3, activation='relu')(y)

# 把x加到y的输出中
y = layers.add([y, x])

# --------------------------------------------
# 当特征大小不一样时
# 假定x是一个四维张量
x = Input(shape=(None,), dtype='int32')
y = layers.Conv2D(128, 3, activation='relu')(x)
y = layers.Conv2D(128, 3, activation='relu')(y)
y = layers.MaxPooling2D(2, strides=2)(y)

# 使用线性的降低取样
residual = layers.Conv2D(1, strides=2)(x)

y = layers.add([y, residual])
