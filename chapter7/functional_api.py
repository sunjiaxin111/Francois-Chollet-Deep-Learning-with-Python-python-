from keras import Input, layers
from keras.models import Sequential, Model
import numpy as np

# 把层看作一个函数
# 这是一个张量
input_tensor = Input(shape=(32,))

# 一层是一个函数
dense = layers.Dense(32, activation='relu')

# 一层调用一个张量，然后返回一个张量
output_tensor = dense(input_tensor)

# 与序列模型等价的functional API
# 序列模型
seq_model = Sequential()
seq_model.add(layers.Dense(32, activation='relu', input_shape=(64,)))
seq_model.add(layers.Dense(32, activation='relu'))
seq_model.add(layers.Dense(10, activation='softmax'))

# functional API
input_tensor = Input(shape=(64,))
x = layers.Dense(32, activation='relu')(input_tensor)
x = layers.Dense(32, activation='relu')(x)
output_tensor = layers.Dense(10, activation='softmax')(x)

# 把一个输入张量和一个输出张量变成一个模型
model = Model(input_tensor, output_tensor)

model.summary()

# 如果输入张量和输出张量没有关联,将会报错
# unrelated_input = Input(shape=(32,))
# bad_model = Model(unrelated_input, output_tensor)

# 编译、训练和评估是一样的
# 编译模型
model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy')

# 生成Numpy的数据去训练
x_train = np.random.random((1000, 64))
y_train = np.random.random((1000, 10))

# 训练模型
model.fit(x_train, y_train, epochs=10, batch_size=128)

# 评估模型
score = model.evaluate(x_train, y_train)
