from keras import layers
from keras import Input
from keras.models import Model

# 实例化一个LSTM层
lstm = layers.LSTM(32)

left_input = Input(shape=(None, 128))
left_output = lstm(left_input)

# 共用lstm层的权重
right_input = Input(shape=(None, 128))
right_output = lstm(right_input)

# 构建一个分类器
merged = layers.concatenate([left_output, right_output], axis=-1)
predictions = layers.Dense(1, activation='sigmoid')(merged)

# 实例化、训练模型
model = Model([left_input, right_input], predictions)
# model.fit([left_data, right_data], targets)
