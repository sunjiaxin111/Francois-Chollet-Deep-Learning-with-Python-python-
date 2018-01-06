from keras.models import Model
from keras import layers
from keras import Input
import numpy as np

text_vocabulary_size = 10000
question_vocabulary_size = 10000
answer_vocabulary_size = 500

# 文本输入是变长的序列
text_input = Input(shape=(None,), dtype='int32', name='text')

# 把文本输入变成大小为64的词向量
embedded_text = layers.Embedding(64, text_vocabulary_size)(text_input)

# LSTM
encoded_text = layers.LSTM(32)(embedded_text)

# 对question做相同步骤
question_input = Input(shape=(None,), dtype='int32', name='question')
embedded_question = layers.Embedding(32, question_vocabulary_size)(question_input)
encoded_question = layers.LSTM(16)(embedded_question)

# 连接encoded_text和encoded_question
concatenated = layers.concatenate([encoded_text, encoded_question], axis=-1)

# 添加一个softmax分类器
answer = layers.Dense(answer_vocabulary_size, activation='softmax')(concatenated)

# 指定二个输入和一个输出
model = Model([text_input, question_input], answer)
model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['acc'])

# 训练模型
# 生成一些模拟数据
num_samples = 500
max_length = 500
text = np.random.randint(1, text_vocabulary_size, size=(num_samples, max_length))
question = np.random.randint(1, question_vocabulary_size, size=(num_samples, max_length))

# one-hot编码
answers = np.random.randint(0, 1, size=(num_samples, answer_vocabulary_size))

# 输入list
model.fit([text, question], answers, epochs=10, batch_size=128)

# 字典方式
model.fit({'text': text, 'question': question}, answers, epochs=10, batch_size=128)
