import keras
import numpy as np
from keras import layers
import random
import sys

# 下载并解析数据
path = keras.utils.get_file('nietzsche.txt',
                            origin='https://s3.amazonaws.com/text-datasets/nietzsche.txt')
text = open(path).read().lower()
print('Corpus length:', len(text))

# 向量化序列字符
# 提取序列字符的最大长度
maxlen = 60

# 取样一个新的序列每step个字符
step = 3

# 用下面的list保存提取出来的序列
sentences = []

# 用下面的list保存targets（接下来的字符）
next_chars = []

# 取样
for i in range(0, len(text) - maxlen, step):
    sentences.append(text[i: i + maxlen])
    next_chars.append(text[i + maxlen])
print('Number of sequences:', len(sentences))

# 语料库中独一无二字符的list
chars = sorted(list(set(text)))
print('Unique characters:', len(chars))
# 用字典映射字符和它们在chars中的下标
char_indices = dict((char, chars.index(char)) for char in chars)

# 使用one-hot把字符编码成二进制数组
print('Vectorization...')
x = np.zeros((len(sentences), maxlen, len(chars)), dtype=np.bool)
y = np.zeros((len(sentences), len(chars)), dtype=np.bool)
for i, sentence in enumerate(sentences):
    for t, char in enumerate(sentence):
        x[i, t, char_indices[char]] = 1
    y[i, char_indices[next_chars[i]]] = 1

# 构建网络模型
model = keras.models.Sequential()
model.add(layers.LSTM(128, input_shape=(maxlen, len(chars))))
model.add(layers.Dense(len(chars), activation='softmax'))

# 模型编译
optimizer = keras.optimizers.RMSprop(lr=0.01)
model.compile(loss='categorical_crossentropy', optimizer=optimizer)


# 取样函数
def sample(preds, temperature=1.0):
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    # multinomial函数用来取样
    # 第一个参数代表每次实验的实验次数
    # 第二个参数代表样本的概率分布
    # 第三个参数代表实验几次
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)


# 生成文本的循环
for epoch in range(1, 60):
    print('epoch', epoch)
    # 让模型在训练数据上训练一个epoch
    model.fit(x, y, batch_size=128, epochs=1)

    # 随机选一个初始文本
    start_index = random.randint(0, len(text) - maxlen - 1)
    generated_text = text[start_index: start_index + maxlen]
    print('---Generating with seed: "' + generated_text + '"')

    for temperature in [0.2, 0.5, 1.0, 1.2]:
        print('------ temperature:', temperature)
        generated_text = text[start_index: start_index + maxlen]
        sys.stdout.write(generated_text)

        # 生成400个字符
        for i in range(400):
            sampled = np.zeros((1, maxlen, len(chars)))
            for t, char in enumerate(generated_text):
                sampled[0, t, char_indices[char]] = 1.

            preds = model.predict(sampled, verbose=0)[0]
            next_index = sample(preds, temperature)
            next_char = chars[next_index]
            generated_text += next_char
            generated_text = generated_text[1:]

            sys.stdout.write(next_char)
            sys.stdout.flush()
        print()
