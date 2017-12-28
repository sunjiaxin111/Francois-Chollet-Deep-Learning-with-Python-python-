from keras.layers import Embedding
from keras.datasets import imdb
from keras import preprocessing
from keras.models import Sequential
from keras.layers import Flatten, Dense

# Embedding层需要至少2个参数
# 一个是可能的符号数(1 + maximum word index)
# 另一个是embedding的维度
embedding_layer = Embedding(1000, 64)

# 读取IMDB数据
# 用来生成词向量空间的最大词数
max_features = 10000
# 只取每个评论的前20个词
maxlen = 20

# 读取数据
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=max_features)

# 把输入转换为二维整型张量
x_train = preprocessing.sequence.pad_sequences(x_train, maxlen=maxlen)
x_test = preprocessing.sequence.pad_sequences(x_test, maxlen=maxlen)

# 使用一个Embedding层和分类器
model = Sequential()
model.add(Embedding(10000, 8, input_length=maxlen))
model.add(Flatten())
model.add(Dense(1, activation='sigmoid'))
model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['acc'])
model.summary()

history = model.fit(x_train, y_train, epochs=10, batch_size=32, validation_split=0.2)
