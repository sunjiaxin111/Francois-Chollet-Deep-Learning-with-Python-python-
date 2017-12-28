from keras.preprocessing.text import Tokenizer

samples = ['The cat sat on the mat.', 'The dog ate my homework.']

# 取最通用的1000个词
tokenizer = Tokenizer(num_words=1000)

# 构建word->index的对应关系
tokenizer.fit_on_texts(samples)

# 把句子转换为搜索引列表
sequences = tokenizer.texts_to_sequences(samples)

# 也可以直接转成二元表示
one_hot_results = tokenizer.texts_to_matrix(samples, mode='binary')

# 获得word->index的字典
word_index = tokenizer.word_index
print('Found %s unique tokens.' % len(word_index))
