import string
import numpy as np

samples = ['The cat sat on the mat.', 'The dog ate my homework.']

# 所有可以打印出来的ASCII编码的字符
characters = string.printable

# 得到符号->索引的对应字典
token_index = dict(zip(characters, range(1, len(characters) + 1)))

max_length = 50
results = np.zeros((len(samples), max_length, max(token_index.values()) + 1))
for i, sample in enumerate(samples):
    for j, character in enumerate(sample[:max_length]):
        index = token_index.get(character)
        results[i, j, index] = 1.

print(results)
