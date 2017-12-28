import numpy as np

samples = ['The cat sat on the mat.', 'The dog ate my homework.']

# 将词储存为1000维的向量，如果有接近或超过1000个词，可能会有hash碰撞
dimensionality = 1000
max_length = 10

results = np.zeros((len(samples), max_length, dimensionality))
for i, sample in enumerate(samples):
    for j, word in list(enumerate(sample.split()))[:max_length]:
        # 将一个词hash为随机index
        index = abs(hash(word)) % dimensionality
        results[i, j, index] = 1.

print(results)
