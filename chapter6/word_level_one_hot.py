import numpy as np

samples = ['The cat sat on the mat.', 'The dog ate my homework.']

# 首先为数据中的每个符号构建一个索引
token_index = {}
for sample in samples:
    # 这里只是简单的用split函数分割,在实际应用时应该去除标点符号和特殊字符
    # split函数中不写分隔符，表示以任意空格为分隔符
    for word in sample.split():
        if word not in token_index:
            # 注意没有索引0
            token_index[word] = len(token_index) + 1

# 向量化我们的样本
# 对于每句话只考虑max_length个字
max_length = 10

# 结果
results = np.zeros((len(samples), max_length, max(token_index.values()) + 1))

# 结果赋值
for i, sample in enumerate(samples):
    for j, word in list(enumerate(sample.split()))[:max_length]:
        index = token_index.get(word)
        results[i, j, index] = 1.

print(results)
