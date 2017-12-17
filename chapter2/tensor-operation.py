import numpy as np


# 基于元素的relu原生实现
def naive_relu(x):
    # x是一个二维张量
    assert len(x.shape) == 2

    x = x.copy()  # 避免覆盖输入的张量
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            x[i, j] = max(x[i, j], 0)
    return x


# 基于元素的add原生实现
def naive_add(x, y):
    # x和y是二维张量
    assert len(x.shape) == 2
    assert x.shape == y.shape

    x = x.copy()  # 避免覆盖输入的张量
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            x[i, j] += y[i, j]
    return x


# 只能广播最后一个维度、最后两个维度。。。
x = np.array([[[5, 78, 2, 34, 0],
               [6, 79, 3, 35, 1],
               [7, 80, 4, 36, 2]],
              [[5, 78, 2, 34, 0],
               [6, 79, 3, 35, 1],
               [7, 80, 4, 36, 2]],
              [[5, 78, 2, 34, 0],
               [6, 79, 3, 35, 1],
               [7, 80, 4, 36, 2]],
              [[5, 78, 2, 34, 0],
               [6, 79, 3, 35, 1],
               [7, 80, 4, 36, 2]]])
# y = np.array([1, 2, 3])  # 维度不匹配
# y = np.array([1, 2, 3, 4])  # 维度不匹配
# y = np.array([1, 2, 3, 4, 5])
# y = np.array([[1, 2, 3, 4, 5],
#               [2, 3, 4, 5, 6],
#               [1, 2, 3, 4, 5],
#               [1, 2, 3, 4, 5]])  # 维度不匹配
y = np.array([[1, 2, 3, 4, 5],
              [2, 3, 4, 5, 6],
              [1, 2, 3, 4, 5]])
z = x + y
print(z)


# 矩阵加向量的原生实现
def naive_add_matrix_and_vector(x, y):
    # x是一个二维张量
    # y是一个向量
    assert len(x.shape) == 2
    assert len(y.shape) == 1
    assert x.shape[1] == y.shape[0]

    x = x.copy()  # 避免覆盖输入的张量
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            x[i, j] += y[j]
    return x


# x是一个随机张量，大小为(64, 3, 32, 10)
x = np.random.random((64, 3, 32, 10))
# y是一个随机张量，大小为(32, 10)
y = np.random.random((32, 10))

# z的大小为(64, 3, 32, 10)
z = np.maximum(x, y)
print(z.shape)


# 向量dot的原生实现
def naive_vector_dot(x, y):
    # x和y是向量
    assert len(x.shape) == 1
    assert len(y.shape) == 1
    assert x.shape[0] == y.shape[0]

    z = 0
    for i in range(x.shape[0]):
        z += x[i] * y[i]
    return z


# 矩阵-向量dot的原生实现
def naive_matrix_vector_dot(x, y):
    # x是矩阵
    # y是向量
    assert len(x.shape) == 2
    assert len(y.shape) == 1
    assert x.shape[1] == y.shape[0]

    z = np.zeros(x.shape[0])
    for i in range(x.shape[0]):
        # for j in range(x.shape[1]):
        #     z[i] += x[i][j] * y[j]
        z[i] = naive_vector_dot(x[i, :], y)
    return z


x = np.array([[1, 2, 3],
              [1, 2, 3]])
y = np.array([1, 2, 3])
z = naive_matrix_vector_dot(x, y)
print(z)


# 矩阵dot的原生实现
def naive_matrix_dot(x, y):
    # x和y是矩阵
    assert len(x.shape) == 2
    assert len(y.shape) == 2
    assert x.shape[1] == y.shape[0]

    z = np.zeros((x.shape[0], y.shape[1]))
    for i in range(y.shape[1]):
        z[:, i] = naive_matrix_vector_dot(x, y[:, i])
    return z


x = np.array([[1, 2, 3],
              [1, 2, 3]])
y = np.array([[1, 2, 3, 4],
              [1, 2, 3, 4],
              [1, 2, 3, 4]])
z = naive_matrix_dot(x, y)
print(z)
