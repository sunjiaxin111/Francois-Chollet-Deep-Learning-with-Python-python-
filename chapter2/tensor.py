from keras.datasets import mnist
import numpy as np
import matplotlib.pyplot as plt

# 标量
x = np.array(12)
print(x)
print(x.shape)
print(x.ndim)

# 向量
x = np.array([12, 3, 6, 14])
print(x)
print(x.shape)
print(x.ndim)

# 矩阵
x = np.array([[5, 78, 2, 34, 0],
              [6, 79, 3, 35, 1],
              [7, 80, 4, 36, 2]])
print(x.shape)
print(x.ndim)

# 3维张量及高维张量
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
print(x.shape)
print(x.ndim)

# 查看MNIST数据集的维度和尺寸
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
print(train_images.ndim)
print(train_images.shape)
print(train_images.dtype)

# 画出训练集中的一张图片
digit = train_images[4]

plt.imshow(digit, cmap=plt.cm.binary)
plt.show()

# tensor切片
my_slice = train_images[10:100]
print(my_slice.shape)

# 等价切片1
my_slice = train_images[10:100, :, :]
print(my_slice.shape)

# 等价切片2
my_slice = train_images[10:100, 0:28, 0:28]
print(my_slice.shape)

# 切右下角的14*14像素
my_slice = train_images[:, 14:, 14:]

# 切正中间的14*14像素
my_slice = train_images[:, 7:-7, 7:-7]

# 批量数据
batch = train_images[:128]
# 下一批数据
batch = train_images[128:256]
# 第n个批量
n = 2
batch = train_images[128 * n:128 * (n + 1)]
