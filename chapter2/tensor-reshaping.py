from keras.datasets import mnist
import numpy as np

(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

# 改形
train_images = train_images.reshape((60000, 28 * 28))

x = np.array([[0., 1.],
              [2., 3.],
              [4., 5.]])
print(x.shape)
x = x.reshape((6, 1))
print(x)
x = x.reshape((2, 3))
print(x)

# 矩阵转置
x = np.zeros((300, 20))
print(x.shape)
x = np.transpose(x)
print(x.shape)

'''
past_velocity = 0
momentum = 0.1  # 一个常量因子
while loss > 0.01:  # 优化循环
    w, loss, gradient = get_current_parameters()
    velocity = past_velocity * momentum - learning_rate * gradient  # 书上写的是加号，我认为是减号，看了keras的源码发现确实是减号
    w = w + momentum * velocity - learning_rate * gradient
    past_velocity = velocity
    update_parameter(w)
'''
