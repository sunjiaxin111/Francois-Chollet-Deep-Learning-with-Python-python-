from keras.datasets import mnist
from keras import models
from keras import layers
from keras.utils import to_categorical

# 读取Keras自带的mnist数据集
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

# 查看训练数据
print(train_images.shape)
print(len(train_labels))
print(train_labels)

# 查看测试数据
print(test_images.shape)
print(len(test_labels))
print(test_labels)

# 网络架构
network = models.Sequential()
network.add(layers.Dense(512, activation='relu', input_shape=(28 * 28,)))
network.add(layers.Dense(10, activation='softmax'))

# 编译
network.compile(optimizer='rmsprop',
                loss='categorical_crossentropy',
                metrics=['accuracy'])

# 预处理图像数据
train_images = train_images.reshape((60000, 28 * 28))
train_images = train_images.astype('float32') / 255

test_images = test_images.reshape((10000, 28 * 28))
test_images = test_images.astype('float32') / 255

# 预处理图像标签
train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)

# 训练网络
network.fit(train_images, train_labels, epochs=5, batch_size=128)

# 评价网络
test_loss, test_acc = network.evaluate(test_images, test_labels)
print('test_acc:', test_acc)
