from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image
import os
import matplotlib.pyplot as plt
from keras import models
from keras import layers
from keras import optimizers

# 通过ImageDataGenerator来构建一个数据增强功能
datagen = ImageDataGenerator(
    # 随机转动的角度（0-180）
    rotation_range=40,
    # 以图像的长宽百分比为变化范围进行平移[0, 1]
    width_shift_range=0.2,
    height_shift_range=0.2,
    # 水平或垂直投影变换
    shear_range=0.2,
    # 按比例随机缩放图像尺寸
    zoom_range=0.2,
    # 水平翻转图像
    horizontal_flip=True,
    # 填充像素(出现在旋转或平移之后)
    fill_mode='nearest')

# 展示随机增强的训练图片
train_cats_dir = 'cats_and_dogs_small/train/cats'
fnames = [os.path.join(train_cats_dir, fname) for fname in os.listdir(train_cats_dir)]

# 选择一个图像去增强
img_path = fnames[3]

# 读取图像并且调整大小到150*150
img = image.load_img(img_path, target_size=(150, 150))

# 把图像转换为大小(150, 150, 3)的数组
x = image.img_to_array(img)

# 改变数组维数到(1, 150, 150, 3)
x = x.reshape((1,) + x.shape)

# flow命令将会无限循环，所以要break
i = 0
for batch in datagen.flow(x, batch_size=1):
    plt.figure(i)
    imgplot = plt.imshow(image.array_to_img(batch[0]))
    i += 1
    if i % 4 == 0:
        break

plt.show()

# 定义一个包含dropout的新模型
model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Flatten())
model.add(layers.Dropout(0.5))
model.add(layers.Dense(512, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy',
              optimizer=optimizers.RMSprop(lr=1e-4),
              metrics=['acc'])

train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)

# 验证集不应该被增强
validation_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    # 训练集文件夹
    'cats_and_dogs_small/train',
    # 所有图片都调整大小到150*150
    target_size=(150, 150),
    batch_size=20,
    # 因为要使用binary_crossentropy,所以需要binary标签
    class_mode='binary')

validation_generator = validation_datagen.flow_from_directory(
    'cats_and_dogs_small/validation',
    target_size=(150, 150),
    batch_size=20,
    class_mode='binary')

# 训练模型
history = model.fit_generator(
    train_generator,
    steps_per_epoch=100,
    epochs=100,
    validation_data=validation_generator,
    validation_steps=50)

# 保存模型
model.save('cats_and_dogs_small_2.h5')

# 画出训练过程中损失函数值和准确率值的变化曲线
acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(1, len(acc) + 1)

plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend()

plt.figure()

plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()

plt.show()
print('end')
