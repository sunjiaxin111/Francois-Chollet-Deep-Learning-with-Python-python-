from keras.models import load_model
from keras.preprocessing import image
import numpy as np
import matplotlib.pyplot as plt
from keras import models

model = load_model('cats_and_dogs_small_2.h5')
print(model.summary())

# 预处理单个图像
img_path = 'cats_and_dogs_small/test/cats/cat.1700.jpg'

# 把该图像预处理为四维张量
img = image.load_img(img_path, target_size=(150, 150))
img_tensor = image.img_to_array(img)
img_tensor = np.expand_dims(img_tensor, axis=0)
img_tensor /= 255.

print(img_tensor.shape)

# 展示图片
plt.imshow(img_tensor[0])
plt.show()
print('end')

# 提取前8层的输出
layer_outputs = [layer.output for layer in model.layers[:8]]
# 创建一个会返回这些输出的模型
activation_model = models.Model(inputs=model.input, outputs=layer_outputs)

# 在预测模式运行模型
activations = activation_model.predict(img_tensor)

# 第一个输出
first_layer_activation = activations[0]
print(first_layer_activation.shape)

# 打印第一个输出的4th通道(像边缘检测器)
plt.matshow(first_layer_activation[0, :, :, 4], cmap='viridis')

# 打印第一个输出的第7th通道(像眼睛)
plt.matshow(first_layer_activation[0, :, :, 7], cmap='viridis')
plt.show()
print('end')

# 画出所有输出
layer_names = []
for layer in model.layers[:8]:
    layer_names.append(layer.name)

images_per_row = 16

for layer_name, layer_activation in zip(layer_names, activations):
    # 特征数量
    n_features = layer_activation.shape[-1]

    # feature map大小为(1, size, size, n_features)
    size = layer_activation.shape[1]

    # 把这些通道拼到矩阵中
    n_cols = n_features // images_per_row
    display_grid = np.zeros((size * n_cols, images_per_row * size))

    # 把每个filter拼到大的水平网格中
    for col in range(n_cols):
        for row in range(images_per_row):
            channel_image = layer_activation[0, :, :, col * images_per_row + row]
            # 处理特征使得看的舒服
            channel_image -= channel_image.mean()
            channel_image /= channel_image.std()
            channel_image *= 64
            channel_image += 128
            channel_image = np.clip(channel_image, 0, 255).astype('uint8')
            display_grid[col * size:(col + 1) * size, row * size:(row + 1) * size] = channel_image

    # 展示网格
    scale = 1. / size
    plt.figure(figsize=(scale * display_grid.shape[1], scale * display_grid.shape[0]))
    plt.title(layer_name)
    plt.grid(False)
    plt.imshow(display_grid, aspect='auto', cmap='viridis')
    plt.show()

print('end')
