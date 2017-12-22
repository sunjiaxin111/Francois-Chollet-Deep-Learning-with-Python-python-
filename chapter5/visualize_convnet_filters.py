from keras.applications import VGG16
from keras import backend as K
import numpy as np
import matplotlib.pyplot as plt

model = VGG16(weights='imagenet',
              include_top=False)

# layer_name = 'block3_conv1'
# filter_index = 0
#
# layer_output = model.get_layer(layer_name).output
# loss = K.mean(layer_output[:, :, :, filter_index])
#
# # 获得loss关于输入的梯度
# grads = K.gradients(loss, model.input)[0]
#
# # 正则化梯度，这里加上1e-5是为了防止除0
# grads /= (K.sqrt(K.mean(K.square(grads))) + 1e-5)
#
# # 定义一个keras函数，接受一个输入，产生loss和grads
# iterate = K.function([model.input], [loss, grads])
# loss_value, grads_value = iterate([np.zeros(1, 150, 150, 3)])
#
# # 通过随机梯度下降使得loss最大化
# # 我们从一个带有噪声的图像开始
# input_img_data = np.random.random((1, 150, 150, 3)) * 20 + 128.
#
# # 运行40步梯度上升
# step = 1
# for i in range(40):
#     loss_value, grads_value = iterate([input_img_data])
#     input_img_data += grads_value * step


# 把一个tensor转变为一个有效图片
def deprocess_image(x):
    # 以平均值为0.、标准差为0.1来正则化
    x -= x.mean()
    x /= (x.std() + 1e-5)
    x *= 0.1

    # 裁剪,把x的元素限制在0-1之间
    x += 0.5
    x = np.clip(x, 0, 1)

    # 转换为RGB数组
    x *= 255
    x = np.clip(x, 0, 255).astype('uint8')
    return x


# 生成filter可视化
def generate_pattern(layer_name, filter_index, size=150):
    # 构建一个使nth个filter输出最大化的损失函数
    layer_output = model.get_layer(layer_name).output
    loss = K.mean(layer_output[:, :, :, filter_index])

    # 计算梯度
    grads = K.gradients(loss, model.input)[0]

    # 正则化梯度
    grads /= (K.sqrt(K.mean(K.square(grads))) + 1e-5)

    # 定义一个keras函数，接受一个输入，产生loss和grads
    iterate = K.function([model.input], [loss, grads])

    # 从一个带有噪声的图像开始
    input_img_data = np.random.random((1, size, size, 3)) * 20 + 128.

    # 运行40步梯度上升
    step = 1
    for i in range(40):
        loss_value, grads_value = iterate([input_img_data])
        input_img_data += grads_value * step

    img = input_img_data[0]
    return deprocess_image(img)


plt.imshow(generate_pattern('block3_conv1', 0))
plt.show()

# 在1层中生成所有过滤器对应的模式
layer_name = 'block4_conv1'
size = 64
margin = 5

# 用来存储结果的空图像
results = np.zeros((8 * size + 7 * margin, 8 * size + 7 * margin, 3))

for i in range(8):
    for j in range(8):
        # 生成模式
        filter_img = generate_pattern(layer_name, i + (j * 8), size = size)

        # 放入结果网格中
        horizontal_start = i * size + i * margin
        horizontal_end = horizontal_start + size
        vertical_start = j * size + j * margin
        vertical_end = vertical_start + size
        results[horizontal_start: horizontal_end, vertical_start: vertical_end, :] = filter_img

# 展示结果网格
plt.figure(figsize=(20, 20))
plt.imshow(results)
plt.show()
