from keras.applications.vgg16 import VGG16
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input, decode_predictions
import numpy as np
from keras import backend as K
import matplotlib.pyplot as plt
import cv2

model = VGG16(weights='imagenet')

# 预处理输入的图片
img_path = 'creative_commons_elephant.jpg'

# img是一个大小为（224,224）PIL图片
img = image.load_img(img_path, target_size=(224, 224))

# x是一个大小为（224,224,3）的数组
x = image.img_to_array(img)

# 添加一个batch维度,使x变为(1,224,224,3)
x = np.expand_dims(x, axis=0)

# 预处理
x = preprocess_input(x)

# 预测图片
preds = model.predict(x)
print('Predicted:', decode_predictions(preds, top=3)[0])

# 类别index
print(np.argmax(preds[0]))

# 构建Grad-CAM算法（Class Activation Map）
# african_elephant的预测值
african_elephant_output = model.output[:, 386]

# 最后一个卷积层
last_conv_layer = model.get_layer('block5_conv3')

# 算出african_elephant类关于block5_conv3的梯度
grads = K.gradients(african_elephant_output, last_conv_layer.output)[0]

# 这是一个大小为512的向量，每个值对应一个channel
pooled_grads = K.mean(grads, axis=(0, 1, 2))

# 定义一个keras函数，输入input，输出pooled_grads和last_conv_layer.output[0]
iterate = K.function([model.input], [pooled_grads, last_conv_layer.output[0]])

# 得到这2个值
pooled_grads_value, conv_layer_output_value = iterate([x])

# 每个值乘以重要性
for i in range(512):
    conv_layer_output_value[:, :, i] *= pooled_grads_value[i]

# 得到heatmap
heatmap = np.mean(conv_layer_output_value, axis=-1)

# 正则化
heatmap = np.maximum(heatmap, 0)
heatmap /= np.max(heatmap)
plt.matshow(heatmap)
plt.show()

# 通过opencv把原图和heatmap叠加在一起，然后保存
# 读取原图
img = cv2.imread(img_path)

# 把heatmap调整为和原图一样大小
heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))

# 把heatmap转换为RGB
heatmap = np.uint8(255 * heatmap)

# 应用heatmap到原图
heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

# 0.4是强度因子
superimposed_img = heatmap * 0.4 + img

# 保存图片
cv2.imwrite('elephant_cam.jpg', superimposed_img)
