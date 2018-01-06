from keras import layers
from keras import applications
from keras import Input

# 共用convolutional base
xception_base = applications.Xception(weights=None, include_top=False)

# 输入为250*250的RGB图像
left_input = Input(shape=(250, 250, 3))
right_input = Input(shape=(250, 250, 3))

# 调用xception_base两次
left_features = xception_base(left_input)
right_features = xception_base(right_input)

merged_features = layers.concatenate([left_features, right_features], axis=-1)
