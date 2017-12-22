from keras.applications import VGG16

# weights参数指定初始化参数为imagenet（在ImageNet上预训练）的参数
# include_top参数指定是否包含全连接层
conv_base = VGG16(weights='imagenet',
                  include_top=False,
                  input_shape=(150, 150, 3))

print(conv_base.summary())
