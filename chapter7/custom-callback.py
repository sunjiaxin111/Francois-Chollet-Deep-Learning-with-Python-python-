import keras
import numpy as np


class ActivationLogger(keras.callbacks.Callback):

    def set_model(self, model):
        # 这个方法会在父模型训练前被调用
        self.model = model
        layer_outputs = [layer.output for layer in model.layers]
        # 这是一个model实例，返回每层的激活值
        self.activations_model = keras.models.Model(model.input, layer_outputs)

    def on_epoch_end(self, epoch, logs=None):
        if self.validation_data is None:
            raise RuntimeError('Requires validation_data.')

        # 获得验证集的第一个输入样本
        validation_sample = self.validation_data[0][0:1]
        activations = self.activations_model.predict(validation_sample)
        # 保存数组到硬盘
        f = open('activations_at_epoch_' + str(epoch) + '.npz', 'w')
        np.savez(f, activations)
        f.close()
