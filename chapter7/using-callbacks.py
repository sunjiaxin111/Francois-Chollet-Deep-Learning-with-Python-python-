import keras
from keras import Model

callbacks_list = [
    # 这个callback会在模型停止优化时打断训练
    keras.callbacks.EarlyStopping(
        # 监测验证集的准确度,这边应该是val_acc？？？
        monitor='acc',
        # 当准确率已经停止优化超过1个epochs，也就是2个epochs，训练将会被打断
        patience=1,
    ),
    # 这个callback将会在每个epoch后保存当前的权重
    keras.callbacks.ModelCheckpoint(
        filepath='my_model.h5',
        # 下面2个参数意味着除非val_loss有优化，不然model文件不会被重写
        monitor='val_loss',
        save_best_only=True,
    )
]

model = Model()

# 因为我们监测acc指标，acc应该成为一个度量标准
model.compile(optimizer='rmsprop',
              loss='binary_crossentropy',
              metrics=['acc'])

'''
model.fit(x, y,
          epochs=10,
          batch_size=32,
          callbacks=callbacks_list,
          validation_data=(x_val, y_val))
'''

# 使用ReduceLROnPlateau
callbacks_list = [
    keras.callbacks.ReduceLROnPlateau(
        # 监测验证集的loss
        monitor='val_loss',
        # 触发时把学习速率除以10
        factor=0.1,
        # 在验证集loss停止优化10个epochs时触发
        patience=10,
    )
]
