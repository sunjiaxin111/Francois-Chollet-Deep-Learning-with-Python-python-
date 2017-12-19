# 查看房价数据集
from keras.datasets import boston_housing

(train_data, train_targets), (test_data, test_targets) = boston_housing.load_data()

# 查看数据
print(train_data.shape)
print(test_data.shape)

print(train_targets)

# 正规化数据
mean = train_data.mean(axis=0)
train_data -= mean
std = train_data.std(axis=0)
train_data /= std

test_data -= mean
test_data /= std

# 构建网络，当训练数据很少时，很容易过拟合，使用一个简单的网络是避免过拟合的一个方式
from keras import models
from keras import layers


def build_model():
    # 我们需要多次使用相同的模型，所以使用一个函数去构建它
    model = models.Sequential()
    model.add(layers.Dense(64, activation='relu', input_shape=(train_data.shape[1],)))
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(1))
    # mse是Mean Squared Error
    # mae是Mean Absolute Error
    model.compile(optimizer='rmsprop',
                  loss='mse',
                  metrics=['mae'])
    return model


# k分区的交叉验证
import numpy as np

k = 4
num_val_samples = len(train_data) // k
num_epochs = 100
all_scores = []
for i in range(k):
    print('processing fold #', i)
    # 准备分区i为验证数据
    val_data = train_data[i * num_val_samples: (i + 1) * num_val_samples]
    val_targets = train_targets[i * num_val_samples: (i + 1) * num_val_samples]

    # 准备其他的分区为训练数据
    partial_train_data = np.concatenate(
        [train_data[:i * num_val_samples],
         train_data[(i + 1) * num_val_samples:]],
        axis=0)
    partial_train_targets = np.concatenate(
        [train_targets[:i * num_val_samples],
         train_targets[(i + 1) * num_val_samples:]],
        axis=0)

    # 构建模型（已编译）
    model = build_model()
    # 训练模型
    model.fit(partial_train_data, partial_train_targets,
              epochs=num_epochs, batch_size=1, verbose=0)
    # 在验证集上评估模型
    val_mse, val_mae = model.evaluate(val_data, val_targets, verbose=0)
    print('使用分区' + str(i) + '作为验证集时，均方误差为' + str(val_mse) + ',平均绝对误差为' + str(val_mae))
    all_scores.append(val_mae)

print(all_scores)
print(np.mean(all_scores))
