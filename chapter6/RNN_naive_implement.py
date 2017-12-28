import numpy as np

# 时间步数
timesteps = 100
# 输入特征维度
inputs_features = 32
# 输出特征维度
output_features = 64

# 输入数据
inputs = np.random.random((timesteps, inputs_features))

# 初始状态
state_t = np.zeros((output_features,))

# 创建权重矩阵
W = np.random.random((inputs_features, output_features))
U = np.random.random((output_features, output_features))
b = np.random.random((output_features,))

successive_outputs = []
for input_t in inputs:
    # 使用当前输入和当前状态（前一个输出）去得到当前输出
    output_t = np.tanh(np.dot(W, input_t) + np.dot(U, state_t) + b)

    # 存储输出
    successive_outputs.append(output_t)

    # 更新状态
    state_t = output_t

# 最终的输出（timesteps, output_features）
final_output_sequence = np.concatenate(successive_outputs, axis=0)
