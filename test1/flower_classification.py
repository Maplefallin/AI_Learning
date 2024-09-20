#利用鸢尾花数据集，实现前向传播、反向传播、可视化loss曲线

import tensorflow as tf
from  sklearn import datasets
from matplotlib import pyplot as plt
import numpy as np


#导入数据集并划分特征和标签
x_data = datasets.load_iris().data
y_data = datasets.load_iris().target

#随机打乱数组
np.random.seed(116)
np.random.shuffle(x_data)
np.random.seed(116)
np.random.shuffle(y_data)
tf.random.set_seed(116)

#将打乱后的数据集分为训练集和测试集，训练集为前120行，测试集围殴后30行
x_train = x_data[:-30]
y_train = y_data[:-30]
x_test = x_data[-30:]
y_test = y_data[-30:]

#将测试机训练中的标签与特征一一对应,并且分批次
x_train = tf.cast(x_train,tf.float32)
x_test = tf.cast(x_train,tf.float32)

train_db = tf.data.Dataset.from_tensor_slices((x_train,y_train)).batch(32)
# test_db = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(32)

w1 = tf.Variable(tf.random.truncated_normal([4,3],stddev=0.1,seed=1))
b1 = tf.Variable(tf.random.truncated_normal([3],stddev=0.1,seed=1))

lr = 0.1
train_loss_results = []
test_acc = []
epoch = 500
loss_all = 0

#训练部分
for epoch in range(epoch):
    for step,(x_train,y_train) in enumerate(train_db):
        with tf.GradientTape() as tape:
            y = tf.matmul(x_train,w1)+b1
            y = tf.nn.softmax(y)
            y_ = tf.one_hot(y_train,depth=3)
            loss = tf.reduce_mean(tf.square(y_-y))
            loss_all += loss.numpy()
        grads = tape.gradient(loss,[w1,b1])

        w1.assign_sub(lr * grads[0])  # 参数w1自更新
        b1.assign_sub(lr * grads[1])  # 参数b自更新

    print("Epoch {}, loss: {}".format(epoch, loss_all / 4))
    train_loss_results.append(loss_all / 4)  # 将4个step的loss求平均记录在此变量中
    loss_all = 0  # loss_all归零，为记录下一个epoch的loss做准备

