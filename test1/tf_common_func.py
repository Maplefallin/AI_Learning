import tensorflow as tf
import numpy as np
from tensorboard.compat.tensorflow_stub.dtypes import float32, int32


a = tf.constant([1,5],dtype=float32)

#创建全为0的张量
z = tf.zeros([2,3])

#创建全为1的张量
o = tf.ones(3)

#创建指定值的张量
f = tf.fill([4,3],4)

#生成正态分布的随机数，默认均值为0，标准差为1
rn = tf.random.normal([3,3],2,1)

#生成截断式正态分布的随机数，随机数更加直接
rtn = tf.random.truncated_normal([3,3],2,1)

#生成均匀分布随机数 [minval,maxval)
ru = tf.random.uniform([3,3],minval=2,maxval=4)

#强制类型转换
c = tf.cast(a,dtype=int32)

#计算张量维度上的最大值和最小值
rmax = tf.reduce_max(a)
rmin = tf.reduce_min(a)

#axois 指定操作维度，axios=0纵向操作，axios=1横向操作
x = tf.constant([[1,2,3],
                 [1,2,3]])
# print(tf.reduce_sum(x,1))
# print(tf.reduce_mean(x,0))

#tf.Variable()将变量标记为可训练变量，标记后的变量会在驯良过程中记录梯度
w = tf.Variable(tf.random.normal([2,2],0,1))

#tf.matmul来实现两个矩阵的相乘
tf.matmul(tf.ones([3,2]),tf.fill([2,3],3.0))

#将标签和特征配对
features = tf.constant([12,23,10,17])
labels = tf.constant([0,1,1,0])
dataset = tf.data.Dataset.from_tensor_slices((features,labels))
print(dataset)
for element in dataset:
    print(element)

#with结构记录计算过程，gradient求出张量梯度
with tf.GradientTape() as tape:
    w = tf.Variable(tf.constant(3.0))
    loss = tf.pow(w,2)
grad = tape.gradient(loss,w)
print(grad)

#tf.one_hot()函数将待转换数据，转换为one-hot形式的书记输出
classes = 3
labels = tf.constant([1,0,2])
output = tf.one_hot(labels,depth=classes)
print(output)

# 返回张量沿指定维度最大值的索引 tf.argmax(张量名，axios=操作轴)
test = np.array([[1,2,3],[2,3,4],[3,4,5],[8,7,2]])
print (test)
print(tf.argmax(test, axis=0))
print(tf.argmax(test, axis=1))