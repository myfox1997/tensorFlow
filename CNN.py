import input_data
import tensorflow as tf

# load the data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

print('训练集信息：')
print(mnist.train.images.shape, mnist.train.labels.shape)
print('测试集信息：')
print(mnist.test.images.shape, mnist.test.labels.shape)
print('验证集信息：')
print(mnist.validation.images.shape, mnist.validation.labels.shape)

# start the function
# 创建权重和偏置，用一个小正数初始化权重


def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

# 卷积和池化  池化用于采样降维


def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

# end of the function

sess = tf.InteractiveSession()  # sess


x = tf.placeholder(tf.float32, shape=[None, 784])  # place holder x 28*28
y_ = tf.placeholder(tf.float32, shape=[None, 10])  # class of img

x_image = tf.reshape(x, [-1, 28, 28, 1])  # [-1, width, height, channel]

# first layer of Convolution
W_conv1 = weight_variable([5, 5, 1, 32])  # 卷积模版，5*5  1通道输入  32通道输出
b_conv1 = bias_variable([32])   # bias for each channel

h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)  # rule激活函数
h_pool1 = max_pool_2x2(h_conv1)

# the second layer
W_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = bias_variable([64])

h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

# reshape the img
W_fc1 = weight_variable([7 * 7*64, 1024])
b_fc1 = bias_variable([1024])

h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

# dropout part : reduce the overFitting
keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)  # 用dropout函数的一个占位符表示神经元不变的概率

# output layer
W_fc2 = weight_variable([1024, 10])
b_fc2 = bias_variable([10])

y_conv = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

# train and evaluate the module
cross_entropy = -tf.reduce_sum(y_ * tf.log(y_conv))  # 计算交叉熵
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
sess.run(tf.initialize_all_variables())
for i in range(20000):
    batch = mnist.train.next_batch(50)
    if i % 100 == 0:
        train_accuracy = accuracy.eval(feed_dict={x: batch[0], y_: batch[1], keep_prob: 1.0})
        print("step %d, train accuracy %g" % (i, train_accuracy))
    train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})

print("test accuracy %g" % accuracy.eval(feed_dict={x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))
# W = tf.Variable(tf.zeros([784, 10]))
# b = tf.Variable(tf.zeros([10]))
#
# sess.run(tf.initialize_all_variables())  # init
#
# # define the module
# y = tf.nn.softmax(tf.matmul(x, W) + b)
# cross_entropy = -tf.reduce_sum(y_ * tf.log(y))
# train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)
#
# # train the data
# for i in range(1000):
#     batch = mnist.train.next_batch(50)
#     train_step.run(feed_dict={x: batch[0], y_: batch[1]})
#
# # evaluate the module
# correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
# accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
# print(accuracy.eval(feed_dict={x: mnist.test.images, y_: mnist.test.labels}))
