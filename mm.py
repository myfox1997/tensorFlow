import input_data
import tensorflow as tf

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
print("train: set: ")
print(mnist.train.images.shape, mnist.train.labels.shape)

sess = tf.InteractiveSession()
x = tf.placeholder(tf.float32, [None, 784])  # means each pic, it is a placeholder
W = tf.Variable(tf.zeros([784, 10]))  # weight
b = tf.Variable(tf.zeros([10]))  # a var

y = tf.nn.softmax(tf.matmul(x, W) + b)   # softMax module     mat nul : x * W

# use the cross-entropy to quantized the cost
# y_ = tf.placeholder(tf.float32, [None, 10])
y_ = tf.placeholder(tf.float32, [None, 10])

# cross_entropy = -tf.reduce_sum(y_ * tf.log(y))   # calculate the sum of the c-e
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))

# use the step = 0.01 to train the module
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)  # use the backPropagation algorithm

# init = tf.initialize_all_variables()
# sess = tf.Session()
# sess.run(init)
tf.global_variables_initializer().run()


for i in range(1000):
    batch_xs, batch_ys = mnist.train.next_batch(100)  # use 100 data to train the module
    # sess.run(train_step, feed_dict={x: batch_xs, y: batch_ys})
    train_step.run({x: batch_xs, y_: batch_ys})

correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))  # check the ans is right or not
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
print("MNIST pic: ")
print(sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))

