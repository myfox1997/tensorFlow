import os
import tensorflow as tf
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# here is the first program

# hello = tf.constant('he')
# sess = tf.Session()
# print(sess.run(hello))  # print(sess.run(hello).decode())  without 'b'



# here is the second program

# sess = tf.InteractiveSession()
#
# x = tf.Variable([1.0, 2.0])
# a = tf.constant([3.0, 3.0])
#
# # init the x  with  initializer run()
# x.initializer.run()
#
# sub = tf.subtract(x, a)
# print(sub.eval())

# here is the third program
# state = tf.Variable(0, name='counter')
#
# # init a op, let state + 1
# one = tf.constant(1)
# new_value = tf.add(state, one)
# update = tf.assign(state, new_value)   # before  run(), it is not working
#
# # init the graph
# init_op = tf.global_variables_initializer()    # tf.initialize_all_variables()
#
# # start the graph
# with tf.Session() as sess:
#     # run the init_op
#     sess.run(init_op)
#     # show the initial state
#     print(sess.run(state))
#     # run the op, update the state and show it
#     for x in range(3):
#         sess.run(update)
#         print(sess.run(state))




