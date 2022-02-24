import tensorflow as tf
hello = tf.constant("Hello tensorflow!")

#sess = tf.Session()
#tensorflow 2.0에서는 Session이 생략된다.


print(sess.run(hello))