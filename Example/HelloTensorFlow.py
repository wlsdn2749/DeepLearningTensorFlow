import tensorflow as tf
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


# create a constant op
# this op is added as anode to the default graph

def hello():
    hello = tf.constant("Hello, TensorFlow!")

    print(hello)


def computationalGraph():
    node1 = tf.constant(3.0, tf.float32)
    node2 = tf.constant(4.0) # also tf.float32 implicity
    node3 = tf.add(node1, node2)
    # equals  : node3 = node1 + node2

    print("node1:", node1, "node2:", node2)
    print("node3: ", node3)


def placeholder():
    @tf.function
    def adder(a, b):
        return a + b

    a = tf.constant(3, tf.float32)
    b = tf.constant(4.5, tf.float32)
    print(adder(a,b))



if __name__ == "__main__":
    #hello()
    #computationalGraph()
    placeholder()