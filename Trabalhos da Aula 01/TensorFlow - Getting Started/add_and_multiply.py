import tensorflow as tf

a = tf.placeholder(tf.float32)
b = tf.placeholder(tf.float32)
adder_node = a + b # + provides a shortcut for tf.add(a, b)
add_and_triple = adder_node * 3.

sess = tf.Session()

print(sess.run(
	add_and_triple, {a: 3, b: 4.5})
)