import tensorflow as tf

def tf_leaky_relu(x):
	alpha = tf.constant(0.01)
	cond = tf.less(x, tf.constant(0.0))
	return tf.where(cond, alpha*x, x)

with tf.Session() as sess:
	W = tf.Variable([-1.0], dtype=tf.float32)
	x = tf.placeholder(tf.float32)
	y = tf.placeholder(tf.float32)

	modelo = tf_leaky_relu(W*x)
	# modelo = tf.nn.relu(W*x)

	loss = tf.reduce_sum(tf.square(modelo - y)) # sum of the squares
	otimizador = tf.train.GradientDescentOptimizer(0.01)
	train = otimizador.minimize(loss)

	# função objetivo: f(x) = a * x ; onde 'a' = 2.0
	x_train = [1, 2, 0]
	y_train = [2, 4, 0]

	init = tf.global_variables_initializer()
	sess = tf.Session()
	sess.run(init)

	print("\n*** Iniciando treinamento ***")
	tolerancia = 0.0001
	i = 0
	while(True):
		i += 1
		sess.run(train, {x: x_train, y: y_train})
		curr_W, curr_loss = sess.run([W, loss], {x: x_train, y: y_train})
		if(curr_loss <= tolerancia): break
		print(i, "Erro atual: ", curr_loss)

	print(i, "iterações para convergir")
	# evaluate training accuracy
	curr_W, curr_loss = sess.run([W, loss], {x: x_train, y: y_train})
	print("W: %s loss: %s"%(curr_W, curr_loss))
	
	# print(W.eval())
	# print(x.eval())
	# print(y.eval())
	# print(tf.gradients(y, [x])[0].eval())