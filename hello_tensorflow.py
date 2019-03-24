import tensorflow as  tf
hello = tf.constant('hello ,tensorflow')

# print(hello)

a=tf.constant(10)
b=tf.constant(32)
c=tf.add(a,b)
sess=tf.Session()

print(sess.run([a,b,c]))
