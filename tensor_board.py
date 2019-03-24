import tensorflow as tf
import numpy as np



global_step=tf.Variable(0,trainable=False,name='global_step')

data=np.loadtxt('./data.csv',delimiter=',',unpack=False,dtype='float32')
x_data=data[0:, 0:2]
y_data=data[0:, 2:]

with tf.name_scope('later1'):
    x=tf.placeholder(tf.float32)
    y=tf.placeholder(tf.float32)

with tf.name_scope('layer2'):
    w1=tf.Variable(tf.random_uniform([2,10],-1,1))
    b1=tf.Variable(tf.zeros([10]))
    l1=tf.add(tf.matmul(x,w1),b1)
    l1=tf.nn.relu(l1)

with tf.name_scope('layer3'):
    w2=tf.Variable(tf.random_uniform([10,10],-1,1))
    b2=tf.Variable(tf.zeros([10]))
    l2=tf.add(tf.matmul(x,w1),b1)

with tf.name_scope('output'):
    w3=tf.Variable(tf.random_uniform([10,3],-1,1))
    b3=tf.Variable(tf.zeros([3]))
    model=tf.add(tf.matmul(l1,w3),b3)

with tf.name_scope('optimizer'):
    cost=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=y,logits=model))
    optimizer=tf.train.AdamOptimizer(learning_rate=0.01)
    train_op=optimizer.minimize(cost,global_step=global_step)

    tf.summary.scalar('cost',cost)



sess=tf.Session()
saver=tf.train.Saver(tf.global_variables())

ckpt=tf.train.get_checkpoint_state('./model') #체크 포인트
if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
    saver.restore(sess, ckpt.model_checkpoint_path)
else:
    sess.run(tf.global_variables_initializer())

merged=tf.summary.merge_all()
writer=tf.summary.FileWriter('./logs',sess.graph)

for step in range(100):
    sess.run(train_op,feed_dict={y:y_data,x:x_data})

    print('step: %d '% sess.run(global_step),'cost :%3f'% sess.run(cost,feed_dict={x:x_data,y:y_data}))
    summary=sess.run(merged, feed_dict={x:x_data,y:y_data})
    writer.add_summary(summary, global_step=sess.run(global_step))

saver.save(sess, './model/dnn.ckpt', global_step=global_step)

prediction=tf.argmax(model,1)
target=tf.argmax(y,1)
print('예측값:',sess.run(prediction,feed_dict={x:x_data}))
print('실제값:' ,sess.run(target,feed_dict={y:y_data}))

is_correct=tf.equal(prediction,target)
accuracy=tf.reduce_mean(tf.cast(is_correct,tf.float32))
print('정확도: %2f' %sess.run(accuracy*100,feed_dict={x:x_data,y:y_data}))


