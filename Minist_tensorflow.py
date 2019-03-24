import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist=input_data.read_data_sets("./mnist/data/",one_hot=True)


#mnist 데이터는 28*28 픽셀=784개 특징 , 그리고 1부터 9를 적어놓았다
x=tf.placeholder(tf.float32,[None,784])
y=tf.placeholder(tf.float32,[None,10])

keep_prob=tf.placeholder(tf.float32) #드롭아웃의 주의 사앙 학습시 0.8 예측시에는 1 이 되야한다.
global_step=tf.Variable(0,trainable=False,name='global_step')
#784 입력, ->256 첫번쩨 은닉층,->256 두번째 은닉층->10 결과값
with tf.name_scope('layer1'):
    w1=tf.Variable(tf.random_normal([784,256],stddev=0.01),name='w1')
    b1=tf.Variable(tf.random_normal([256],stddev=0.01),name='b1')
    l1=tf.nn.relu(tf.add(tf.matmul(x,w1),b1))
    l1=tf.nn.dropout(l1,keep_prob) #해당 계층의 80퍼만 사용 하는 드롭아웃 기법
    tf.summary.histogram("weights1",w1)
    tf.summary.histogram("bias", b1)
with tf.name_scope('layer2'):
    w2=tf.Variable(tf.random_normal([256,256],stddev=0.01),name='w2')
    b2=tf.Variable(tf.random_normal([256],stddev=0.01),'name=b2')
    l2=tf.nn.relu(tf.add(tf.matmul(l1,w2),b2))
    l2=tf.nn.dropout(l2,keep_prob)
    tf.summary.histogram("weights2", w2)
    tf.summary.histogram("bias", b2)
with tf.name_scope('output'):
    w3=tf.Variable(tf.random_normal([256,10],stddev=0.01),name='w3')
    model=tf.matmul(l2,w3)


#코스트와 최적화
with tf.name_scope('optimizer'):
    cost=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=model,labels=y))
    optimizer=tf.train.AdamOptimizer(0.001).minimize(cost,global_step=global_step)
    tf.summary.scalar('cost', cost)


init=tf.global_variables_initializer()
sess=tf.Session()
sess.run(init)

batch_size=100
total_batch=int(mnist.train.num_examples/batch_size)

## tensorboard

merged=tf.summary.merge_all()
wirter=tf.summary.FileWriter('./logs2',sess.graph)

for epoch in range(15):
    for i in range(total_batch):
        batch_xs,batch_ys=mnist.train.next_batch(batch_size)

        _,cost_val=sess.run([optimizer,cost],feed_dict={x:batch_xs,y:batch_ys,keep_prob:0.8})#학습시 드롭아웃 적용
        total_cost=cost_val
    print('Epock:','%04d'%(epoch+1),'avg.cost=','{:3f}'.format(total_cost/total_batch))
    summary = sess.run(merged, feed_dict={x: batch_xs, y: batch_ys, keep_prob: 0.8})
    wirter.add_summary(summary,global_step=sess.run(global_step))

print('최적화 완료!')
is_correct=tf.equal(tf.argmax(model,1),tf.argmax(y,1))

accuracy=tf.reduce_mean(tf.cast(is_correct,tf.float32))
print('정학도:',sess.run(accuracy,feed_dict={x:mnist.test.images, y:mnist.test.labels,keep_prob:1}))#드롭아웃 제거



labels=sess.run(model,feed_dict={x:mnist.test.images,y:mnist.test.labels,keep_prob:1})
fig=plt.figure()


#그래프 부분
for i in range(10):
    subplot=fig.add_subplot(2,5,i+1)
    subplot.set_xticks([])
    subplot.set_yticks([])

    subplot.set_title('%d' %np.argmax(labels[i]))

    subplot.imshow(mnist.test.images[i].reshape((28,28)),cmap=plt.cm.gray_r)
plt.show()

sess.close()