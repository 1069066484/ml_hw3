import sys
import os

from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow.python.ops import control_flow_ops
from tensorflow.python import debug as tf_debug


def model():
    x_raw = tf.placeholder(tf.float32, [None, 784])
    kp = tf.placeholder(tf.float32, [])
    y = tf.placeholder(tf.float32, [None, 10])
    x = tf.reshape(x_raw, [-1,28,28,1])
    is_training
    with slim.arg_scope([slim.conv2d, slim.fully_connected],
                        activation_fn=tf.nn.relu,
                        normalizer_fn=tf.layers.batch_normalization,
                        normalizer_params={'training': is_training, 'momentum': 0.95}):
        conv1 = slim.conv2d(x, 16, [5,5], scope='conv1')
        pool1 = slim.max_pool2d(conv1, [2,2], scope='pool1')
        conv2 = slim.conv2d(pool1, 32, [5,5], scope='conv2')
        pool2 = slim.max_pool2d(conv2, [2,2], scope='pool2')
        flattened = slim.flatten(pool2)
        fc = slim.fully_connected(flattened, 1024, scope='fc1')
        dropped_fc = slim.dropout(fc, keep_prob=kp)
        logits = slim.fully_connected(dropped_fc, 10, activation_fn=None, scope='logits')
    corrects = tf.equal(tf.arg_max(logits,1),tf.argmax(y,1))
    acc = tf.reduce_mean(tf.cast(corrects, tf.float32))
    cross_entropy = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=logits))
    step = tf.get_variable("step", [], initializer=tf.constant_initializer(0.0), trainable=False)
    optimizer = tf.train.AdamOptimizer(learning_rate=0.2)
    train_step = slim.learning.create_train_op(cross_entropy, optimizer, global_step=step)
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    if update_ops:
        print("BN params:", update_ops)
        updates = tf.group(*update_ops)
        train_step = control_flow_ops.with_dependencies([updates], train_step)
    tf.summary.scalar('accuracy', acc)
    tf.summary.scalar('cross_entropy', cross_entropy)
    for v in tf.all_variables():
        # print(v.name)
        if 'batch_normalization' in v.name:
            tf.summary.histogram(v.name, v)
    merged_summary_op = tf.summary.merge_all()
    return {
        'x_raw': x_raw,
        'y': y,
        'kp': kp,
        'is_training': is_training,
        'train_step': train_step,
        'global_step': step,
        'acc': acc,
        'cross_entropy': cross_entropy,
        'summary': merged_summary_op
        }

def train():
    net = model()
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    batch_size = 128
    mnist = input_data.read_data_sets('mnist_data', one_hot=True)
    for i in range(5000):
        batch_xs, batch_ys = mnist.train.next_batch(batch_size)
        train_dict = {net['x_raw']: batch_xs,
                      net['y']: batch_ys,
                      net['kp']: 0.5,
                      net['is_training']: True}
        step, _ = sess.run([net['global_step'], net['train_step']], feed_dict=train_dict)
        if step % 50 == 0:
            train_dict.update({net['kp']: 1.0, net['is_training']: True})
            acc = sess.run(net['acc'],feed_dict=train_dict)
            print(acc)


if __name__=='__main__':
    train()