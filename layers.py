import tensorflow as tf
import numpy as np

# functions
log = lambda x: tf.log(x + 1e-20)
softmax = tf.nn.softmax
relu = tf.nn.relu
softplus = tf.nn.softplus
sqrt = tf.sqrt
exp = tf.exp

# layers
flatten = tf.layers.flatten
batch_norm = tf.contrib.layers.batch_norm

# distribution
normal = tf.distributions.Normal

# blocks
def conv_block(x, wt, bt, wp, bp, sample=False, bn_scope='conv_bn', maml=False):
  mu = tf.nn.conv2d(x, wt, [1,1,1,1], 'SAME') + bt # NHWC
  alpha = tf.nn.conv2d(x, wp, [1,1,1,1], 'SAME') + bp # NHWC

  ones = tf.ones_like(alpha)
  mult_noise = normal(alpha, ones).sample() if sample else alpha

  if maml:
    x = mu
  else:
    x = mu * softplus(mult_noise)

  x = batch_norm(x, activation_fn=relu, scope=bn_scope, reuse=tf.AUTO_REUSE)
  x = tf.nn.max_pool(x, [1,2,2,1], [1,2,2,1], 'VALID')
  return x

def dense_block(x, wt, bt, wp, bp, sample=False, maml=False):
  mu = tf.matmul(flatten(x), wt) + bt
  sigma = softplus(tf.matmul(flatten(x), wp) + bp)

  if maml or not sample:
    x = mu
  else:
    x = normal(mu, sigma).sample()
  return x

# training modules
def cross_entropy(logits, labels):
  return tf.losses.softmax_cross_entropy(logits=logits,
      onehot_labels=labels)

def accuracy(logits, labels):
  correct = tf.equal(tf.argmax(logits, 1), tf.argmax(labels, 1))
  return tf.reduce_mean(tf.cast(correct, tf.float32))

# for gradient clipping
def get_train_op(optim, loss, global_step=None, clip=None, var_list=None):
  grad_and_vars = optim.compute_gradients(loss, var_list=var_list)
  if clip is not None:
      grad_and_vars = [((None if grad is None \
              else tf.clip_by_value(grad, clip[0], clip[1])), var) \
              for grad, var in grad_and_vars]
  train_op = optim.apply_gradients(grad_and_vars, global_step=global_step)
  return train_op
