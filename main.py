from __future__ import print_function
import tensorflow as tf
import argparse
import numpy as np
import time
import os

from model import MetaDropout
from data import Data
from accumulator import Accumulator
from layers import get_train_op

parser = argparse.ArgumentParser()

parser.add_argument('--gpu_id', type=int, default=0,
    help='GPU id')

parser.add_argument('--mode', type=str, default='meta_train',
    help='either meta_train or meta_test')

parser.add_argument('--savedir', type=str, default=None,
    help='save directory')

parser.add_argument('--save_freq', type=int, default=1000,
    help='save frequency')

parser.add_argument('--n_train_iters', type=int, default=60000,
    help='number of meta-training iterations')

parser.add_argument('--n_test_iters', type=int, default=1000,
    help='number of meta-testing iterations')

parser.add_argument('--dataset', type=str, default='omniglot',
    help='either omniglot or mimgnet')

parser.add_argument('--way', type=int, default=20,
    help='number of classes per task')

parser.add_argument('--shot', type=int, default=1,
    help='number of training examples per class')

parser.add_argument('--query', type=int, default=5,
    help='number of test examples per class')

parser.add_argument('--metabatch', type=int, default=16,
    help='number of tasks per each meta-iteration')

parser.add_argument('--meta_lr', type=float, default=1e-3,
    help='meta learning rate')

parser.add_argument('--inner_lr', type=float, default=0.1,
    help='inner-gradient stepsize')

parser.add_argument('--n_steps', type=int, default=5,
    help='number of inner-gradient steps')

parser.add_argument('--n_test_mc_samp', type=int, default=1,
    help='number of MC samples to evaluate the expected inner-step loss')

parser.add_argument('--maml', action='store_true', default=False,
    help='whether to convert this model back to the base MAML or not')

args = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu_id)

if not os.path.isdir(args.savedir):
  os.makedirs(args.savedir)

# for generating episode
data = Data(args)

# model object
model = MetaDropout(args)
epi = model.episodes
placeholders = [epi['xtr'], epi['ytr'], epi['xte'], epi['yte']]

# meta-training pipeline
net = model.get_loss_multiple(True)
net_cent = net['cent']
net_acc = net['acc']
net_acc_mean = tf.reduce_mean(net['acc'])
net_weights = net['weights']

# meta-testing pipeline
tnet = model.get_loss_multiple(False)
tnet_cent = tnet['cent']
tnet_acc = tnet['acc']
tnet_acc_mean = tf.reduce_mean(tnet['acc'])
tnet_weights = tnet['weights']

# meta-training
def meta_train():
  global_step = tf.train.get_or_create_global_step()

  #if (not args.maml) and args.dataset == 'omniglot' and args.shot == 1:
  #  lr = tf.train.piecewise_constant(tf.cast(global_step, tf.int32),
  #      [2000], [args.meta_lr, 0.1*args.meta_lr])
  #else:
  lr = tf.convert_to_tensor(args.meta_lr)

  optim = tf.train.AdamOptimizer(lr)

  if args.maml:
    var_list = [v for v in net_weights if 'phi' not in v.name]
  else:
    var_list = net_weights

  meta_train_op = get_train_op(optim, net_cent, clip=[-3., 3.],
      global_step=global_step, var_list=var_list)

  saver = tf.train.Saver(tf.trainable_variables())
  logfile = open(os.path.join(args.savedir, 'meta_train.log'), 'w')

  argdict = vars(args)
  print(argdict)
  for k, v in argdict.items():
      logfile.write(k + ': ' + str(v) + '\n')
  logfile.write('\n')

  config = tf.ConfigProto()
  config.gpu_options.allow_growth = True
  sess = tf.Session(config=config)
  sess.run(tf.global_variables_initializer())

  meta_train_logger = Accumulator('cent', 'acc')
  meta_train_to_run = [meta_train_op, net_cent, net_acc_mean]

  meta_test_logger = Accumulator('cent', 'acc')
  meta_test_to_run = [tnet_cent, tnet_acc_mean]

  start = time.time()
  for i in range(args.n_train_iters+1):
    episode = data.generate_episode(args, meta_training=True,
        n_episodes=args.metabatch)
    fd_mtr = dict(zip(placeholders, episode))
    meta_train_logger.accum(sess.run(meta_train_to_run, feed_dict=fd_mtr))

    if i % 50 == 0:
      line = 'Iter %d start, learning rate %f' % (i, sess.run(lr))
      print('\n' + line)
      logfile.write('\n' + line + '\n')
      meta_train_logger.print_(header='meta_train', episode=i*args.metabatch,
          time=time.time()-start, logfile=logfile)
      meta_train_logger.clear()

    if i % 1000 == 0:
      for j in range(50):
        episode = data.generate_episode(args, meta_training=False,
            n_episodes=args.metabatch)
        fd_mte= dict(zip(placeholders, episode))
        meta_test_logger.accum(sess.run(meta_test_to_run, feed_dict=fd_mte))

      meta_test_logger.print_(header='meta_test ', episode=i*args.metabatch,
          time=time.time()-start, logfile=logfile)
      meta_test_logger.clear()

    if i % args.save_freq == 0:
      saver.save(sess, os.path.join(args.savedir, 'model'))

  logfile.close()

# meta-testing
def meta_test():
  config = tf.ConfigProto()
  config.gpu_options.allow_growth = True
  sess = tf.Session(config=config)
  saver = tf.train.Saver(tnet_weights)
  saver.restore(sess, os.path.join(args.savedir, 'model'))

  f = open(os.path.join(args.savedir, 'meta_test.log'), 'w')

  start = time.time()
  acc = []
  for j in range(args.n_test_iters//args.metabatch):
    if j % 10 == 0:
      print('(%.3f secs) meta test iter %d start'\
          % (time.time()-start,j*args.metabatch))
    epi = model.episodes
    episode = data.generate_episode(args, meta_training=False,
        n_episodes=args.metabatch)
    fd_mte= dict(zip(placeholders, episode))
    acc.append(sess.run(tnet_acc, feed_dict=fd_mte))

  acc = 100.*np.concatenate(acc, axis=0)

  acc_mean = np.mean(acc)
  acc_95conf = 1.96*np.std(acc)/float(np.sqrt(args.n_test_iters))

  result = 'accuracy : %f +- %f'%(acc_mean, acc_95conf)
  print(result)
  f.write(result)
  f.close()

if __name__=='__main__':
  if args.mode == 'meta_train':
    meta_train()
  elif args.mode == 'meta_test':
    meta_test()
  else:
    raise ValueError('Invalid mode %s' % args.mode)
