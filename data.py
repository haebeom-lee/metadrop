import numpy as np

class Data:
  def __init__(self, args):
    if args.dataset == 'omniglot':
      self.N = 20 # total num instances per class
      self.K_mtr = 4800 # total num meta_train classes
      self.K_mte = 1692 # total num meta_test classes

      x_mtr = np.load('./data/omniglot/train.npy')
      x_mte = np.load('./data/omniglot/test.npy')
      self.x_mtr = np.reshape(x_mtr, [4800,20,28*28*1])
      self.x_mte = np.reshape(x_mte, [1692,20,28*28*1])

    elif args.dataset == 'mimgnet':
      self.N = 600 # total num instances per class
      self.K_mtr = 64 # total num meta_train classes
      self.K_mte = 20 # total num meta_test classes

      x_mtr = np.load('./data/mimgnet/train.npy')
      x_mte = np.load('./data/mimgnet/test.npy')
      self.x_mtr = np.reshape(x_mtr, [64,600,84*84*3])
      self.x_mte = np.reshape(x_mte, [20,600,84*84*3])
    else:
      raise ValueError('No such dataset %s' % args.dataset)

  def generate_episode(self, args, meta_training=True, n_episodes=1):
    generate_label = lambda way, n_samp: np.repeat(np.eye(way), n_samp, axis=0)
    n_way, n_shot, n_query = args.way, args.shot, args.query
    (K,x) = (self.K_mtr, self.x_mtr) if meta_training else (self.K_mte, self.x_mte)

    xtr, ytr, xte, yte = [], [], [], []
    for t in range(n_episodes):
      # sample WAY classes
      classes = np.random.choice(range(K), size=n_way, replace=False)

      xtr_t = []
      xte_t = []
      for k in list(classes):
        # sample SHOT and QUERY instances
        idx = np.random.choice(range(self.N), size=n_shot+n_query, replace=False)
        x_k = x[k][idx]
        xtr_t.append(x_k[:n_shot])
        xte_t.append(x_k[n_shot:])

      xtr.append(np.concatenate(xtr_t, 0))
      xte.append(np.concatenate(xte_t, 0))
      ytr.append(generate_label(n_way, n_shot))
      yte.append(generate_label(n_way, n_query))

    xtr, ytr = np.stack(xtr, 0), np.stack(ytr, 0)
    xte, yte = np.stack(xte, 0), np.stack(yte, 0)
    return [xtr, ytr, xte, yte]
