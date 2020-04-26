import argparse
import os
import pickle

import matplotlib; matplotlib.use('agg')
from matplotlib.legend_handler import HandlerTuple
import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE


parser = argparse.ArgumentParser()
parser.add_argument('--savedir', type=str, required=True)
args = parser.parse_args()

path = os.path.join(args.savedir, 'plot')
if not os.path.exists(path):
    os.mkdir(path)


class ArrayZipper:
    def __init__(self, lengths):
        self.lengths = lengths

    def zip(self, elements):
        for length, element in zip(self.lengths, elements):
            assert (length == len(element))
        return np.concatenate(elements, axis=0)

    def unzip(self, zipped):
        assert len(zipped) == sum(self.lengths)
        elements = []
        idx = 0
        for length in self.lengths:
            elements.append(zipped[idx:idx+length])
            idx += length
        return elements

class CustomTSNE(TSNE):
    def __init__(self, n_components, fixed_component):
        super(CustomTSNE, self).__init__(n_components-1)
        self.fixed_component = fixed_component

    def fit_transform(self, X, y=None):
        projection = np.matmul(X, self.fixed_component[:, None])  # (N, d) @ (d, 1)
        remainder = X - np.matmul(projection, self.fixed_component[None, :])
        remainder_transformed = super(CustomTSNE, self).fit_transform(remainder)
        return np.concatenate([projection, remainder_transformed], axis=1)


def plot_metadrop(stats):
  shot = stats[0]['htr'].shape[1] // 2
  query = stats[0]['hte'].shape[1] // 2
  n_sample = stats[0]['htr_sample'].shape[1]

  for i, stat in enumerate(stats):
      print('Processing {}/{}'.format(i+1, len(stats)))
      for step in range(len(stat['htr'])):
          # Prepare points for fitting
          htr, hte = stat['htr'][step], stat['hte'][step]
          ytr, yte = stat['ytr'], stat['yte']

          htr_sample = np.array(stat['htr_sample'][step])
          htr_sample = htr_sample.reshape(n_sample*2*shot, -1)

          zipper = ArrayZipper([len(htr), len(hte), len(htr_sample), ])
          zipped_h = zipper.zip([htr, hte, htr_sample, ])
          zipped_y = zipper.zip([ytr, yte, np.repeat(ytr, n_sample), ])

          # Fit TSNE
          db_orthogonal = stat['w'][step][:, 0] - stat['w'][step][:, 1]
          db_orthogonal_unit = db_orthogonal / np.linalg.norm(db_orthogonal)
          projection = CustomTSNE(n_components=2, fixed_component=db_orthogonal_unit)
          
          htr_2d, hte_2d, htr_sample_2d = zipper.unzip(projection.fit_transform(zipped_h))
          htr_sample_2d = htr_sample_2d.reshape(n_sample, 2*shot, 2).swapaxes(0, 1)

          # Split class
          zipper = ArrayZipper([shot]*2)
          htr_2d = zipper.unzip(htr_2d)
          htr_sample_2d = zipper.unzip(htr_sample_2d)

          zipper = ArrayZipper([query]*2)
          hte_2d = zipper.unzip(hte_2d)

          # Plot
          plt.figure(dpi=300)

          # Data points
          cms = ['firebrick', 'royalblue']
          markers = ['o', '^']
          for c in range(2):
              for s in range(shot):
                  # training data
                  plt.scatter(
                      htr_2d[c][s, 0], htr_2d[c][s, 1],
                      marker=markers[s], s=300, c=cms[c], edgecolors='black',
                      linewidths=1, zorder=7
                  )
                  # training with noise
                  plt.scatter(
                      htr_sample_2d[c][s, :, 0], htr_sample_2d[c][s, :, 1],
                      marker=markers[s], s=300, c=cms[c], edgecolors='black',
                      linewidths=1, linestyle=':', alpha=0.5, zorder=0
                  )

              # test data
              plt.scatter(
                  hte_2d[c][:, 0], hte_2d[c][:, 1],
                  marker='*', s=300, c=cms[c], edgecolors='black',
                  linewidths=1, zorder=3
              )

          db = (stat['b'][step][1] - stat['b'][step][0]) / np.linalg.norm(db_orthogonal)
          plt.axvline(db, c='black')

          # Custom legend
          train_markers = []
          sample_markers = []
          for s in range(shot):
            train_marker = plt.scatter([], [], c='none', marker=markers[s], edgecolor='black', linestyle='-', s=100)
            train_marker.remove()
            train_markers.append(train_marker)
            sample_marker = plt.scatter([], [], c='none', marker=markers[s], edgecolor='black', linestyle=':', s=100)
            sample_marker.remove()
            sample_markers.append(sample_marker)

          train_markers = tuple(train_markers)
          sample_markers = tuple(sample_markers)
          test_marker = matplotlib.lines.Line2D([], [], color='none', markeredgecolor='black', marker='*', linestyle='None', markersize=15)
          db_legend = matplotlib.lines.Line2D([], [], color='black', linestyle='-', label='decision boundary')

          plt.legend(handles=[train_markers, sample_markers, test_marker, db_legend],
                     labels=['train', 'perturbed train', 'test', 'decision boundary'],
                     prop={'size': 13}, loc="lower left",
                     handler_map={train_markers: HandlerTuple(ndivide=2, pad=0.),
                                  sample_markers: HandlerTuple(ndivide=2, pad=0.),}).set_zorder(12)

          plt.axis('off')
          plt.tight_layout()
              
          path = os.path.join(args.savedir, 'plot/step{}'.format(step))
          if not os.path.exists(path):
              os.mkdir(path)
          filename = '{}.png'.format(i)
          plt.tight_layout()
          plt.savefig(os.path.join(path, filename), bbox_inches='tight')
          plt.close()


def plot_maml(stats):
  shot = stats[0]['htr'].shape[1] // 2
  query = stats[0]['hte'].shape[1] // 2

  for i, stat in enumerate(stats):
      print('Processing {}/{}'.format(i+1, len(stats)))
      for step in range(len(stat['htr'])):
          # Prepare points for fitting
          htr, hte = stat['htr'][step], stat['hte'][step]
          ytr, yte = stat['ytr'], stat['yte']

          zipper = ArrayZipper([len(htr), len(hte), ])
          zipped_h = zipper.zip([htr, hte, ])
          zipped_y = zipper.zip([ytr, yte, ])

          # Fit TSNE
          db_orthogonal = stat['w'][step][:, 0] - stat['w'][step][:, 1]
          db_orthogonal_unit = db_orthogonal / np.linalg.norm(db_orthogonal)
          projection = CustomTSNE(n_components=2, fixed_component=db_orthogonal_unit)
          
          htr_2d, hte_2d = zipper.unzip(projection.fit_transform(zipped_h))

          # Split class
          zipper = ArrayZipper([shot]*2)
          htr_2d = zipper.unzip(htr_2d)

          zipper = ArrayZipper([query]*2)
          hte_2d = zipper.unzip(hte_2d)

          # Plot
          plt.figure(dpi=300)

          # Data points
          cms = ['firebrick', 'royalblue']
          markers = ['o', '^']
          for c in range(2):
              for s in range(shot):
                  # training data
                  plt.scatter(
                      htr_2d[c][s, 0], htr_2d[c][s, 1],
                      marker=markers[s], s=300, c=cms[c], edgecolors='black',
                      linewidths=1, zorder=7
                  )

              # test data
              plt.scatter(
                  hte_2d[c][:, 0], hte_2d[c][:, 1],
                  marker='*', s=300, c=cms[c], edgecolors='black',
                  linewidths=1, zorder=3
              )

          db = (stat['b'][step][1] - stat['b'][step][0]) / np.linalg.norm(db_orthogonal)
          plt.axvline(db, c='black')

          # Custom legend
          train_markers = []
          for s in range(shot):
            train_marker = plt.scatter([], [], c='none', marker=markers[s], edgecolor='black', linestyle='-', s=100)
            train_marker.remove()
            train_markers.append(train_marker)

          train_markers = tuple(train_markers)
          test_marker = matplotlib.lines.Line2D([], [], color='none', markeredgecolor='black', marker='*', linestyle='None', markersize=15)
          db_legend = matplotlib.lines.Line2D([], [], color='black', linestyle='-', label='decision boundary')

          plt.legend(handles=[train_markers, test_marker, db_legend],
                     labels=['train', 'test', 'decision boundary'],
                     prop={'size': 13}, loc="lower left",
                     handler_map={train_markers: HandlerTuple(ndivide=2, pad=0.),}).set_zorder(12)

          plt.axis('off')
          plt.tight_layout()
              
          path = os.path.join(args.savedir, 'plot/step{}'.format(step))
          if not os.path.exists(path):
              os.mkdir(path)
          filename = '{}.png'.format(i)
          plt.tight_layout()
          plt.savefig(os.path.join(path, filename), bbox_inches='tight')
          plt.close()


if __name__ == '__main__':
  with open(os.path.join(args.savedir, 'export.pkl'), 'rb') as f:
    stats = pickle.load(f)
  if 'htr_sample' in stats[0]:
    plot_metadrop(stats)
  else:
    plot_maml(stats)

