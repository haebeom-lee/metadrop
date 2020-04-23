"""
2020/04/24
Script for downloading Omniglot or miniImageNet dataset
Run this file as follows:
    python get_data.py --dataset DATASET_NAME
"""

import os
from tqdm import tqdm
import requests
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str,
    choices=['omniglot', 'mimgnet'], default='omniglot')
args = parser.parse_args()

if not os.path.isdir('data'):
  os.makedirs('data')

def download_file(url, filename):
    """
    Helper method handling downloading large files from `url`
    to `filename`. Returns a pointer to `filename`.
    """
    chunkSize = 1024
    r = requests.get(url, stream=True)
    with open(filename, 'wb') as f:
        pbar = tqdm( unit="B", total=int( r.headers['Content-Length'] ) )
        for chunk in r.iter_content(chunk_size=chunkSize):
            if chunk: # filter out keep-alive new chunks
                pbar.update (len(chunk))
                f.write(chunk)
    return filename

# download Omniglot dataset
if args.dataset == 'omniglot':

  path = os.path.join('data', 'omniglot')
  if not os.path.isdir(path):
    os.makedirs(path)

  print("Downloading train.npy of Omniglot\n")
  download_file('https://www.dropbox.com/s/h13g4b2awd7xdr6/train.npy?dl=1',
      os.path.join(path, 'train.npy'))
  print("Downloading test.npy of Omniglot\n")
  download_file('https://www.dropbox.com/s/w313ybz6rls1e83/test.npy?dl=1',
      os.path.join(path, 'test.npy'))
  print("Downloading done.\n")

# download miniImageNet dataset
elif args.dataset == 'mimgnet':

  path = os.path.join('data', 'mimgnet')
  if not os.path.isdir(path):
    os.makedirs(path)

  print("Downloading train.npy of miniImageNet\n")
  download_file('https://www.dropbox.com/s/ir54llgjfjv3naa/train.npy?dl=1',
      os.path.join(path, 'train.npy'))
  print("Downloading done.\n")

  print("Downloading test.npy of miniImageNet\n")
  download_file('https://www.dropbox.com/s/i40h3wyjlpxeesr/test.npy?dl=1',
      os.path.join('test.npy'))
  print("Downloading done.\n")
