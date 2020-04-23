# Meta Dropout: Learning to Perturb Latent Features for Generalization
This is the **TensorFlow implementation** for the paper Meta Dropout: Learning to Perturb Latent Features for Generalization (**ICLR 2020**) : https://openreview.net/forum?id=BJgd81SYwr.

You can reproduce the results of Table 1 in the main paper.

## Abstract
<img align="middle" width="700" src="https://github.com/haebeom-lee/metadrop/blob/master/images/concept.png">

A machine learning model that generalizes well should obtain low errors on unseen test examples. Thus, if we know how to optimally perturb training examples to account for test examples, we may achieve better generalization performance. However, obtaining such perturbation is not possible in standard machine learning frameworks as the distribution of the test data is unknown. To tackle this challenge, we propose a novel regularization method, meta-dropout, which learns to perturb the latent features of training examples for generalization in a meta-learning framework. Specifically, we meta-learn a noise generator which outputs a multiplicative noise distribution for latent features, to obtain low errors on the test instances in an input-dependent manner. Then, the learned noise generator can perturb the training examples of unseen tasks at the meta-test time for improved generalization. We validate our method on few-shot classification datasets, whose results show that it significantly improves the generalization performance of the base model, and largely outperforms existing regularization methods such as information bottleneck, manifold mixup, and information dropout.

## Prerequisites
- Python 3.5 (Anaconda)
- Tensorflow 1.12.0
- CUDA 9.0
- cudnn 7.6.5

If you are not familiar with preparing conda environment, please follow the below instructions:
```
$ conda create --name py35 python=3.5
$ conda activate py35
$ pip install --upgrade pip
$ pip install tensorflow-gpu==1.12.0
$ conda install -c anaconda cudatoolkit=9.0
$ conda install -c anaconda cudnn
```

And for data preprocessing,
```
$ pip install tqdm
$ pip install requests
$ pip install Pillow
$ pip install scipy
```

### Data Preparation
Go to the folder of each dataset (i.e. ```data/omniglot```, ```data/mimgnet```) and run ```python get_data.py``` there. For example, to download miniImageNet dataset and preprocess it,
```
$ cd ./data/mimgnet
$ python get_data.py
```
It will take some time to download and preprocess each dataset.

## Citation
If you found the provided code useful, please cite our work.
```
@inproceedings{
    lee2020l2b,
    title={Learning to Balance: Bayesian Meta-Learning for Imbalanced and Out-of-distribution Tasks},
    author={Hae Beom Lee and Hayeon Lee and Donghyun Na and Saehoon Kim and Minseop Park and Eunho Yang and Sung Ju Hwang},
    booktitle={ICLR},
    year={2020}
}
```
