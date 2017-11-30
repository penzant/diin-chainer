# diin-chainer
A chainer implementation of [Densely Interactive Inference Network (arXiv:1709.04348)](https://arxiv.org/abs/1709.04348)

Requirements:
* python 3.6+ (verified)
* Chainer 3.1.0+ (verified)
* Preprocessed MultiNLI/SNLI data from [the original DIIN (github)](https://github.com/YichenGong/Densely-Interactive-Inference-Network)

To Run,
```
python train.py
```

For debug with a small data:
```
python train.py --debug_mode
```

TODO:
* Learning rate in adadelta
* Use SGD in the middle of training
* Early stopping
* Confusion matrix
* Use ParallelUpdater? (for data parallel processing)
* Improve performance...
