# Key Value Memory Networks

This repo contains the implementation of [Key Value Memory Networks for Directly Reading Documents](https://arxiv.org/abs/1606.03126) in Tensorflow. The model is tested on [bAbI](http://arxiv.org/abs/1502.05698).

![Structure of Key Value Memory Networks](key_value_mem.png)

There is a [must-read tutorial](http://www.thespermwhale.com/jaseweston/icml2016/) on Memory Networks for NLP from Jason Weston @ ICML 2016.

### Get Started

```
git clone https://github.com/siyuanzhao/deep-networks.git

mkdir ./deep-networks/key_value_memory/data/
cd ./key_value_memory/data/
wget http://www.thespermwhale.com/jaseweston/babi/tasks_1-20_v1-2.tar.gz
tar xzvf ./tasks_1-20_v1-2.tar.gz

cd ../
python single.py
```

### Examples

Running a [single bAbI task](./key_value_memory/single.py)

Running a [joint model on all bAbI tasks](./key_value_memory/joint.py)

### Results

The model is jointly trained on 20 tasks with following hyperparameters.
- epochs: 200
- feature_size: 50
- embedding_size: 40
- hops: 3
- learning_rate (with decay): 0.005


| Task | Testing Accuracy | Training Accuracy | Validation Accuracy |
|------|------------------|-------------------|---------------------|
| 1    | 0.99             | 1.0               | 1.0                 |
| 2    | 0.61             | 0.96              | 0.74                |
| 3    | 0.30             | 0.94              | 0.47                |
| 4    | 0.99             | 1.0               | 0.99                |
| 5    | 0.49             | 0.54              | 0.47                |
| 6    | 0.57             | 0.58              | 0.45                |
| 7    | 0.49             | 0.49              | 0.49                |
| 8    | 0.80             | 0.89              | 0.79                |
| 9    | 0.66             | 0.67              | 0.73                |
| 10   | 0.47             | 0.54              | 0.51                |
| 11   | 0.98             | 1.0               | 0.97                |
| 12   | 1.0              | 1.0               | 0.99                |
| 13   | 0.98             | 1.0               | 0.96                |
| 14   | 0.92             | 1.0               | 0.95                |
| 15   | 0.21             | 0.29              | 0.31                |
| 16   | 0.24             | 0.28              | 0.29                |
| 17   | 0.54             | 0.91              | 0.69                |
| 18   | 0.91             | 0.96              | 0.96                |
| 19   | 0.08             | 0.09              | 0.09                |
| 20   | 0.84             | 0.85              | 0.84                |

I'm still trying to tune hyperparameters or model to improve results.

### Requirements

* tensorflow 0.9
* scikit-learn 0.17.1
* six 1.10.0
