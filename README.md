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
#### Bag of Words

The model is jointly trained on 20 tasks with following hyperparameters.
- epochs: 200
- feature_size: 50
- embedding_size: 40
- hops: 3
- learning_rate (with decay): 0.005
```
python joint.py --epochs 200 --feature_size 50 --learning_rate 0.005
```
| Task | Testing Accuracy | Training Accuracy | Validation Accuracy |
|------|------------------|-------------------|---------------------|
| 1    | 1.00             | 1.00              | 1.00                |
| 2    | 0.66             | 0.99              | 0.82                |
| 3    | 0.42             | 0.97              | 0.60                |
| 4    | 0.99             | 1.00              | 0.99                |
| 5    | 0.49             | 0.53              | 0.57                |
| 6    | 0.93             | 0.99              | 0.96                |
| 7    | 0.49             | 0.49              | 0.46                |
| 8    | 0.88             | 0.92              | 0.91                |
| 9    | 0.95             | 0.99              | 0.98                |
| 10   | 0.89             | 0.98              | 0.87                |
| 11   | 1.00             | 1.00              | 1.00                |
| 12   | 1.00             | 1.00              | 1.00                |
| 13   | 1.00             | 1.00              | 1.00                |
| 14   | 0.95             | 1.00              | 0.98                |
| 15   | 0.21             | 0.28              | 0.36                |
| 16   | 0.24             | 0.28              | 0.31                |
| 17   | 0.53             | 0.94              | 0.78                |
| 18   | 0.91             | 0.99              | 0.94                |
| 19   | 0.08             | 0.10              | 0.06                |
| 20   | 0.84             | 0.85              | 0.82                |

I'm still trying to tune hyperparameters or model to improve results.
#### RNN(GRU)

### Requirements

* tensorflow 0.9
* scikit-learn 0.17.1
* six 1.10.0
