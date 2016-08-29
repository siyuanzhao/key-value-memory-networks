# Key Value Memory Networks

This repo contains the implementation of [Key Value Memory Networks for Directly Reading Documents](https://arxiv.org/abs/1606.03126) in Tensorflow. The model is tested on [bAbI](http://arxiv.org/abs/1502.05698).

![Structure of Key Value Memory Networks](key_value_mem.png)

### Tutorial
- There is a [must-read tutorial](http://www.thespermwhale.com/jaseweston/icml2016/) on Memory Networks for NLP from Jason Weston @ ICML 2016.

- [[Video](http://videolectures.net/deeplearning2016_chopra_attention_memory/)] [[Slides](https://drive.google.com/file/d/0B_hO8cnpcIMgYnlsMFlGSkxRLUk/view?usp=sharing)]
Sumit Chopra, from Facebook AI, gave a lecture about Reasoning, Attention and Memory at Deep Learning Summer School 2016.

### Get Started

```
git clone https://github.com/siyuanzhao/key-value-memory-networks.git

mkdir ./key-value-memory-networks/key_value_memory/logs
mkdir ./key-value-memory-networks/key_value_memory/data/
cd ./key-value-memory-networks/key_value_memory/data
wget http://www.thespermwhale.com/jaseweston/babi/tasks_1-20_v1-2.tar.gz
tar xzvf ./tasks_1-20_v1-2.tar.gz

cd ../
python single.py
```

### Usage

#### Running a [single bAbI task](./key_value_memory/single.py)

```
# Train the model on a single task <task_id>
python single.py --task_id <task_id>
```
There are serval flags within single.py. Below is an example of training the model on task 20 with specific learning rate, feature_size and epochs.
```
python single.py --task_id 20 --learning_rate 0.005 --feature_size 40 --epochs 200
```
Check all avaiable flags with the following command.
```
python single.py -h
```
#### Running a [joint model on all bAbI tasks](./key_value_memory/joint.py)
```
python joint.py
```
There are also serval flags within joint.py. Below is an example of training the joint model with specific learning rate, feature_size and epochs.
```
python joint.py --learning_rate 0.005 --feature_size 40 --epochs 200
```
Check all avaiable flags with the following command.
```
python joint.py -h
```
### Results
#### Bag of Words
##### Joint Model
The model is jointly trained on 20 tasks (1k training examples / weakly supervised) with following hyperparameters.
- BATCH_SIZE=50
- EMBEDDING_SIZE=40
- EPOCHS=200
- EPSILON=0.1
- FEATURE_SIZE=50
- HOPS=3
- L2_LAMBDA=0.1
- LEARNING_RATE=0.001
- MAX_GRAD_NORM=20.0
- MEMORY_SIZE=50
- READER=bow

```
python joint.py
```
| Task | Testing Accuracy | Training Accuracy | Validation Accuracy |
|------|------------------|-------------------|---------------------|
| 1    | 1.00             | 1.00              | 1.00                |
| 2    | 0.80             | 0.87              | 0.85                |
| 3    | 0.66             | 0.77              | 0.69                |
| 4    | 0.73             | 0.79              | 0.74                |
| 5    | 0.84             | 0.91              | 0.80                |
| 6    | 0.98             | 0.99              | 0.98                |
| 7    | 0.83             | 0.85              | 0.80                |
| 8    | 0.89             | 0.92              | 0.86                |
| 9    | 0.98             | 0.99              | 0.96                |
| 10   | 0.85             | 0.96              | 0.89                |
| 11   | 0.97             | 0.98              | 0.99                |
| 12   | 0.99             | 0.99              | 1.00                |
| 13   | 0.99             | 0.99              | 1.00                |
| 14   | 0.80             | 0.90              | 0.84                |
| 15   | 0.56             | 0.57              | 0.45                |
| 16   | 0.46             | 0.48              | 0.37                |
| 17   | 0.57             | 0.72              | 0.70                |
| 18   | 0.93             | 0.95              | 0.92                |
| 19   | 0.10             | 0.11              | 0.06                |
| 20   | 0.98             | 0.99              | 0.99                |

- results on 10k training examples are [here](kv_joint_10k_results.csv)

#### RNN(GRU)

### Requirements

* tensorflow 0.9
* scikit-learn 0.17.1
* six 1.10.0
