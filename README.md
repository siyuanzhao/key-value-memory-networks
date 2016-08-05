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
The model is jointly trained on 20 tasks with following hyperparameters.
- epochs: 200
- feature_size: 50
- embedding_size: 40
- hops: 3
- learning_rate (with decay): 0.005
- l2 lambda: 0.2
```
python joint.py --epochs 200 --feature_size 50 --learning_rate 0.005
```
| Task | Testing Accuracy | Training Accuracy | Validation Accuracy |
|------|------------------|-------------------|---------------------|
| 1    | 1.00             | 1.00              | 1.00                |
| 2    | 0.61             | 0.99              | 0.78                |
| 3    | 0.36             | 0.95              | 0.56                |
| 4    | 1.00             | 1.00              | 1.00                |
| 5    | 0.49             | 0.53              | 0.48                |
| 6    | 0.97             | 0.99              | 0.98                |
| 7    | 0.49             | 0.49              | 0.48                |
| 8    | 0.85             | 0.92              | 0.90                |
| 9    | 0.96             | 1.00              | 0.96                |
| 10   | 0.90             | 0.97              | 0.87                |
| 11   | 0.99             | 1.00              | 0.99                |
| 12   | 1.00             | 1.00              | 1.00                |
| 13   | 1.00             | 1.00              | 0.99                |
| 14   | 0.97             | 1.00              | 0.98                |
| 15   | 0.21             | 0.29              | 0.26                |
| 16   | 0.24             | 0.28              | 0.30                |
| 17   | 0.56             | 0.94              | 0.77                |
| 18   | 0.92             | 0.98              | 0.95                |
| 19   | 0.08             | 0.09              | 0.07                |
| 20   | 1.00             | 1.00              | 1.00                |
##### Single Model
The model is trained on a single task at a time with following hyperparameters.
- epochs: 200
- feature_size: 40
- embedding_size: 30
- hops: 3
- learning_rate (with decay): 0.005
- l2 lambda: 0.9

| Task | Testing Accuracy | Training Accuracy | Validation Accuracy |
|------|------------------|-------------------|---------------------|
| 1    | 1.00             | 1.00              | 1.00                |
| 2    | 0.35             | 0.97              | 0.66                |
| 3    | 0.27             | 0.94              | 0.45                |
| 4    | 1.00             | 1.00              | 1.00                |
| 5    | 0.79             | 0.99              | 0.83                |
| 6    | 0.85             | 0.97              | 0.93                |
| 7    | 0.78             | 1.00              | 0.77                |
| 8    | 0.89             | 0.95              | 0.85                |
| 9    | 0.81             | 1.00              | 0.83                |
| 10   | 0.81             | 1.00              | 0.70                |
| 11   | 0.86             | 1.00              | 0.78                |
| 12   | 1.00             | 1.00              | 0.97                |
| 13   | 0.90             | 0.99              | 0.92                |
| 14   | 0.91             | 1.00              | 0.88                |
| 15   | 1.00             | 1.00              | 1.00                |
| 16   | 0.43             | 0.93              | 0.41                |
| 17   | 0.53             | 0.87              | 0.65                |
| 18   | 0.90             | 0.99              | 0.92                |
| 19   | 0.12             | 0.60              | 0.12                |
| 20   | 1.00             | 1.00              | 1.00                |

On single model, overfitting is a problem.

I'm still trying to tune hyperparameters or model to improve results.


#### RNN(GRU)

### Requirements

* tensorflow 0.9
* scikit-learn 0.17.1
* six 1.10.0
