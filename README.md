# PyTorch Implementation for Neural Graph Collaborative Filtering
This is a PyTorch Implemenation for this paper: 

Xiang Wang, Xiangnan He, Meng Wang, Fuli Feng, and Tat-Seng Chua (2019). Neural Graph Collaborative Filtering. SIGIR 2019


Original TensorFlow Implementation can be found [here](https://github.com/xiangwang1223/neural_graph_collaborative_filtering)

## Environment Requirement
You may simply run 
``
pip install -r requirements.txt
``

In this implementation, we use Python 3.7.5 with CUDA 10.1. 
The required packages are as follows:
* pytorch==1.3.1      
* numpy==1.16.4
* scipy==1.4.1
* scikit-learn==0.22


## Run the Code
### NGCF 
* Gowalla: 
```
python main.py --dataset gowalla --alg_type ngcf --regs [1e-5] --embed_size 64 --layer_size [64,64,64] --lr 0.0001 --save_flag 1 --pretrain 0 --batch_size 4096 --epoch 500 --verbose 1 --mess_dropout [0.1,0.1,0.1] 
```
* Amazon-book: 
```
python main.py --dataset amazon-book --alg_type ngcf --regs [1e-5] --embed_size 64 --layer_size [64,64,64] --lr 0.0005 --save_flag 1 --pretrain 0 --batch_size 4096 --epoch 500 --verbose 1 --mess_dropout [0.1,0.1,0.1]
```
### MF 
* Gowalla: 
```
python main.py --dataset gowalla --alg_type mf --regs [1e-5] --embed_size 64  --lr 0.0001 --save_flag 1 --pretrain 0 --batch_size 4096 --epoch 500 --verbose 1 
```
* Amazon-book: 
```
python main.py --dataset amazon-book --alg_type mf --regs [1e-5] --embed_size 64  --lr 0.0005 --save_flag 1 --pretrain 0 --batch_size 4096 --epoch 500 --verbose 1 --mess_dropout [0.1,0.1,0.1]
```


## Dataset
Datasets and Data files are the same as thoese in [the original repository](https://github.com/xiangwang1223/neural_graph_collaborative_filtering). 