# PyTorch Implementation for Neural Graph Collaborative Filtering (SNGCF)
## Dataset
* Gowalla: n_users=29858, n_items=40981, n_nodes=70839, n_interactions=1027370, sparsity=0.084%
* Amazon-book: n_users=52643, n_items=91599, n_nodes=144242, n_interactions=2984108, sparsity=0.062%
* Yelp2018: n_users=31668, n_items=38048, n_nodes=69716, n_interaction=1561406, sparsity=0.130%

## Usage 
* Gowalla: python Torch_SNGCF.py --dataset gowalla --alg_type ngcf --regs [1e-5] --embed_size 64 --sngcf_layers 8 --lr 0.0004 --save_flag 1 --pretrain 0 --batch_size 4096 --epoch 2000 --verbose 1 --mess_dropout [0.1] 
* Amazon-book: python Torch_SNGCF.py --dataset amazon-book --alg_type ngcf --regs [1e-5] --embed_size 64 --sngcf_layers 12 --lr 0.001 --save_flag 1 --pretrain 0 --batch_size 4096 --epoch 2000 --verbose 1 --mess_dropout [0.1]
* Yelp-2018: python Torch_SNGCF.py --dataset yelp2018 --alg_type ngcf --regs [1e-5] --embed_size 64 --sngcf_layers 10 --lr 0.0005 --save_flag 1 --pretrain 0 --batch_size 4096 --epoch 2000 --verbose 1 --mess_dropout [0.1]
