# Pytorch-online-metric-learning
Pytorch implementation of online metric learning losses.

To make full use of the information in a batch, the losses consider all pairs in a batch by using pure tensor operations.

Requirement: Pytorch 0.4.0 or higher(tested)

## Implementation

### Online Contrastive Loss

![](http://latex.codecogs.com/gif.latex?Loss=\\frac{1}{N}\\sum%20_{i,j}%20y_{ij}%20D_{ij}^{2}+(1-y_{ij})\\big[\\alpha-D_{ij}^{2}\\big]_{+})

###  Triplet Loss with Hard Negative Mining Strategy

![](http://latex.codecogs.com/gif.latex?Loss=\\frac{1}{N}\\sum\\big[D_{ap}^{2}+\\alpha-D_{an}^{2}\\big]_{+})

* Hardest Mining Strategy

    ![](http://latex.codecogs.com/gif.latex?D_{an}%20:=%20argmin(D_{an}^2))

* Semi-hard Mining Strategy

    ![](http://latex.codecogs.com/gif.latex?D_{an}%20:=%20argmin(D_{an}^2)\\quad%20s.t.\\%20D_{an}^2>D_{ap}^2)

    ref. [*FaceNet: A Unified Embedding for Face Recognition and Clustering*](https://arxiv.org/abs/1503.03832)
    
* No Mining Strategy (Use all valid triplets in a batch)

### Lifted Structured Feature Embedding

![](https://ws1.sinaimg.cn/large/006tNbRwly1fxvtlzv4nkj30cy023t8m.jpg)

![](http://latex.codecogs.com/gif.latex?D_{ij}) is positive pair and ![](http://latex.codecogs.com/gif.latex?D_{ik}~~D_{jl}) is negative pair

ref. [*Deep Metric Learning via Lifted Structured Feature Embedding*](https://arxiv.org/abs/1511.06452)

### No Fuss Distance Metric Learning using Proxies

![](https://ws2.sinaimg.cn/large/006tNc79gy1fz7j25bkdpj30be01o0sv.jpg)

Proxies loss with static proxy assignment

ref. [*No Fuss Distance Metric Learning using Proxies*](https://arxiv.org/abs/1703.07464)



## Examples

All examples is fine-tuned on Resnet50 pre-trained model.

### Dataset
* CUB-200-2011

```
python3 main.py --dataroot DATASET_PATH --batch_size 128 --batch_size_test 32 --embedding_dim 128 --eval_interval 50 --loss contrastive --dataset CUB-200-2011 --seed 18 --checkpoints_path SAVING_PATH --margin 0.1 --lr 1e-4 --maxiter 10000 --decay_iter_step 5000 --decay_gamma 0.1 --l2norm --cuda
```

* In-shop Clothes Retrieval

```
python3 main.py --dataroot DATASET_PATH --batch_size 128 --batch_size_test 32 --embedding_dim 128 --eval_interval 50 --loss contrastive --dataset In-shop --seed 18 --checkpoints_path SAVING_PATH --margin 0.1 --lr 1e-4 --maxiter 10000 --decay_iter_step 5000 --decay_gamma 0.1 --l2norm --cuda
```

