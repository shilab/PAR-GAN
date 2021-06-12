# PAR-GAN: Improving the Generalization of Generative Adversarial Networks Against Membership Inference Attacks
This is a demo of the PAR-GAN framework. For more details, please refer to our KDD 2021 paper 'PAR-GAN: Improving the Generalization of Generative Adversarial Networks Against Membership Inference Attacks'.

<!-- # Additinal resources
paper| slides ()| presentations -->

## Introduction
Recent works have shown that Generative Adversarial Networks (GANs) may generalize poorly and thus are vulnerable to privacy attacks. In this paper, we seek to improve the generalization of GANs from a perspective of privacy protection, specifically in terms of defending against the membership inference attack (MIA) which aims to infer whether a particular sample was used for model training. We design a GAN framework, partition GAN (PAR-GAN), which consists of one generator and multiple discriminators trained over disjoint partitions of the training data. The key idea of PAR-GAN is to reduce the generalization gap by approximating a mixture distribution of all partitions of the training data. Our theoretical analysis shows that PAR-GAN can achieve global optimality just like the original GAN. Our experimental results on simulated data and multiple popular datasets demonstrate that PAR-GAN can improve the generalization of GANs while mitigating information leakage induced by MIA.


### Prerequirements
1. Python 3.6.8
2. Tensorflow 2.1.0
3. sklearn 0.22.1
4. skimage 0.16.2

### Datasets
We used two simulated toy datasets, two public image datasets, and two structured datasets. 
The two simulated toy datasets are generated blobs and circles with Gaussian noise, which can be found in Blobs and Circles directories. 
The two structured datasets are the Texas hospital dataset and the Broward County dataset, which can be found in Hospital and Broward directories.
Two public image datasets are MNIST and CIFAR10, which can be download by using Tensorflow datasets.

### Run the code
We provide two PAR-GAN demos (with JS divergence and Wasserstain distance) for each dataset in jupyter notebooks. After setuping the prerequirements, the demos can be run directly.  



# Citation
Please cite our paper if you find this code useful for your research:
```
@inproceedings{chen2021par-gan,
  title = {PAR-GAN: Improving the Generalization of Generative Adversarial Networks Against Membership Inference Attacks},
  author = {Chen, Junjie and Wang, Wendy Hui and Gao, Hongchang and Shi, Xinghua},
  booktitle = {Proceedings of the 27th ACM SIGKDD International Conference on Knowledge Discovery and Data Mining},
  year = {2021}，
  publisher = {ACM},
}
```
