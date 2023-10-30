<h1 align="center">FeAT: Feature Augmented Training</h1>
<p align="center">
    <a href="https://arxiv.org/abs/2304.11327"><img src="https://img.shields.io/badge/arXiv-2304.11327-b31b1b.svg" alt="Paper"></a>
    <a href="https://github.com/LFhase/FeAT"><img src="https://img.shields.io/badge/-Github-grey?logo=github" alt="Github"></a>
    <!-- <a href=""><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Colab"></a> -->
    <a href="https://openreview.net/forum?id=eozEoAtjG8"> <img alt="License" src="https://img.shields.io/static/v1?label=Pub&message=NeurIPS%2723&color=blue"> </a>
    <a href="https://github.com/LFhase/PAIR/blob/main/LICENSE"> <img alt="License" src="https://img.shields.io/github/license/LFhase/PAIR?color=blue"> </a>
    <!-- <a href="https://neurips.cc/virtual/2022/poster/54643"> <img src="https://img.shields.io/badge/Video-grey?logo=Kuaishou&logoColor=white" alt="Video"></a> -->
    <!-- <a href="https://lfhase.win/files/slides/PAIR.pdf"> <img src="https://img.shields.io/badge/Slides-grey?&logo=MicrosoftPowerPoint&logoColor=white" alt="Slides"></a> -->
   <!--  <a href="https://icml.cc/media/PosterPDFs/ICML%202022/a8acc28734d4fe90ea24353d901ae678.png"> <img src="https://img.shields.io/badge/Poster-grey?logo=airplayvideo&logoColor=white" alt="Poster"></a> -->
</p>

This repo contains the sample code for reproducing the results of our NeurIPS 2023: *[Understanding and Improving Feature Learning for Out-of-Distribution Generalization](https://arxiv.org/abs/2304.11327)*, which has also been presented as ***spotlight*** at [ICLR DG](https://domaingen.github.io/), and at [ICML SCIS](https://sites.google.com/view/scis-workshop/home) Workshop. 😆😆😆

Updates:

<!-- - [x] Results are updated to [Wilds leaderboard](https://wilds.stanford.edu/leaderboard/). Note there are some slight differences due to the [evaluation](./WILDS/README.md).
- [x] Camera ready version of the paper [link](https://openreview.net/forum?id=esFxSb_0pSL)!
- [x] PAIR is accepted as an ***oral presentation*** by [ICLR DG](https://domaingen.github.io/) workshop! -->

- [x] Camera-ready version of the paper is updated [link](https://arxiv.org/abs/2304.11327)!
- [ ] Detailed running instructions will be released soon!

## What feature does ERM learn for generalization?
Empirical risk minimization (ERM) is the *de facto* objective adopted in Machine Learning and obtains impressive generalization performance. Nevertheless, ERM is shown to be prone to spurious correlations, and is suspected to learn predictive but **<ins>spurious</ins>** features for minimizing the empirical risk.
However, recently [Rosenfeld et al., 2022](https://arxiv.org/abs/2202.06856);[Kirichenko et al., 2022](https://arxiv.org/abs/2204.02937) empirically show that ERM already learn **<ins>invariant features</ins>** that hold an invariant relation with the label for in-distribution and Out-of-Distribution (OOD) generalization.

<p align="center"><img src="./figures/feat_motivation.png"></p>

We resolve the puzzle by theoretically proving that ERM essentially learns both **<ins>spurious and invariant features</ins>**. 
Meanwhile, we also find OOD objectives such as IRMv1 can **<ins>hardly learn new features</ins>** even at the begining of the optimization.
Therefore, when optimizing OOD objectives such as IRMv1, pre-training the model with ERM is usually necessary for satisfactory performance. 
As shown in the right subfigure, the OOD performance of various OOD objective first grows with more ERM pre-training epochs. 

However, **<ins>ERM has its preference to learning features</ins>** depending on the inductive biases of the dataset and the architecture. The limited feature learning can pose a bottleneck for OOD generalization. Therefore, we propose Feature Augmented Training (FeAT), that aims to learn all features so long as they are useful for generalization. Iteratively, FeAT divides the training data $\mathcal{D}_{tr}$ into **<ins>augmentation sets</ins>** $D^a$ where the features not sufficiently well learned by the model, and the **<ins>retention sets</ins>** $D^r$ that contain features already learned by the current model at the round. Learning on the partitioned datasets with FeAT augments the model with new features contained in the growing augmentation sets while retaining the already learned features contained in the retention sets, which will lead the model to learn **<ins>richer features</ins>** for OOD training and obtain a better OOD performance.

For more interesting stories of rich feature learning, please read more into the repositories [Bonsai](https://github.com/TjuJianyu/RFC), [RRL](https://github.com/TjuJianyu/RRL) and the [blog](https://www.jianyuzhang.com/blog/rich-representation-learning) by [Jianyu](https://www.jianyuzhang.com/home). 😆


## Structure of Codebase

The whole code base contain four parts, corresponding to experiments presented in the paper:

- `ColoredMNIST`: Proof of Concept on ColoredMNIST
- `WILDS`: Verification of FeAT in WILDS


### Dependencies
We are running with cuda=10.2 and python=3.8.12 with the following key libraries:
```
wilds==2.0.0
torch==1.9.0
```

## ColoredMNIST

The corresponding code is in the folder [ColoredMNIST](./ColoredMNIST).
The code is modified from [RFC](https://github.com/TjuJianyu/RFC/).

<!-- To reproduce results of FeAT, simply run the following commands under the directory:

For the original ColoredMNIST data (CMNIST-25):

```
python run_exp.py  --methods pair  --verbose True --penalty_anneal_iters 150 --dataset coloredmnist025 --n_restarts 10 --lr 0.1 --opt 'pair' 
```

For the modified ColoredMNIST data (CMNIST-01):

```
python run_exp.py  --methods pair  --verbose True --penalty_anneal_iters 150 --dataset coloredmnist01 --n_restarts 10 --lr 0.01 --opt 'pair'
``` -->

## WILDS

The corresponding code is in the folder [WILDS](./WILDS).
The code is modified from [PAIR](https://github.com/LFhase/PAIR) and [spurious-feature-learning](https://github.com/izmailovpavel/spurious_feature_learning).


<!-- To run with wilds codes,
for example,

```
python main.py --need_pretrain --data-dir ./data --dataset civil --algorithm pair -pc 3 --seed 0 -ac 1e-4 -al
```

We add additional commands to control `PAIR-o`:

- `-pc`: specify preferences;
- `--use_old`: to avoid repeated pretraining of ERM and directly use the pretrained weights;

To avoid negative loss inputs, we use the following commands to adjust IRMv1 loss values:

- `-al` and `-ac`: adjust negative irm penalties in pair by multiplying a negative number;
- `-ai`: adjust negative irm penalties in pair by adding up a sufficient large number;

We also provide a accelerated mode by freezing the featurizer by specifying `--frozen`.
The running scripts fow wilds experiments can be found [here](./WILDS/scripts). -->



## Misc

If you find our paper and repo useful, please cite our paper:

```bibtex
@inproceedings{
chen2023FeAT,
title={Understanding and Improving Feature Learning for Out-of-Distribution Generalization},
author={Yongqiang Chen and Wei Huang and Kaiwen Zhou and Yatao Bian and Bo Han and James Cheng},
booktitle={Thirty-seventh Conference on Neural Information Processing Systems},
year={2023},
url={https://openreview.net/forum?id=eozEoAtjG8}
}
```
