# SPLIT
Original implementation of _**S**eparated **P**aths for **L**ocal and Global **I**nforma**t**ion framework_ (**SPLIT**) in TensorFlow 2.

***An Explicit Local and Global Representation Disentanglement Framework with Applications in Deep Clustering and Unsupervised Object Detection.***

Rujikorn Charakorn, Yuttapong Thawornwattana, Sirawaj Itthipuripat, Poramate Manoonpong, and Nat Dilokthanakul

___



## Installation

Tested on Ubuntu 18.04 and Linux Mint 19.2 with Python 3.6

`pip install -r requirements.txt`


## Experiments

All results will be in `output/` folder.
___
### SPLIT-VAE

#### Generation
- SVHN

```
cd vae
python main.py --beta 40 --patch_size 1
```

- CelebA

```
cd vae
python main.py --beta 120 --patch_size 8 --dataset celeba64 -no_label
```

#### Style transfer and reconstruction accuracy table (SVHN)
- SVHN

```
cd vae
python main.py --beta 1 --patch_size 1
```

- CelebA

```
cd vae
python main.py --beta 30 --patch_size 8 --dataset celeba64 -no_label
```  
___
### GMVAE and SPLIT-GMVAE

:soon:
___
### SPAIR and SPLIT-SPAIR

:soon:
