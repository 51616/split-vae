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

#### Generation (Fig. 4)
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

#### Style transfer (Fig. 5) and reconstruction accuracy table (Table 1)
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

#### Unsupervised clustering (Table 2)
- SVHN

```
cd vae
python main.py --model lggmvae --beta 40 --alpha 40 --y_size 30 --patch_size 4 --dataset svhn --training_steps 3000000
```

#### Cluster generation (Fig. 6) and Unseen data clustering visualisation (Fig. 7)
- SVHN

```
cd vae
python main.py --model lggmvae --beta 40 --alpha 40 --y_size 30 --patch_size 4 --dataset svhn --training_steps 3000000 -viz
```

- CelebA

```
cd vae
python main.py --model lggmvae --beta 120 --alpha 40 --y_size 30 --patch_size 8 --dataset celeba64 -no_label -viz --training_steps 3000000
```
___
### SPAIR and SPLIT-SPAIR
#### Fig. 8 and 9
- Multi-Bird-Easy

GMVAE
```
cd spair
python main.py --dataset cub_solid_fixed --z_bg_beta 10 --latent_size 64 --bg_latent_size 4 --model bg_spair -dense_bg --training_steps 200000
```

SPLIT-VAE
```
cd spair
python main.py --dataset cub_solid_fixed --z_bg_beta 10 --patch_size 8 --latent_size 64 --bg_latent_size 4 --local_latent_size 4 --model lg_spair -split_z_l -concat_z_what -dense_local -dense_bg --training_steps 200000
```

- Multi-Bird-Hard

GMVAE
```
cd spair
python main.py --dataset cub_ckb_rot_6 --z_bg_beta 1 --latent_size 64 --bg_latent_size 64 --model bg_spair -dense_bg --training_steps 200000
```

SPLIT-VAE
```
cd spair
python main.py --dataset cub_ckb_rot_6 --z_bg_beta 1 --patch_size 8 --latent_size 64 --bg_latent_size 64 --local_latent_size 64 --model lg_spair -split_z_l --z_what_beta 0.5 -concat_z_what -dense_local -dense_bg --training_steps 200000
```
