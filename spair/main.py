import sys 
sys.path.append('..')

import tensorflow as tf
from spair import get_model
import os
from data import get_cub_dataset

import trainer
import argparse
from utils import dotdict
from augmentation import Augmentator
import warnings

warnings.filterwarnings("ignore", module="matplotlib")

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

parser = argparse.ArgumentParser()
parser.add_argument('--learning_rate', type=float, nargs='?', default=1e-4)
parser.add_argument('--beta', type=float, nargs='?', default = 0.5)
parser.add_argument('--dataset', type=str, nargs='?', default = 'cub_solid_fixed')
parser.add_argument('--channel', type=int, nargs='?', default = 3)
parser.add_argument('--training_steps', type=int, nargs='?', default = 100000)
parser.add_argument('--batch_size', type=int, nargs='?', default=32)
parser.add_argument('--runs', type=int, nargs='?', default=1)
parser.add_argument('--tau', type=float, nargs='?', default = 0.8)
parser.add_argument('--object_size', type=int, nargs='?', default = 32)
parser.add_argument('--latent_size', type=int, nargs='?', default = 128)
parser.add_argument('-no_label', action='store_true')
parser.add_argument('--anneal_until', type=float, nargs='?', default = 1.0)
parser.add_argument('--z_pres_anneal_step', type=float, nargs='?', default = 10000.0)
parser.add_argument('--prior_z_zoom', type=float, nargs='?', default = 0.0)
parser.add_argument('--prior_z_zoom_start', type=float, nargs='?', default = 10.0)
parser.add_argument('--reconstruction_weight', type=float, nargs='?', default = 1.0)
parser.add_argument('--bg_latent_size', type=int, nargs='?', default = 4)
parser.add_argument('--local_latent_size', type=int, nargs='?', default = 64)
parser.add_argument('--z_bg_beta', type=float, nargs='?', default = 10.0) #10 for solid bg, 4 for ckb
parser.add_argument('--z_l_beta', type=float, nargs='?', default = 0.1)
parser.add_argument('--z_what_beta', type=float, nargs='?', default = 0.1)
parser.add_argument('-allow_growth', action='store_true')
parser.add_argument('--model', type=str, nargs='?', default = 'spair')
parser.add_argument('--patch_size', type=int, nargs='?', default = 4)
parser.add_argument('--augmentation', type=str, nargs='?', default = 'scramble')
parser.add_argument('-split_z_l', action='store_true')
parser.add_argument('-dense_bg', action='store_true')
parser.add_argument('-dense_local', action='store_true')
parser.add_argument('-concat_bg', action='store_true')
parser.add_argument('-concat_z_what', action='store_true')
parser.add_argument('-concat_backbone', action='store_true')

args = parser.parse_args()

if args.allow_growth:
    from tensorflow.compat.v1 import ConfigProto
    from tensorflow.compat.v1 import InteractiveSession
    config = ConfigProto()
    config.gpu_options.allow_growth = True
    session = InteractiveSession(config=config)


BATCH_SIZE = args.batch_size
AUTOTUNE = tf.data.experimental.AUTOTUNE

args_dict = vars(args)
config = dotdict(args_dict)
config.label = not config.no_label

print('Config:',config)

train_dataset, test_dataset, input_shape, test_size = get_cub_dataset(config.dataset,channel=config.channel)
config.image_size = input_shape[1:4]
config.test_size = test_size[1:4]

augmentor = Augmentator(type=config.augmentation,size=config.patch_size)

if config.model=='lg_spair':
    train_dataset = train_dataset.cache().shuffle(20000).repeat().map(augmentor.augment, num_parallel_calls=AUTOTUNE).batch(BATCH_SIZE).prefetch(buffer_size=AUTOTUNE)

    if isinstance(test_dataset,list):
        for i in range(len(test_dataset)):
            test_dataset[i] = test_dataset[i].map(lambda x,y : (augmentor.augment(x),y), num_parallel_calls=AUTOTUNE).batch(BATCH_SIZE).prefetch(buffer_size=AUTOTUNE)
    else:
        test_dataset = [test_dataset.map(lambda x,y : (augmentor.augment(x),y), num_parallel_calls=AUTOTUNE).batch(BATCH_SIZE).prefetch(buffer_size=AUTOTUNE)]

else:
    train_dataset = train_dataset.cache().shuffle(20000).repeat().batch(BATCH_SIZE).prefetch(buffer_size=AUTOTUNE)

    if isinstance(test_dataset,list):
        for i in range(len(test_dataset)):
            test_dataset[i] = test_dataset[i].batch(BATCH_SIZE).prefetch(buffer_size=AUTOTUNE)
    else:
        test_dataset = [test_dataset.batch(BATCH_SIZE).prefetch(buffer_size=AUTOTUNE)]

for _ in range (args.runs):
    print('Creating model...')
    model = get_model(config)
    print(type(model))
    # model = SPAIR(image_size = input_shape[1:4], test_size = test_size[1:4],object_size=config.object_size, latent_size=config.latent_size,
    #     tau=config.tau, z_what_dense_size=config.z_what_dense_size, bg_latent_sizemodel=config.bg_model, bg_latent_size=config.bg_latent_size)

    # model(tf.random.uniform([BATCH_SIZE,test_size[1],test_size[2],test_size[3]]),training=False) #build test stn
    if config.model=='lg_spair' or config.model=='lg_spair_2':
        model(tf.random.uniform([BATCH_SIZE,input_shape[1],input_shape[2],input_shape[3]*2]))

    else:
        model(tf.random.uniform([BATCH_SIZE,input_shape[1],input_shape[2],input_shape[3]])) #build training stn
    model.summary()
    optimizer = tf.keras.optimizers.Adam(config.learning_rate, clipnorm=1.0)

    print('Training SPAIR')
    trainer.train_spair(model, optimizer, config.dataset, train_dataset, test_dataset, config)