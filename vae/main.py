import sys 
sys.path.append('..')

import tensorflow as tf
import argparse
from utils import dotdict
from model import LGVae, LGGMVae, GMVae
import trainer
import data
import os
from augmentation import Augmentator

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

parser = argparse.ArgumentParser()  
parser.add_argument('-viz', action='store_true') # visualize results
parser.add_argument('--global_latent_dims', type=int, nargs='?', default=128)
parser.add_argument('--local_latent_dims', type=int, nargs='?', default=128)
parser.add_argument('--learning_rate', type=float, nargs='?', default=1e-4)
parser.add_argument('--beta', type=float, nargs='?', default = 40)
parser.add_argument('--dataset', type=str, nargs='?', default = 'svhn')
parser.add_argument('--training_steps', type=int, nargs='?', default = 1000000)
parser.add_argument('--batch_size', type=int, nargs='?', default=64)
parser.add_argument('--patch_size', type=int, nargs='?', default=1)
parser.add_argument('--augmentation', type=str, nargs='?', default='scramble')
parser.add_argument('-no_label', action='store_true')
parser.add_argument('--model', type=str, nargs='?', default='lgvae')  
parser.add_argument('--y_size', type=int, nargs='?', default=30)
parser.add_argument('--tau', type=float, nargs='?', default = 0.4)
parser.add_argument('--alpha', type=float, nargs='?', default = 40)
parser.add_argument('-allow_growth', action='store_true')

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

augmentor = Augmentator(type=config.augmentation,size=config.patch_size)
train_dataset, test_dataset, input_shape = data.get_dataset(dataset=args.dataset,get_label = config.label)
if config.label:
  train_dataset = train_dataset.shuffle(20000).repeat().map(lambda x,y : (augmentor.augment(x),y), num_parallel_calls=8).batch(BATCH_SIZE).prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
  test_dataset = test_dataset.shuffle(20000).map(lambda x,y : (augmentor.augment(x),y), num_parallel_calls=8).batch(BATCH_SIZE).prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
else:
  train_dataset = train_dataset.shuffle(20000).repeat().map(augmentor.augment, num_parallel_calls=8).batch(BATCH_SIZE).prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
  test_dataset = test_dataset.shuffle(20000).map(augmentor.augment, num_parallel_calls=8).batch(BATCH_SIZE).prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

if args.model=='lgvae':
  model = LGVae(global_latent_dims = config.global_latent_dims, local_latent_dims = config.local_latent_dims, image_shape = input_shape)
  optimizer = tf.keras.optimizers.Adam(learning_rate = config.learning_rate)
elif args.model=='lggmvae':
  lr_schedule = tf.optimizers.schedules.ExponentialDecay(config.learning_rate,decay_steps=1000000,decay_rate=0.4,staircase=True)
  optimizer = tf.keras.optimizers.Adam(learning_rate = lr_schedule)
  model = LGGMVae(global_latent_dims = config.global_latent_dims, local_latent_dims = config.local_latent_dims, image_shape = input_shape, y_size=config.y_size, tau=config.tau)
elif args.model=='gmvae':
  lr_schedule = tf.optimizers.schedules.ExponentialDecay(config.learning_rate,decay_steps=1000000,decay_rate=0.4,staircase=True)
  optimizer = tf.keras.optimizers.Adam(learning_rate = lr_schedule)
  model = GMVae(global_latent_dims = config.global_latent_dims, image_shape = input_shape, y_size=config.y_size, tau=config.tau)
model(tf.zeros([8,input_shape[1],input_shape[2],6]))
model.summary()
  
print('Training local-global autoencoder')

trainer.train_local_global_autoencoder(model, optimizer, config.dataset, train_dataset, test_dataset, config = config)

