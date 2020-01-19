import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
import os 

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

width = height = 32
channel = 3
patch_size_x = 8 ;patch_size_y = 8

class Augmentator(object):
    def __init__(self,type,size=1,mean=0,std=1):
        self.size = size
        if type=='scramble':
            self.augment = self.scramble
        elif type=='mix_scramble':
            self.augment = self.mix_scramble  
        elif type=='blur':
            self.augment = self.gaussian_blur
            self.pointwise_filter = tf.eye(3, batch_shape=[1, 1])

        elif type=='high_low_pass':
            self.augment = self.high_low_pass
            self.kernel = self.gaussian_kernel(size,mean,std)
            self.kernel = tf.tile(self.kernel[:, :, tf.newaxis, tf.newaxis], [1, 1, 3, 1])
            self.pointwise_filter = tf.eye(3, batch_shape=[1, 1])
            self.paddings = [[size,size],[size,size],[0,0]]
        elif type=='no_op':
            self.augment = self.no_op


    def gaussian_kernel(self,size,mean,std):
        """Makes 2D gaussian Kernel for convolution."""
        d = tfp.distributions.Normal(mean, std)
        vals = d.prob(tf.range(start = -size, limit = size + 1, dtype = tf.float32))
        gauss_kernel = tf.einsum('i,j->ij',vals,vals)
        return gauss_kernel / tf.reduce_sum(gauss_kernel)

    def get_random_patch_size(self):
        return np.random.choice([1,2,4,8])

    def scramble(self,x):
        # assume square patch
        n_row,n_col,n_channel = x.shape
        n_patch = n_row*n_col // (self.size**2)
        patches = tf.image.extract_patches(tf.expand_dims(x,0),sizes=[1,self.size,self.size,1],strides=[1,self.size,self.size,1],rates=[1, 1, 1, 1],padding='VALID')
        patches = tf.reshape(patches,[n_patch,self.size,self.size,n_channel])
        patches = tf.random.shuffle(patches)
        # rand_idx = tf.reshape(tf.random.shuffle(tf.range(0,n_patch)),[n_patch])
        # patches = tf.gather(patches, rand_idx, axis=0)
        rows = tf.split(patches,n_col//self.size,axis=0)
        rows = [tf.concat(tf.unstack(x),axis=1) for x in rows]
        x_aug = tf.concat(rows,axis=0)

        x_aug = tf.convert_to_tensor(x_aug)
        return tf.concat([x, x_aug],axis=2)

    def mix_scramble(self,x):
        # assume square patch
        # sizes = tf.convert_to_tensor([1,2,4,8])
        # idx = tf.random.categorical([tf.ones_like(sizes)], 1)
        # print(idx)
        # patch_size = int(sizes[idx[0][0]])
        patch_size = self.get_random_patch_size()
        print('Patch size:',patch_size)
        window = [1,patch_size,patch_size,1]
        print('Window:',window)

        n_row,n_col,n_channel = x.shape
        n_patch = n_row*n_col // (patch_size**2)
        patches = tf.image.extract_patches(tf.expand_dims(x,0),sizes=window,strides=window,rates=[1, 1, 1, 1],padding='VALID')
        patches = tf.reshape(patches,[n_patch,patch_size,patch_size,n_channel])
        patches = tf.random.shuffle(patches)
        rows = tf.split(patches,n_col//patch_size,axis=0)
        rows = [tf.concat(tf.unstack(x),axis=1) for x in rows]
        x_aug = tf.concat(rows,axis=0)

        x_aug = tf.convert_to_tensor(x_aug)

        return tf.concat([x, x_aug],axis=2)

    def gaussian_blur(self,x):
        #create random gaussian blur filter
        mean = 0
        std = tf.random.uniform(shape=[],minval=5,maxval=10,dtype=tf.float32) # std [5-10]
        size = tf.random.uniform(shape=[],minval=3,maxval=7,dtype=tf.int32) # size [7-15]

        self.kernel = self.gaussian_kernel(size,mean,std)
        self.kernel = tf.tile(self.kernel[:, :, tf.newaxis, tf.newaxis], [1, 1, 3, 1])
        self.paddings = tf.convert_to_tensor([[size,size],[size,size],[0,0]])
        x_aug = tf.nn.separable_conv2d(tf.expand_dims(tf.pad(x,self.paddings,'SYMMETRIC'), 0), self.kernel, self.pointwise_filter,strides=[1, 1, 1, 1], padding='VALID')
        x_aug = tf.squeeze(x_aug)
        return tf.concat([x, x_aug],axis=2)


    def high_low_pass(self,x):
        x_low = tf.nn.separable_conv2d(tf.expand_dims(tf.pad(x,self.paddings,'SYMMETRIC'), 0), self.kernel, self.pointwise_filter,strides=[1, 1, 1, 1], padding='VALID')
        x_low = tf.squeeze(x_low)
        x_high = x - x_low
        return tf.concat([x, x_high, x_low],axis=2)

    def no_op(self,x):
        return x



