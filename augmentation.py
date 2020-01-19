import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
import os 
from vae import data
import matplotlib.pyplot as plt
import skimage
from scipy import ndimage

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





# def scramble(x):
#     # Shuffle to destroy global context
#     print('Scramble')
#     patch_x_nums = width // patch_size_x
#     patch_y_nums = height // patch_size_y

#     patch_nums = patch_x_nums * patch_y_nums
#     # seed = tf.random.uniform(shape=[1], minval=0, maxval=2**16-1, dtype=tf.int32, seed=None, name=None)
#     patch_idx = np.arange(patch_nums)
#     np.random.shuffle(patch_idx)

#     x_list = []
#     for patch_idx in patch_idx:
#         x_patch = x[(patch_idx//patch_y_nums)*patch_size_y:(patch_idx//patch_y_nums + 1)*patch_size_y, (patch_idx%patch_x_nums) * patch_size_x:(patch_idx%patch_x_nums + 1)*patch_size_x]
#         x_list.append(x_patch)
        
#     x_row_aug = [x_list[i*patch_x_nums:(i+1)*patch_x_nums] for i in range(patch_y_nums)]
#     # x_row_aug = tf.random.shuffle(x_row_aug)
#     x_aug = [tf.concat(x,axis=1) for x in x_row_aug]
#     # x_aug = tf.random.shuffle(x_aug)
#     x_aug = tf.concat(x_aug,axis=0)
    
#     return tf.concat([x, x_aug],axis=2) #first 3 channels are original RGB image, last 3 are augmented RGB image

# def random_scramble(x,patch_size):
#     # Shuffle to destroy global context
#     # print('Scramble')
#     patch_size_x = patch_size_y = patch_size
#     patch_x_nums = width // patch_size_x
#     patch_y_nums = height // patch_size_y
#     patch_nums = patch_x_nums * patch_y_nums
#     patch_idx = np.arange(patch_nums)
#     np.random.shuffle(patch_idx)

#     x_list = []
#     for patch_idx in patch_idx:
#         x_patch = x[(patch_idx//patch_y_nums)*patch_size_y:(patch_idx//patch_y_nums + 1)*patch_size_y, (patch_idx%patch_x_nums) * patch_size_x:(patch_idx%patch_x_nums + 1)*patch_size_x]
#         x_list.append(x_patch)
    
#     x_list = tf.unstack(tf.random.shuffle(x_list))
#     x_row_aug = [x_list[i*patch_x_nums:(i+1)*patch_x_nums] for i in range(patch_y_nums)]
#     x_aug = [tf.concat(x,axis=1) for x in x_row_aug]
#     x_aug = tf.concat(x_aug,axis=0)

    
#     return tf.concat([x, x_aug],axis=2) #first 3 channels are original RGB image, last 3 are augmented RGB image








# def rotate(x: tf.Tensor) -> tf.Tensor:
#     """Rotation augmentation

#     Args:
#         x: Image

#     Returns:
#         Augmented image
#     """

#     # Rotate 0, 90, 180, 270 degrees
#     print('rotate')
#     return tf.image.rot90(x, tf.random.uniform(shape=[], minval=0, maxval=4, dtype=tf.int32))



if __name__=='__main__':

    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

    # a = np.arange(height*width*channel, dtype=np.float).reshape((height,width,channel))
    # a = tf.convert_to_tensor(a)
    # print(a.shape)
    # print(a[:,:,0])
    # a, a_aug = scramble(a)
    # print(a_aug)
    # print(a_aug.shape)
    # print(a_aug[:,:,0])

    # augmentator = Augmentator(type='blur')
    # train_dataset, test_dataset, input_shape = data.get_dataset(dataset='svhn', get_label=True)
    # train_dataset = train_dataset.map(lambda x,y : (augmentator.augment(x),y)).batch(16)
    # for ep in range(2):
    #   for images,labels in train_dataset:
    #     # Get single batch
    #     x,x_high = images[:,:,:,:3], images[:,:,:,3:]
    #     for i in range(1):
    #         plt.imshow((x[i]+1)*0.5)
    #         plt.show()
    #         plt.imshow((x_high[i]+1)*0.5)
    #         plt.show()
    #     break


    augmentator = Augmentator(type='scramble',size=8)
    train_dataset, test_dataset, input_shape = data.get_dataset(dataset='svhn')
    train_dataset = train_dataset.map(augmentator.augment).batch(16)
    for ep in range(2):
      for images in train_dataset:
        # Get single batch
        x,x_high = images[:,:,:,:3], images[:,:,:,3:]
        for i in range(1):
            plt.imshow((x[i]+1)*0.5)
            plt.show()
            plt.imshow((x_high[i]+1)*0.5)
            plt.show()
        break


    
    # augmentator = Augmentator(type='high_low_pass',size=2,mean=0,std=1)
    # train_dataset, test_dataset, input_shape = data.get_dataset(dataset='svhn', get_label=True)
    # train_dataset = train_dataset.map(lambda x,y : (augmentator.augment(x),y)).batch(16)
    # for ep in range(1):
    #   for images,labels in train_dataset:
    #     # Get single batch
    #     x,x_high,x_low = images[:,:,:,:3], images[:,:,:,3:6], images[:,:,:,6:]
    #     for i in range(16):
    #         plt.imshow((x[i]+1)*0.5)
    #         plt.show()
    #         plt.imshow((x_high[i]+1)*0.5)
    #         plt.show()
    #         plt.imshow((x_low[i]+1)*0.5)
    #         plt.show()
    #     break