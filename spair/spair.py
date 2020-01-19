import tensorflow as tf
import numpy as np

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Layer, Dense, Conv2D, Flatten, GaussianNoise
from utils import *

def get_model(config):
    print('config.model:',config.model)
    if config.model=='lg_spair':
        return LGSPAIR(config)
    elif config.model=='spair' or config.model == 'bg_spair':
        return SPAIR(config)
    elif config.model=='lg_glimpse_spair':
        return LGGlimpseSPAIR(config)
    else:
        raise NotImplementedError('Model type not implemented')

class SPAIR(Model):
    def __init__(self,config,name='SPAIR'):
        super(SPAIR, self).__init__(name=name)

        image_size, test_size, object_size, latent_size, tau, model, bg_latent_size =\
         config.image_size, config.test_size, config.object_size, config.latent_size, config.tau, config.model, config.bg_latent_size

        self.model = model
        self.encoder = Encoder(object_size, latent_size, tau)
        self.decoder = Decoder(image_size, test_size, object_size, latent_size)
        if model=='bg_spair':
            self.bg_model = BackgroundModel(image_size, bg_latent_size, image_size[2])
        self.renderer = Renderer(num_channel=image_size[2])
        

    def call(self, inputs, training=False):
        (z_what, z_what_mean, z_what_sigma, z_where, z_where_mean, z_where_sigma,
            z_depth, z_depth_mean, z_depth_sigma, z_pres, z_pres_logits, z_pres_pre_sigmoid, all_glimpses) = self.encoder(inputs, training)
        obj_recon_unnorm, obj_recon_alpha, obj_full_recon_unnorm, obj_bbox_mask = self.decoder([z_what, z_where, z_depth, z_pres, z_pres_logits, tf.shape(inputs)], training)
        bg_recon = 0.0
        if self.model=='bg_spair':
            bg_recon, z_bg, z_bg_mean, z_bg_sig = self.bg_model(inputs)
            x_recon = self.renderer([obj_full_recon_unnorm, bg_recon, z_depth, z_pres, z_pres_logits],training)
            return (x_recon, z_what, z_what_mean, z_what_sigma, z_where, z_where_mean, z_where_sigma,
                z_depth, z_depth_mean, z_depth_sigma, z_pres, z_pres_logits, z_pres_pre_sigmoid, all_glimpses,
                obj_recon_unnorm, obj_recon_alpha, obj_full_recon_unnorm,obj_bbox_mask, z_bg, z_bg_mean, z_bg_sig)
        else:
            x_recon = self.renderer([obj_full_recon_unnorm, bg_recon, z_depth, z_pres, z_pres_logits],training)
            return (x_recon, z_what, z_what_mean, z_what_sigma, z_where, z_where_mean, z_where_sigma,
                z_depth, z_depth_mean, z_depth_sigma, z_pres, z_pres_logits, z_pres_pre_sigmoid, all_glimpses,
                obj_recon_unnorm, obj_recon_alpha, obj_full_recon_unnorm, obj_bbox_mask)


class LGSPAIR(Model):

    def __init__(self,config,name='LGSPAIR'):
        super(LGSPAIR, self).__init__(name=name)

        image_size, test_size, object_size, latent_size, tau, bg_model, bg_latent_size =\
         config.image_size, config.test_size, config.object_size, config.latent_size, config.tau, config.bg_model, config.bg_latent_size
        
        self.bg_model = bg_model
        self.concat_z_what = config.concat_z_what
        self.concat_backbone = config.concat_backbone
        self.concat_z_bg = config.concat_z_bg

        self.encoder = Encoder(object_size, latent_size, tau, concat=config.concat_backbone)
        self.decoder = Decoder(image_size, test_size, object_size, latent_size)
        self.renderer = Renderer(num_channel=image_size[2])
        
        if config.dense_bg:
            self.bg_encoder = ImageEncoderDense(image_size, bg_latent_size, name='bg_encoder')
            self.bg_decoder = ImageDecoderDense(image_size, bg_latent_size + config.local_latent_size, name='bg_decoder') # z_bg + z_l
        else:
            self.bg_encoder = ImageEncoder(image_size, bg_latent_size, name='bg_encoder')
            self.bg_decoder = ImageDecoder(image_size, bg_latent_size + config.local_latent_size, name='bg_decoder') # z_bg + z_l

        if config.dense_local:
            self.x_hat_encoder = ImageEncoderDense(image_size, config.local_latent_size, name='x_hat_encoder')
            self.x_hat_decoder = ImageDecoderDense(image_size, config.local_latent_size, name='x_hat_decoder')
        else:
            self.x_hat_encoder = ImageEncoder(image_size, config.local_latent_size, name='x_hat_encoder')
            self.x_hat_decoder = ImageDecoder(image_size, config.local_latent_size, name='x_hat_decoder')

    def call(self, inputs, training=False):
        x, x_hat = inputs[:,:,:,:3] , inputs[:,:,:,3:]

        z_l, z_l_mean, z_l_sig = self.x_hat_encoder(x_hat) # [B,latent//2]
        z_bg, z_bg_mean, z_bg_sig = self.bg_encoder(x)
        if self.concat_backbone:
            x = [x,z_l]
        (z_what, z_what_mean, z_what_sigma, z_where, z_where_mean, z_where_sigma,
            z_depth, z_depth_mean, z_depth_sigma, z_pres, z_pres_logits, z_pres_pre_sigmoid, all_glimpses) = self.encoder(x, training)

        x_hat_recon = self.x_hat_decoder(z_l)
        if self.concat_z_bg:
            z_bg = tf.concat([z_bg,z_l],axis=-1)
        bg_recon = self.bg_decoder(z_bg)

        if self.concat_z_what:
            z_what = tf.concat([z_what,tf.tile(z_l[:,tf.newaxis,tf.newaxis,:],[1,4,4,1])],axis=-1)
        obj_recon_unnorm, obj_recon_alpha, obj_full_recon_unnorm, obj_bbox_mask = self.decoder([z_what, z_where, z_depth, z_pres, z_pres_logits, tf.shape(inputs)], training)

        x_recon = self.renderer([obj_full_recon_unnorm, bg_recon, z_depth, z_pres, z_pres_logits],training)

        return (x_recon, z_what, z_what_mean, z_what_sigma, z_where, z_where_mean, z_where_sigma,
            z_depth, z_depth_mean, z_depth_sigma, z_pres, z_pres_logits, z_pres_pre_sigmoid, all_glimpses,
            obj_recon_unnorm, obj_recon_alpha, obj_full_recon_unnorm, obj_bbox_mask, z_bg, z_bg_mean, z_bg_sig, x_hat_recon, z_l, z_l_mean, z_l_sig)



class ImageEncoder(Layer):
    def __init__(self, image_size, latent_size, name='ImageEncoder'):

        super(ImageEncoder, self).__init__(name=name)

        self.num_channel = image_size[2]
        self.image_size = image_size
        self.e1 = Conv2D(filters = 32, kernel_size = 3, strides = 2, padding='same', activation = 'relu')
        self.e2 = Conv2D(filters = 64, kernel_size = 3, strides = 2, padding='same', activation = 'relu')
        self.e3 = Conv2D(filters = 128, kernel_size = 3, strides = 2, padding='same', activation = 'relu')
        self.flatten = Flatten()

        self.z_mu = Dense(latent_size, activation=None)
        self.z_sigma = Dense(latent_size, activation='softplus')

    def call(self,x,training=False):
        x = self.e3(self.e2(self.e1(x)))
        x = self.flatten(x)
        z_mean = self.z_mu(x)
        z_sig = self.z_sigma(x)
        z = z_mean + z_sig * tf.random.normal(shape=tf.shape(z_mean))
        
        return z, z_mean, z_sig


class ImageEncoderDense(Layer):
    def __init__(self, image_size, latent_size, name='ImageEncoder'):

        super(ImageEncoderDense, self).__init__(name=name)

        self.e1 = Dense(1024, activation = 'relu')
        self.e2 = Dense(500, activation = 'relu')
        self.z_mu = Dense(latent_size, activation = None)
        self.z_sigma = Dense(latent_size, activation='softplus')

        self.flatten = Flatten()

    def call(self,x,training=False):
        x = self.flatten(x)
        x = self.e2(self.e1(x))
        z_mean = self.z_mu(x)
        z_sig = self.z_sigma(x)
        z = z_mean + z_sig * tf.random.normal(shape=tf.shape(z_mean))
        
        return z, z_mean, z_sig


class ImageDecoder(Layer):
    def __init__(self, image_size, latent_size, name='ImageDecoder'):

        super(ImageDecoder, self).__init__(name=name)

        self.num_channel = image_size[2]
        self.image_size = image_size

        self.d1 = Dense(image_size[0]//8 * image_size[1]//8 * 128, activation='relu')
        self.d2 = Conv2D(filters = 128, kernel_size = 3, strides = 1, padding='same', activation='relu')
        self.d3 = Conv2D(filters = 64, kernel_size = 3, strides = 1, padding='same', activation='relu')
        self.d4 = Conv2D(filters = 32, kernel_size = 3, strides = 1, padding='same', activation='sigmoid')
        self.d5 = Conv2D(filters = self.num_channel, kernel_size = 3, strides = 1, padding='same', activation='sigmoid')

    def call(self,x,training=False):
        x = self.d1(x)
        x = tf.reshape(x,[-1,self.image_size[0]//8,self.image_size[1]//8,128]) # 16,4,4,128
        x = self.d2(x)
        x = tf.image.resize(x,[self.image_size[0]//4,self.image_size[1]//4]) #16,8,8,64
        x = self.d3(x)
        x = tf.image.resize(x,[self.image_size[0]//2,self.image_size[1]//2]) #16,16,16,32
        x = self.d4(x)
        x = tf.image.resize(x,[self.image_size[0],self.image_size[1]]) #16,32,32,2
        x = self.d5(x)
        
        return x


class ImageDecoderDense(Layer):
    def __init__(self, image_size, latent_size, name='ImageDecoder'):

        super(ImageDecoderDense, self).__init__(name=name)

        self.num_channel = image_size[2]
        self.image_size = image_size

        self.d1 = Dense(500, activation='relu')
        self.d2 = Dense(1024, activation='relu')
        self.d3 = Dense(image_size[0]*image_size[1]*image_size[2], activation='sigmoid')

    def call(self,x,training=False):
        x = self.d3(self.d2(self.d1(x)))
        x = tf.reshape(x,[-1,self.image_size[0],self.image_size[1],self.image_size[2]])
        
        
        return x


class BackgroundModel(Layer):
    def __init__(self, image_size, bg_latent_size, num_channel, name='BackgroundModel'):

        super(BackgroundModel, self).__init__(name=name)

        self.num_channel = num_channel
        self.image_size = image_size
        self.e1 = Conv2D(filters = 32, kernel_size = 3, strides = 2, padding='same', activation = 'relu')
        self.e2 = Conv2D(filters = 64, kernel_size = 3, strides = 2, padding='same', activation = 'relu')
        self.e3 = Conv2D(filters = 128, kernel_size = 3, strides = 2, padding='same', activation = 'relu')
        self.flatten = Flatten()

        self.z_bg_mu = Dense(bg_latent_size, activation=None)
        self.z_bg_sigma = Dense(bg_latent_size, activation='softplus')

        self.d1 = Dense(image_size[0]//8 * image_size[1]//8 * 128, activation='relu')
        self.d2 = Conv2D(filters = 128, kernel_size = 3, strides = 1, padding='same', activation='relu')
        self.d3 = Conv2D(filters = 64, kernel_size = 3, strides = 1, padding='same', activation='relu')
        self.d4 = Conv2D(filters = 32, kernel_size = 3, strides = 1, padding='same', activation='sigmoid')
        self.d5 = Conv2D(filters = self.num_channel, kernel_size = 3, strides = 1, padding='same', activation='sigmoid')

    def call(self,x):
        x = self.e3(self.e2(self.e1(x)))
        x = self.flatten(x)
        z_bg_mean = self.z_bg_mu(x)
        z_bg_sig = self.z_bg_sigma(x)

        z_bg = z_bg_mean + z_bg_sig * tf.random.normal(shape=tf.shape(z_bg_mean))

        x = self.d1(z_bg)
        x = tf.reshape(x,[-1,self.image_size[0]//8,self.image_size[1]//8,128]) # 512,4,4,128
        x = self.d2(x)
        x = tf.image.resize(x,[self.image_size[0]//4,self.image_size[1]//4]) #512,8,8,64
        x = self.d3(x)
        x = tf.image.resize(x,[self.image_size[0]//2,self.image_size[1]//2]) #512,16,16,32
        x = self.d4(x)
        x = tf.image.resize(x,[self.image_size[0],self.image_size[1]]) #512,32,32,2
        x = self.d5(x)
        
        return x, z_bg, z_bg_mean, z_bg_sig

class ObjEncoder(Layer):
    def __init__(self, latent_size, name='objEncoder',**kwargs):
        super(ObjEncoder, self).__init__(name=name, **kwargs)

        self.conv1 = Conv2D(filters = 32, kernel_size = 3, strides = 2, padding='same', activation = 'relu')
        self.conv2 = Conv2D(filters = 64, kernel_size = 3, strides = 2, padding='same', activation = 'relu')
        self.flatten = Flatten()
        self.dense1 = Dense(latent_size*2, activation='relu')
        self.z_what_mu = Dense(latent_size, activation=None)
        self.z_what_sigma = Dense(latent_size, activation='softplus')

    
    def call(self, inputs, training=False):


        input_shape = tf.shape(inputs)
        inputs = tf.reshape(inputs, [input_shape[0] * input_shape[1], input_shape[2], input_shape[3], input_shape[4]])
        x = self.conv1(inputs)
        x = self.conv2(x)
        x = self.flatten(x)
        h = self.dense1(x)
        z_what_mean = self.z_what_mu(h)
        z_what_sigma = self.z_what_sigma(h)

        epsilon = tf.random.normal(shape=tf.shape(z_what_sigma))
        z_what = z_what_mean + epsilon * z_what_sigma

        return z_what, z_what_mean, z_what_sigma

class ObjEncoderScramble(Layer):
    def __init__(self, latent_size, patch_size, local_latent_size, name='objEncoder',**kwargs):
        super(ObjEncoderScramble, self).__init__(name=name, **kwargs)
        self.patch_size = patch_size
        self.flatten = Flatten()

        self.conv1 = Conv2D(filters = 32, kernel_size = 3, strides = 2, padding='same', activation = 'relu')
        self.conv2 = Conv2D(filters = 64, kernel_size = 3, strides = 2, padding='same', activation = 'relu')
        self.dense1 = Dense(latent_size*2, activation='relu')
        self.z_what_mu = Dense(latent_size, activation=None)
        self.z_what_sigma = Dense(latent_size, activation='softplus')

        self.x_hat_conv1 = Conv2D(filters = 32, kernel_size = 3, strides = 2, padding='same', activation = 'relu')
        self.x_hat_conv2 = Conv2D(filters = 64, kernel_size = 3, strides = 2, padding='same', activation = 'relu')
        self.x_hat_dense1 = Dense(latent_size*2, activation='relu')
        self.z_l_mu = Dense(local_latent_size, activation=None)
        self.z_l_sigma = Dense(local_latent_size, activation='softplus')

    
    def call(self, inputs, training=False):

        print('patch_size:',self.patch_size)
        input_shape = tf.shape(inputs)
        inputs = tf.reshape(inputs, [input_shape[0] * input_shape[1], input_shape[2], input_shape[3], input_shape[4]])

        num_patches = input_shape[2] * input_shape[3] // (self.patch_size**2)
        print('num_patches:',num_patches)
        patches = tf.image.extract_patches(inputs,sizes=[1,self.patch_size,self.patch_size,1],strides=[1,self.patch_size,self.patch_size,1],rates=[1, 1, 1, 1],padding='VALID')
        print('patch shape:',patches.shape)
        patches = tf.reshape(patches, [input_shape[0] * input_shape[1], num_patches, self.patch_size, self.patch_size, input_shape[4]]) 
        rand_idx = tf.reshape(tf.random.shuffle(tf.range(0,num_patches)),[num_patches])
        patches = tf.gather(patches, rand_idx, axis=1) #same shape but random patch location
        

        rows = tf.split(patches, inputs.shape[2]//self.patch_size,axis=1) #list of  batch*num_glimpse,4,patch_size,patch_size,3
        rows = [tf.concat(tf.unstack(x,axis=1),axis=2) for x in rows]
        x_hat = tf.concat(rows,axis=1)
        x_hat = tf.convert_to_tensor(x_hat)
        print('x_hat shape:',x_hat.shape)


        x = self.conv1(inputs)
        x = self.conv2(x)
        x = self.flatten(x)
        x = self.dense1(x)
        z_what_mean = self.z_what_mu(x)
        z_what_sigma = self.z_what_sigma(x)

        epsilon = tf.random.normal(shape=tf.shape(z_what_sigma))
        z_what = z_what_mean + epsilon * z_what_sigma

        z = self.x_hat_conv1(x_hat)
        z = self.x_hat_conv2(z)
        z = self.flatten(z)
        z = self.x_hat_dense1(z)
        z_l_mean = self.z_l_mu(z)
        z_l_sigma = self.z_l_sigma(z)

        epsilon = tf.random.normal(shape=tf.shape(z_l_sigma))
        z_l = z_l_mean + epsilon * z_l_sigma

        x_hat = tf.reshape(patches,[input_shape[0], input_shape[1],input_shape[2], input_shape[3], input_shape[4]])

        return z_what, z_what_mean, z_what_sigma, z_l, z_l_mean, z_l_sigma, x_hat


class ObjDecoder(Layer):
    def __init__(self, object_size, num_channel,latent_size,
                 name='ObjDecoder',
                 **kwargs):
        super(ObjDecoder, self).__init__(name=name, **kwargs) 
        self.object_size = object_size
        self.num_channel = num_channel
        self.d0 = Dense(latent_size*2, activation='relu')
        self.d1 = Dense(object_size//4 * object_size//4 * 32, activation='relu')
        self.d2 = Conv2D(filters = 64, kernel_size = 3, strides = 1, padding='same', activation='relu')
        self.d3 = Conv2D(filters = 32, kernel_size = 3, strides = 1, padding='same', activation='relu')
        self.d5 = Conv2D(filters = self.num_channel+1, kernel_size = 3, strides = 1, padding='same', activation=None) # recon + alpha channel
        

    def call(self, inputs,training=False):
        x = self.d0(inputs)
        x = self.d1(x)
        x = tf.reshape(x,[-1,self.object_size//4,self.object_size//4,32]) # 512,4,4,128
        x = self.d2(x)
        x = tf.image.resize(x,[self.object_size//2,self.object_size//2]) #512,8,8,64
        x = self.d3(x)
        x = tf.image.resize(x,[self.object_size,self.object_size]) #512,32,32,2
        x = self.d5(x)
        x_recon_unnorm, x_alpha = tf.sigmoid(x[:,:,:,:self.num_channel]), tf.sigmoid(x[:,:,:,self.num_channel:])
        
        return x_recon_unnorm, x_alpha

class Encoder(Layer):
    def __init__(self, object_size, latent_size, tau, concat=False, glimpse_local=False, patch_size=None, local_latent_size=None, name='encoder', **kwargs):
        super(Encoder, self).__init__(name=name, **kwargs)


        self.tau = tau
        self.n_z_where = 4
        self.n_z_depth = 1
        self.n_z_pres = 1
        self.glimpse_local = glimpse_local
        self.latent_size = latent_size
        self.object_size = [object_size,object_size]

        self.conv1 = Conv2D(filters=128, kernel_size=4, strides=(2,2), padding='SAME', activation='relu')
        self.conv2 = Conv2D(filters=128, kernel_size=4, strides=(2,2), padding='SAME', activation='relu')
        self.conv3 = Conv2D(filters=128, kernel_size=4, strides=(3,3), padding='SAME', activation='relu')

        self.z1 = Conv2D(filters=128, kernel_size=1, strides=(1,1), padding='VALID', activation='relu')
        self.z2 = Conv2D(filters=128, kernel_size=1, strides=(1,1), padding='VALID', activation='relu')
        self.z3 = Conv2D(filters=100, kernel_size=1, strides=(1,1), padding='VALID', activation='relu')
        
        self.n_pass_through_features = 8

        self.dense_z_where = tf.keras.Sequential([Dense(128,activation='relu'),Dense(64,activation='relu'),Dense(self.n_z_where*2 + self.n_pass_through_features, activation=None)])

        self.dense_z_depth = tf.keras.Sequential([Dense(64,activation='relu'),Dense(self.n_z_depth*2 + self.n_pass_through_features, activation=None)])
        self.dense_z_pres = tf.keras.Sequential([Dense(64,activation='relu'),Dense(self.n_z_pres, activation=None)])

        self.g_sampling = Sampling()
        self.gumbel_sampling = GumbelSM_Sampling(tau=self.tau)
        self.stn = STN(name='encoder_stn', H_out=object_size, W_out=object_size)
        if glimpse_local:
            self.obj_encoder = ObjEncoderScramble(latent_size, patch_size, local_latent_size)
        else:
            self.obj_encoder = ObjEncoder(latent_size)
        if concat:
            self.dense_z_l = tf.keras.Sequential([Dense(16,activation='relu'),Dense(16,activation='relu')])

    def call(self, inputs, training=False):

        if isinstance(inputs,list):
            x, z_l = inputs
            z_l = self.dense_z_l(z_l)
            z_l = tf.tile(z_l[:,tf.newaxis,:],[1,16,1])
            z_l = tf.reshape(z_l,[-1,z_l.shape[-1]])
        else:
            x = inputs

        h = self.conv1(x)
        h = self.conv2(h)
        h = self.conv3(h)
        h = self.z1(h)
        h = self.z2(h)
        z = self.z3(h)



        if isinstance(inputs,list):
            features_vector = tf.concat([tf.reshape(z,[-1,z.shape[-1]]),z_l],axis=-1) #flatten to [Batch*cells_num, features]
        else:
            features_vector = tf.reshape(z,[-1,z.shape[-1]])
        # box network
        z_where_mean, z_where_sigma, features_1 = tf.split(self.dense_z_where(features_vector),[self.n_z_where, self.n_z_where, self.n_pass_through_features],axis=-1)
        
        z_where_sigma = tf.math.softplus(z_where_sigma  - tf.constant(1.0, shape=z_where_sigma.shape))
        z_where = self.g_sampling([z_where_mean, z_where_sigma])


        partial_program = z_where
        z_where = tf.reshape(z_where,[z.shape[0],z.shape[1],z.shape[2],self.n_z_where])


        features_1 = tf.nn.relu(features_1)

        #attr network
        all_glimpses, _ = self.stn([x,z_where])
        obj_output = self.obj_encoder(all_glimpses, training)
        if not self.glimpse_local:
            z_what, z_what_mean, z_what_sigma = obj_output
        else:
            z_what, z_what_mean, z_what_sigma, z_l, z_l_mean, z_l_sigma, x_hat = obj_output
            z_l = tf.reshape(z_l,[z.shape[0],z.shape[1],z.shape[2],-1])
            z_l_sigma = tf.reshape(z_l_sigma,[z.shape[0],z.shape[1],z.shape[2],-1])
            z_l_mean = tf.reshape(z_l_mean,[z.shape[0],z.shape[1],z.shape[2],-1])


        partial_program = tf.concat([partial_program,z_what],axis=1)
        layer_inp = tf.concat([features_vector,features_1,partial_program],axis=1)

        #z_depth_network
        z_depth_mean, z_depth_sigma, features_2 = tf.split(self.dense_z_depth(layer_inp),[self.n_z_depth, self.n_z_depth, self.n_pass_through_features],axis=-1) # self.dense_z_depth(layer_inp)
        z_depth_sigma = tf.math.softplus(z_depth_sigma)
        z_depth = self.g_sampling([z_depth_mean, z_depth_sigma])
        partial_program = tf.concat([partial_program,z_depth],axis=1)

        features_2 = tf.nn.relu(features_2)
        layer_inp = tf.concat([features_vector,features_2,partial_program],axis=1)

        # z_pres_network
        z_pres_logits = tf.clip_by_value(self.dense_z_pres(layer_inp),-10.,10.)
        z_pres_pre_sigmoid = concrete_binary_pre_sigmoid_sample(z_pres_logits,self.tau)
        z_pres = tf.sigmoid(z_pres_pre_sigmoid)

        # reshape
        z_what = tf.reshape(z_what,[z.shape[0],z.shape[1],z.shape[2],-1])
        z_what_sigma = tf.reshape(z_what_sigma,[z.shape[0],z.shape[1],z.shape[2],-1])
        z_what_mean = tf.reshape(z_what_mean,[z.shape[0],z.shape[1],z.shape[2],-1])

        z_where = tf.reshape(z_where,[z.shape[0],z.shape[1],z.shape[2],-1])
        z_where_sigma = tf.reshape(z_where_sigma,[z.shape[0],z.shape[1],z.shape[2],-1])
        z_where_mean = tf.reshape(z_where_mean,[z.shape[0],z.shape[1],z.shape[2],-1])

        z_depth = tf.reshape(z_depth,[z.shape[0],z.shape[1],z.shape[2],-1])
        z_depth_sigma = tf.reshape(z_depth_sigma,[z.shape[0],z.shape[1],z.shape[2],-1])
        z_depth_mean = tf.reshape(z_depth_mean,[z.shape[0],z.shape[1],z.shape[2],-1])

        z_pres = tf.reshape(z_pres,[z.shape[0],z.shape[1],z.shape[2],-1])
        z_pres_logits = tf.reshape(z_pres_logits,[z.shape[0],z.shape[1],z.shape[2],-1])
        z_pres_pre_sigmoid = tf.reshape(z_pres_pre_sigmoid,[z.shape[0],z.shape[1],z.shape[2],-1])
        

        if self.glimpse_local:
            
            return (z_what, z_what_mean, z_what_sigma, z_where, z_where_mean, z_where_sigma,
                z_depth, z_depth_mean, z_depth_sigma, z_pres, z_pres_logits, z_pres_pre_sigmoid, all_glimpses,
                z_l, z_l_mean, z_l_sigma, x_hat)
        else:
            return (z_what, z_what_mean, z_what_sigma, z_where, z_where_mean, z_where_sigma,
                z_depth, z_depth_mean, z_depth_sigma, z_pres, z_pres_logits, z_pres_pre_sigmoid, all_glimpses)



class Decoder(Layer):
    def __init__(self, image_size, test_size, object_size, latent_size, name='decoder', **kwargs):
        super(Decoder, self).__init__(name=name, **kwargs)
        self.object_size = object_size
        self.num_channel = image_size[2]

        self.obj_decoder = ObjDecoder(object_size, self.num_channel, latent_size)
        self.stn = STN(name='decoder_stn', inverse=True, H_out=image_size[0], W_out=image_size[1])
        # self.stn_test = STN(name='decoder_stn_test', inverse=True, H_out=test_size[0], W_out=test_size[1])
        self.renderer = Renderer(self.num_channel)

        # print('Decoder channel:',self.num_channel)


    def call(self, inputs,training=False):

        z_what, z_where, z_depth, z_pres, z_pres_logits, output_shape = inputs

        obj_recon_unnorm, obj_recon_alpha = self.obj_decoder(z_what,training)
        # print('Object recon shape:',obj_recon_unnorm.shape)

        H_obj = tf.shape(z_where)[1]
        W_obj = tf.shape(z_where)[2]

        obj_recon_unnorm = tf.reshape(obj_recon_unnorm, [-1, H_obj*W_obj, self.object_size, self.object_size, self.num_channel]) # H_obj*W_obj = num cells
        obj_recon_alpha  = tf.reshape(obj_recon_alpha, [-1, H_obj*W_obj, self.object_size, self.object_size, 1])

        concat_recon_alpha = tf.concat([obj_recon_unnorm, obj_recon_alpha], axis=4) # STACK Channel

        obj_full_recon_unnorm, obj_bbox_mask = self.stn([concat_recon_alpha, z_where]) 


        return obj_recon_unnorm, obj_recon_alpha, obj_full_recon_unnorm, obj_bbox_mask

class Renderer(Layer):
    def __init__(self, num_channel,name='Renderer',**kwargs):

        super(Renderer, self).__init__(name=name, **kwargs)
        self.num_channel = num_channel
        self.g_noise = GaussianNoise(0.01)
        return

    def call(self,inputs,training=False):
        obj_full_recon_unnorm = inputs[0] #[B, 256, 128, 128, C+1]

        background_img = inputs[1]
        z_depth = inputs[2] #[B, 16, 16, 1] 
        z_pres = inputs[3] #[B, 16, 16, 1]
        if not training:
            z_pres = tf.nn.sigmoid(inputs[4]) # use z_pres_logits during test
        
        z_shape = tf.shape(z_depth)
        B = z_shape[0]
        Bp = z_shape[1] * z_shape[2]

        z_depth = tf.reshape(z_depth, [B, Bp, 1, 1, 1])
        z_pres = tf.reshape(z_pres, [B, Bp, 1, 1, 1]) # [:, :, :, :, :1]  #[B, 256, 1, 1, 1]
        if not training:
            z_pres = tf.maximum(tf.round(z_pres),tf.constant(1e-8,shape=z_pres.shape))

        obj_recon_image = obj_full_recon_unnorm[:, :, :, :, :self.num_channel]
        obj_recon_alpha = tf.clip_by_value(obj_full_recon_unnorm[:, :, :, :, self.num_channel:],1e-8,1.)


        transparency_map = z_pres * obj_recon_alpha # [B, 256, 1, 1, 1] * [B, 256, 128, 128, 1] # per pixel density
        importance_map = z_pres * obj_recon_alpha * (tf.nn.sigmoid( - z_depth) + 0.5) #[B, 256, 1, 1, 1] 

        obj_recon_image = tf.clip_by_value(self.g_noise(obj_recon_image,training), 0.0, 1.0)

        unnorm_canvas = tf.reduce_sum(importance_map * obj_recon_image, axis=1)# [B, 256, 128, 128, C] ->[B, 128, 128, C]
        normalise_const = tf.reduce_sum(importance_map, axis=1) # [B, 256, 1, 1, 1] * [B, 256, 1, 1, 1] * [B, 256, 128, 128, 1] -> [B,128,128,C]
        normalised_canvas =  unnorm_canvas / (normalise_const + 1e-8) # no Bp

        normalised_alpha_canvas = tf.reduce_sum(transparency_map * importance_map, axis=1) / (normalise_const + 1e-8)

        canvas_with_bg = normalised_alpha_canvas * normalised_canvas + (1. - normalised_alpha_canvas) * background_img        



        return canvas_with_bg


if __name__ == "__main__":

    numpy_data = np.load('multimnist_6_128_overlap_dummy.npy')
    
    dataset = tf.data.Dataset.from_tensor_slices(numpy_data).shuffle(10000).batch(3)
    
    model = SPAIR()

    # for i in dataset:
    #     x_recon, z_what, z_where, z_depth, z_pres = model(i)
    #     break
        
