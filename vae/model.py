import tensorflow as tf
import tensorflow.keras.layers as layers
from tensorflow.keras.layers import Dense, Conv2D, BatchNormalization, Flatten, Dropout
from tensorflow.keras import Model
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


class Sampling(layers.Layer):
    def call(self, inputs):
        z_mean, z_sig = inputs
        epsilon = tf.random.normal(shape=z_sig.shape, mean=0, stddev=1, dtype=tf.float32)
        return z_mean + z_sig * epsilon


class Encoder(layers.Layer):
    def __init__(self, latent_dims = 32, variational = True, type = 'conv', y_size=None, tau=None):
        super(Encoder, self).__init__()
        self.variational = variational
        self.latent_dims = latent_dims
        self.flatten = Flatten()
        self.tau = tau
        if type=='fc':
            self.e1 = Dense(1024, activation = 'relu')
            self.e2 = Dense(512, activation = 'relu')
            if variational:
                self.e3_mean = Dense(latent_dims, activation = None)
                self.e3_sd = Dense(latent_dims, activation = None)
                self.sampling = Sampling()
            else:
                self.e3 = Dense(latent_dims, activation = 'relu')
            self.call = self.call_fc

        elif type == 'conv': #always variational
            if self.variational:
                self.e1 = Conv2D(filters = 32, kernel_size = 6, strides = 2, padding='same', activation = 'relu')
                self.e2 = Conv2D(filters = 64, kernel_size = 6, strides = 2, padding='same', activation = 'relu')
                self.e3 = Conv2D(filters = 128, kernel_size = 4, strides = 2, padding='same', activation = 'relu')


                self.e4_mean = Dense(latent_dims, activation = None)
                self.e4_sd = Dense(latent_dims, activation = 'softplus') #, kernel_initializer=tf.initializers.TruncatedNormal(),bias_initializer=tf.keras.initializers.constant(-1)
                self.sampling = Sampling()

                self.call = self.call_conv
            else:
                raise NotImplemented('Deterministic convolution autoencoder not implemented')
        elif type == 'gmvae':
            self.h_block = tf.keras.Sequential([
                Conv2D(filters = 128, kernel_size = 6, strides = 2, padding='same', activation = 'elu'),
                Conv2D(filters = 128, kernel_size = 6, strides = 2, padding='same', activation = 'elu'),
                Conv2D(filters = 128, kernel_size = 4, strides = 2, padding='same', activation = 'elu')])

            self.y_block = tf.keras.Sequential([
                Dense(1024, activation='elu'),
                Dropout(rate=0.2),
                Dense(128, activation='elu'),
                ])
            self.do1 = Dropout(rate=0.2)
            self.y_dense = Dense(y_size, activation=None, name='y_dense')

            # Z prior block
            self.do2 = Dropout(rate=0.2)
            self.h_top_dense = Dense(512, activation='elu')
            self.do3 = Dropout(rate=0.2)
            self.z_prior_mean = Dense(latent_dims, activation=None, name='z_prior_mean')
            self.do4 = Dropout(rate=0.2)
            self.z_prior_sig = Dense(latent_dims, activation='softplus', name='z_prior_sig', bias_initializer=tf.keras.initializers.constant(1)) # kernel_initializer=tf.initializers.TruncatedNormal(),bias_initializer=tf.keras.initializers.constant(1))

            #Encoder block
            self.do5 = Dropout(rate=0.2)
            self.e1 = Dense(512,activation='elu')
            self.do6 = Dropout(rate=0.1)
            self.z_mean = Dense(latent_dims,activation=None)
            self.do7 = Dropout(rate=0.1)
            self.z_sig = Dense(latent_dims,activation='softplus', bias_initializer=tf.keras.initializers.constant(1)) #kernel_initializer=tf.initializers.TruncatedNormal(),bias_initializer=tf.keras.initializers.constant(1))
            self.sampling = Sampling()

            self.call = self.call_gmvae
        else:
            raise Exception('Type undefined')



    def call_fc(self, x, training=False):
        # x = tf.reshape(x, [-1,28*28])
        x = self.flatten(x)
        # print(x.shape)
        x = self.e1(x)
        x = self.e2(x)
        if self.variational:
            z_mean = self.e3_mean(x)
            z_log_var = self.e3_sd(x)
            z = self.sampling((z_mean,z_log_var))
            return z, z_mean, z_log_var
        else:
            z = self.e3(x)
            return z

    def call_conv(self, x, training=False):
        # x = tf.reshape(x, [-1,28,28,1])
        # print('input shape:',x.shape)
        x = self.e1(x)
        x = self.e2(x)
        x = self.e3(x)

        # precode_size = np.prod(x.shape[1:])
        # print('Precode shape:',x.shape)
        # print('Precode size:',precode_size)
        x = self.flatten(x)
        z_mean = self.e4_mean(x)
        z_sig = self.e4_sd(x)
        z = self.sampling((z_mean,z_sig))
        return z, z_mean, z_sig

    def call_gmvae(self,x, training=False):
        h = self.h_block(x)
        h = self.flatten(h)
        # y block
        y_hidden = self.y_block(h)
        y_logits = self.y_dense(y_hidden)
        noise = tf.random.uniform(shape = y_logits.shape)
        y = tf.nn.softmax( (y_logits - tf.math.log(-tf.math.log(noise))) / self.tau, axis=1) #gumbel softmax

        z_prior_mean = self.z_prior_mean(y)
        z_prior_sig = self.z_prior_sig(y)

        h_top = self.h_top_dense(y)
        h = self.e1(self.do5(h,training))
        h = h + h_top
        z_mean = self.z_mean(h)
        z_sig = self.z_sig(h)
        z = self.sampling((z_mean,z_sig))

        return z, z_mean, z_sig, y, y_logits, z_prior_mean, z_prior_sig

    def encode_y(self,y):
        z_prior_mean = self.z_prior_mean(y)
        z_prior_sig = self.z_prior_sig(y)
        return z_prior_mean, z_prior_sig




class Decoder(layers.Layer):
    def __init__(self, latent_dims = 32, image_shape = None, type='conv'):
        super(Decoder, self).__init__()
        self.latent_dims = latent_dims
        self.image_shape = image_shape
        self.type = type

        self.d1 = Dense(self.image_shape[1]//8*self.image_shape[2]//8*128, activation = 'relu')
        self.d2 = Conv2D(filters = 128, kernel_size = 4, strides = 1, padding='same', activation='relu')
        self.d3 = Conv2D(filters = 64, kernel_size = 4, strides = 1, padding='same', activation='relu')
        self.d4 = Conv2D(filters = 32, kernel_size = 6, strides = 1, padding='same', activation='relu')
        self.d5 = Conv2D(filters = 6, kernel_size = 6, strides = 1, padding='same', activation=None)

    def call(self, x):
        # print('Code shape:',x.shape)
        x = self.d1(x)
        x = tf.reshape(x,[-1,self.image_shape[1]//8,self.image_shape[2]//8,128])
        x = self.d2(x)
        x = tf.image.resize(x,[self.image_shape[1]//4,self.image_shape[2]//4])
        x = self.d3(x)
        x = tf.image.resize(x,[self.image_shape[1]//2,self.image_shape[2]//2])
        x = self.d4(x)
        x = tf.image.resize(x,[self.image_shape[1],self.image_shape[2]])
        x = self.d5(x)
        return x[:,:,:,:3], x[:,:,:,3:] #x_mean, x_log_scale




class LGVae(Model):
    def __init__(self, global_latent_dims, local_latent_dims, image_shape = None,  variational = True, type = 'conv'):
        super(LGVae, self).__init__()
        # self.latent_dims = latent_dims
        self.global_latent_dims = global_latent_dims
        self.local_latent_dims = local_latent_dims
        self.variational = variational
        self.image_shape = image_shape
        self.encoder_x = Encoder(latent_dims = global_latent_dims)
        self.encoder_x_hat = Encoder(latent_dims = local_latent_dims)

        self.decoder_x = Decoder(latent_dims = global_latent_dims + local_latent_dims, image_shape = image_shape) # use both z_g and z_l
        self.decoder_x_hat = Decoder(latent_dims = local_latent_dims, image_shape = image_shape)

    
    def call(self, inputs):
        x, x_hat = inputs[:,:,:,:3], inputs[:,:,:,3:]
        # print(x.shape)
        # print(x_hat.shape)
        if self.variational:
            z_x, z_mean_x, z_sig_x = self.encoder_x(x)
            z_x_hat, z_mean_x_hat, z_sig_x_hat = self.encoder_x_hat(x_hat)

            x_mean, x_log_scale = self.decoder_x(tf.concat([z_x,z_x_hat],axis=1))
            x_hat_mean, x_hat_log_scale = self.decoder_x_hat(z_x_hat)

            return x_mean, x_log_scale, z_x, z_mean_x, z_sig_x, z_x_hat, x_hat_mean, x_hat_log_scale, z_mean_x_hat, z_sig_x_hat
        #else:
        raise NotImplementedError('Determiistic LG-AE not implemented')

    def encode(self, inputs):
        x, x_hat = inputs[:,:,:,:3], inputs[:,:,:,3:]
        if self.variational:
            z_x, z_mean_x, z_sig_x = self.encoder_x(x)
            z_x_hat, z_mean_x_hat, z_sig_x_hat = self.encoder_x_hat(x_hat)
            return z_x, z_x_hat

    def decode(self, z_x, z_x_hat, rescale = True):
        x_mean, x_log_scale = self.decoder_x(tf.concat([z_x,z_x_hat],axis=1))
        x_hat_mean, x_hat_log_scale = self.decoder_x_hat(z_x_hat)
        if rescale:
            x_recon = tf.clip_by_value((x_mean + 1)*0.5,0.,1.)
            x_hat_recon = tf.clip_by_value((x_hat_mean + 1)*0.5,0.,1.)
            return x_recon, x_hat_recon
        return x_mean, x_hat_mean


class LGGMVae(Model):
    def __init__(self, global_latent_dims, local_latent_dims, image_shape, y_size, tau, variational = True, type = 'conv'):
        super(LGGMVae, self).__init__()
        # self.latent_dims = latent_dims
        self.global_latent_dims = global_latent_dims
        self.local_latent_dims = local_latent_dims
        self.variational = variational
        self.image_shape = image_shape
        self.y_size = y_size
        self.encoder_x = Encoder(latent_dims = global_latent_dims, type='gmvae', y_size=y_size, tau=tau)
        self.encoder_x_hat = Encoder(latent_dims = local_latent_dims)

        self.decoder_x = Decoder(latent_dims = global_latent_dims + local_latent_dims, image_shape = image_shape) # use both z_g and z_l
        self.decoder_x_hat = Decoder(latent_dims = local_latent_dims, image_shape = image_shape)

    
    def call(self, inputs, training=False):
        x, x_hat = inputs[:,:,:,:3], inputs[:,:,:,3:]
        # print(x.shape)
        # print(x_hat.shape)
        if self.variational:
            z_x, z_mean_x, z_sig_x, y, y_logits, z_prior_mean, z_prior_sig = self.encoder_x(x)
            z_x_hat, z_mean_x_hat, z_sig_x_hat = self.encoder_x_hat(x_hat)

            x_mean, x_log_scale = self.decoder_x(tf.concat([z_x,z_x_hat],axis=1))
            x_hat_mean, x_hat_log_scale = self.decoder_x_hat(z_x_hat)

            return x_mean, x_log_scale, z_x, z_mean_x, z_sig_x, z_x_hat, x_hat_mean, x_hat_log_scale, z_mean_x_hat, z_sig_x_hat, y, y_logits, z_prior_mean, z_prior_sig
        #else:
        raise NotImplementedError('Determiistic LG-AE not implemented')

    def encode(self, inputs):
        x, x_hat = inputs[:,:,:,:3], inputs[:,:,:,3:]
        if self.variational:
            z_x, z_mean_x, z_sig_x, y, y_logits, z_prior_mean, z_prior_sig = self.encoder_x(x)
            z_x_hat, z_mean_x_hat, z_sig_x_hat = self.encoder_x_hat(x_hat)
            return z_x, z_x_hat

    def decode(self, z_x, z_x_hat, rescale = True):
        x_mean, x_log_scale = self.decoder_x(tf.concat([z_x,z_x_hat],axis=1))
        x_hat_mean, x_hat_log_scale = self.decoder_x_hat(z_x_hat)
        if rescale:
            x_recon = tf.clip_by_value((x_mean + 1)*0.5,0.,1.)
            x_hat_recon = tf.clip_by_value((x_hat_mean + 1)*0.5,0.,1.)
            return x_recon, x_hat_recon
        return x_mean, x_hat_mean

    def encode_y(self, y, rescale = True):
        z_prior_mean, z_prior_sig = self.encoder_x.encode_y(y)
        return z_prior_mean, z_prior_sig

    def get_y(self,x):
        z_x, z_mean_x, z_sig_x, y, y_logits, z_prior_mean, z_prior_sig = self.encoder_x(x)
        # print('y_logits.shape:', y_logits.shape)
        return y, y_logits

class GMVae(Model):
    def __init__(self, global_latent_dims, image_shape, y_size, tau, variational = True, type = 'conv'):
        super(GMVae, self).__init__()
        # self.latent_dims = latent_dims
        self.global_latent_dims = global_latent_dims
        self.variational = variational
        self.image_shape = image_shape
        self.y_size = y_size
        self.encoder_x = Encoder(latent_dims = global_latent_dims, type='gmvae', y_size=y_size, tau=tau)
        self.decoder_x = Decoder(latent_dims = global_latent_dims, image_shape = image_shape) # use both z_g and z_l
    
    def call(self, inputs,training=False):
        x = inputs[:,:,:,:3]
        # print(x.shape)
        # print(x_hat.shape)
        if self.variational:
            z_x, z_mean_x, z_sig_x, y, y_logits, z_prior_mean, z_prior_sig = self.encoder_x(x)

            x_mean, x_log_scale = self.decoder_x(z_x)

            return x_mean, x_log_scale, z_x, z_mean_x, z_sig_x, y, y_logits, z_prior_mean, z_prior_sig
        #else:
        raise NotImplementedError('Determiistic LG-AE not implemented')

    def encode(self, inputs):
        x= inputs[:,:,:,:3]
        if self.variational:
            z_x, z_mean_x, z_sig_x, y, y_logits, z_prior_mean, z_prior_sig = self.encoder_x(x)
            return z_x

    def decode(self, z_x, rescale = True):
        x_mean, x_log_scale = self.decoder_x(z_x)
        if rescale:
            x_recon = tf.clip_by_value((x_mean + 1)*0.5,0.,1.)
            return x_recon
        return x_mean

    def encode_y(self, y, rescale = True):
        z_prior_mean, z_prior_sig = self.encoder_x.encode_y(y)
        return z_prior_mean, z_prior_sig

    def get_y(self,x):
        z_x, z_mean_x, z_sig_x, y, y_logits, z_prior_mean, z_prior_sig = self.encoder_x(x)
        return y, y_logits




class Classifier(Model):
    def __init__(self, latent_dims = 256, target_shape = None):
        super(Classifier, self).__init__()
        self.bn1 = BatchNormalization()
        self.e1 = Conv2D(filters = 32, kernel_size = 6, strides = 2, padding='same', activation = 'relu')
        self.bn2 = BatchNormalization()
        self.e2 = Conv2D(filters = 64, kernel_size = 6, strides = 2, padding='same', activation = 'relu')
        self.bn3 = BatchNormalization()
        self.e3 = Conv2D(filters = 128, kernel_size = 4, strides = 2, padding='same', activation = 'relu')
        self.bn3 = BatchNormalization()
        self.e3 = Conv2D(filters = 256, kernel_size = 4, strides = 2, padding='same', activation = 'relu')
        self.flatten = Flatten()
        self.d1 = Dropout(0.25)
        self.e4 = Dense(latent_dims, activation = 'relu')
        self.d2= Dropout(0.25)
        self.e5 = Dense(latent_dims//4, activation = 'relu')
        self.d3 = Dropout(0.25)
        self.e6 = Dense(target_shape, activation = None)
    
    def call(self,x, training=False):
        x = self.e1(self.bn1(x,training))
        x = self.e2(self.bn2(x,training))
        x = self.e3(self.bn3(x,training))
        x = self.e4(self.d1(self.flatten(x),training))
        x = self.e5(self.d2(x,training))
        x = self.e6(self.d3(x,training))
        # print('output shape:',x.shape)
        return x