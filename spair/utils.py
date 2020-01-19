import tensorflow as tf
import numpy as np

from tensorflow.keras.layers import Layer, Dense
# import cv2

class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

# https://github.com/e2crawfo/auto_yolo/blob/d7a070549999d42566db66f6c25b88e20730fd27/auto_yolo/models/core.py#L37
def concrete_binary_pre_sigmoid_sample(log_odds, temperature, eps=1e-8):
    u = tf.random.uniform(tf.shape(log_odds), minval=0, maxval=1)
    noise = tf.math.log(u + eps) - tf.math.log(1.0 - u + eps)
    return (log_odds + noise) / temperature

class Sampling(Layer):
    def call(self, inputs):
        z_mean, z_sigma = inputs
        epsilon = tf.random.normal(shape=tf.shape(z_sigma))
        
        return z_mean + z_sigma * epsilon




class GumbelSM_Sampling(Layer):
    def __init__(self, name='GumbelSM', tau=0.4, **kwargs):
        super(GumbelSM_Sampling, self).__init__(name=name, **kwargs)
        self.tau = tau

    def softmax(self, logits, axis):
            x = logits/self.tau
            b = tf.reduce_max(x, axis=axis, keepdims=True)
            exp_logits = tf.exp(x-b)
            return exp_logits / tf.reduce_sum(exp_logits, axis, keepdims=True)

    def call(self, inputs):
        # beta_logits = inputs

        G = - tf.math.log(-tf.math.log( tf.random.uniform(shape=tf.shape(inputs), minval=0, maxval=1) ))
        return self.softmax(inputs + G, axis=-1)


class STN(Layer):
    """
    Adapted from https://github.com/kevinzakka/spatial-transformer-network/blob/master/stn/transformer.py
    """
    def __init__(self, name='STN', inverse=False, H_out=32, W_out=32, **kwargs):
        super(STN, self).__init__(name=name, 
                                  **kwargs)
         
        self.inverse = inverse
        self.H_img = H_out
        self.W_img = W_out
        """
        if self.inverse:
            self.H_img, self.W_img = H_img, W_img = IMAGE_SIZE_Y, IMAGE_SIZE_X
        else:
            self.H_img, self.W_img = H_img, W_img = GRID_SIZE_Y, GRID_SIZE_X # the number of points in grid 
		"""

    def build(self, input_shape):
        """
        This function create the Grid points with size equal to the output size.
        Also, this function generates lists for (tx, ty) which specify the origin of each SPAIR object. 

        """


        z_where_shape = input_shape[1]
        
        H_obj, W_obj = z_where_shape[1], z_where_shape[2]

        H_img = self.H_img
        W_img = self.W_img
        
        # normal grid
        x = np.linspace(-1.0, 1.0, W_img)
        y = np.linspace(-1.0, 1.0, H_img)
        X, Y = np.meshgrid(x, y) 
        x_grids = tf.convert_to_tensor(X, dtype=tf.float32) #[W_img, H_img]
        y_grids = tf.convert_to_tensor(Y, dtype=tf.float32) 

        x_grids = tf.expand_dims( tf.reshape(x_grids, [-1]), axis=0)
        y_grids = tf.expand_dims( tf.reshape(y_grids, [-1]), axis=0) #[1, H_img*W_img]

        x_grids = tf.tile(x_grids, [H_obj*W_obj, 1])
        y_grids = tf.tile(y_grids, [H_obj*W_obj, 1]) #[B', H_img*W_img]

        ones = tf.ones_like(x_grids) # [B', H_img*W_img]

        self.sampling_grids = tf.expand_dims( tf.stack([x_grids, y_grids, ones], axis=1), axis=0) #[1, B', 3, H_img*W_img]
        
        self.Bp = H_obj*W_obj 

        bias_tx = np.zeros([H_obj, W_obj])
        bias_ty = np.zeros([H_obj, W_obj])
        
        self.cell_width_ratio = (2.0 * 12) / 48  # HACK todo
        self.cell_height_ratio = (2.0 * 12) / 48

        for i in range(H_obj):
            i_p = (2.-self.cell_height_ratio)*i/(H_obj-1) - (1-0.5*self.cell_height_ratio) #put bias in the middle of the cell
            #i_p = 2.*i/(H_obj - 1.) - 1.
            for j in range(W_obj):
                j_p = (2.-self.cell_width_ratio)*j/(W_obj-1) - (1-0.5*self.cell_width_ratio) 
                #j_p = 2.*j/(W_obj - 1.) - 1. 

                bias_ty[i, j] = i_p
                bias_tx[i, j] = j_p

        self.bias_tx = tf.expand_dims(tf.convert_to_tensor(bias_tx, dtype=tf.float32), axis=0)#[1, H_obj, W_obj]
        self.bias_ty = tf.expand_dims(tf.convert_to_tensor(bias_ty, dtype=tf.float32), axis=0)#[1, H_obj, W_obj] 
        

    def call(self, inputs):

        """
        This function:
        1. take the grid and do affine transformation according to the parameters sx, sy, tx, ty
        2. perform bilinear sampling using the grid
        """

        x, z_where = inputs
        # z_where :: [batch, H, W, 4]
        # z_where[0,0,0,:] :: x_offset, y_offset, box_width, box_height
        # transform this parameterisation into sx, sy, tx, ty for STN

        

        shape = tf.shape(z_where)
        B = shape[0]
        Bp = self.Bp

        H_img = self.H_img
        W_img = self.W_img

        
        sx = 0.5 * tf.nn.sigmoid( z_where[:, :, :, 0] ) # 0 < sx < s_max / avoid reflection
        sy = 0.5 * tf.nn.sigmoid( z_where[:, :, :, 1] )  # 0 < sy < s_max / 
        tx = 0.5 * tf.nn.tanh( z_where[:, :, :, 2] ) + self.bias_tx # offset limit ?
        ty = 0.5 * tf.nn.tanh( z_where[:, :, :, 3] ) + self.bias_ty # [B, H_obj, W_obj] 

        box_height = sy / tf.constant(2.0)
        box_height = box_height[:,:,:,tf.newaxis]
        box_width = sx / tf.constant(2.0)
        box_width = box_width[:,:,:,tf.newaxis]
        bbox_ty = (ty[:,:,:,tf.newaxis] + tf.constant(1.0)) / 2.0
        bbox_tx = (tx[:,:,:,tf.newaxis] + tf.constant(1.0)) / 2.0
        obj_bbox_mask = tf.concat([bbox_ty-box_height, bbox_tx-box_width, bbox_ty+box_height, bbox_tx+box_width],axis=-1) # [B,4,4,4]
        # print('obj_bbox_mask.shape:',obj_bbox_mask.shape)
        obj_bbox_mask = tf.reshape(obj_bbox_mask, [obj_bbox_mask.shape[0],obj_bbox_mask.shape[1]*obj_bbox_mask.shape[2],obj_bbox_mask.shape[3]]) #[B,B',4]
        # print('obj_bbox_mask.shape:',obj_bbox_mask.shape)

        if self.inverse:
            tx = -tx / (sx + 1e-5)
            ty = -ty / (sy + 1e-5)
            sx = 1/(sx + 1e-5)
            sy = 1/(sy + 1e-5)

        # theta_in = tf.transpose(tf.stack([sx,sy,tx,ty]),[1,2,3,0]) # [4,B,cell_y,cell_x] -> [B,cell_y,cell_x,4]
        # theta_in = tf.reshape(theta_in,[-1,4]) #flatten
        # theta = self.dense(theta_in)

        # sx = theta[:,0]
        # sy = theta[:,1]
        # tx = theta[:,2]
        # ty = theta[:,3]

        # sx = 0.5 * tf.nn.sigmoid(sx)
        # sy = 0.5 * tf.nn.sigmoid(sy)
        # tx = tf.nn.tanh(tx) + self.bias_tx
        # ty = tf.nn.tanh(ty) + self.bias_ty

        
        #(x,y) = A * (x,y,1)
        sx = tf.reshape(sx, [B, Bp]) #from [B, H_obj, W_obj] --> [B, B']
        sy = tf.reshape(sy, [B, Bp]) #
        tx = tf.reshape(tx, [B, Bp])
        ty = tf.reshape(ty, [B, Bp])
        zeros = tf.zeros_like(sx)


        A_top = tf.stack([sx, zeros, tx], axis=2)  # [B, B', 3]
        A_bottom = tf.stack([zeros, sy, ty], axis=2) # [B, B', 3]
        A = tf.stack([A_top, A_bottom], axis=2) # [B, B', 2, 3]

        sampling_grids = tf.tile(self.sampling_grids, [B, 1, 1, 1]) #[B, B', 3, H_img*W_img]
       
        batch_grids = tf.matmul(A, sampling_grids) #[B, B', 2, H_img*W_img]
        
        batch_grids = tf.reshape(batch_grids, [B, Bp, 2, H_img, W_img])


        outputs = self.bilinear_sampler(x, batch_grids)  #[B, B', H_img, W_img, C]

        return outputs, obj_bbox_mask

    @tf.function
    def bilinear_sampler(self, img, batch_grids):
        """
        At each point in the Grid, 
        we map that into the position in the original image.
        We gather 4 corner pixel of the gridpoint and do bilinear interpolation. 
        """
        if self.inverse:
            input_shape = tf.shape(img) # [B, B', H, W, C]
            H_x, W_x = input_shape[2], input_shape[3]
        else:
            input_shape = tf.shape(img) # [B, H, W, C]
            H_x, W_x = input_shape[1], input_shape[2]


        x = batch_grids[:, :, 0, :, :] # [B, B', H, W]
        y = batch_grids[:, :, 1, :, :] # [B, B', H, W]


        W_xf = tf.cast(W_x, tf.float32)
        H_xf = tf.cast(H_x, tf.float32)

        # rescale x,y with Ht and Wt
        x = 0.5 * (x+1.0) * (W_xf - 1) #* tf.cast(W_x - 1, tf.float32) # [0, W-1]
        y = 0.5 * (y+1.0) * (H_xf - 1) #tf.cast(H_x - 1, tf.float32) # [0, H-1]
        
        # grab four nearest corner 
        x0 = tf.floor(x)
        x1 = x0 + 1
        y0 = tf.floor(y)
        y1 = y0 + 1

        
        # clip to range [0, H/W] to not violate img boundaries
        x0 = tf.clip_by_value(x0, 0., W_xf-1)
        x1 = tf.clip_by_value(x1, 0., W_xf-1)
        y0 = tf.clip_by_value(y0, 0., H_xf-1)
        y1 = tf.clip_by_value(y1, 0., H_xf-1)


        # calculate deltas
        wa = (x1-x) * (y1-y)
        wb = (x1-x) * (y-y0)
        wc = (x-x0) * (y1-y)
        wd = (x-x0) * (y-y0)
        # delta has to be zero if the pixel violate boundaries


        #recast as int for index calculation
        x0 = tf.cast(x0, 'int32')
        x1 = tf.cast(x1, 'int32')
        y0 = tf.cast(y0, 'int32')
        y1 = tf.cast(y1, 'int32')


        # get pixel value at corner coords
        Ia = self.get_pixel_value(img, x0, y0) # (B, B', H_img, W_img, C)
        Ib = self.get_pixel_value(img, x0, y1)
        Ic = self.get_pixel_value(img, x1, y0)
        Id = self.get_pixel_value(img, x1, y1)

        # add dimension for addition
        wa = tf.expand_dims(wa, axis=4)
        wb = tf.expand_dims(wb, axis=4)
        wc = tf.expand_dims(wc, axis=4)
        wd = tf.expand_dims(wd, axis=4)

        # compute output
        out = tf.add_n([wa*Ia, wb*Ib, wc*Ic, wd*Id])

        return  out
    
    @tf.function
    def get_pixel_value(self, img, x, y):
        """
        Gathers the pixel from original image according to coordinate x,y
        """

        if self.inverse:
            """
               img: (B, B', H_x, W_x, C)
               x: (B, B', H_img, W_img)
               y: (B, B', H_img, W_img)

               indices: (B, B', H_img, W_img, 4)
            """ 

            x_shape = tf.shape(x)
            B = x_shape[0]
            Bp = x_shape[1]
            H_img = x_shape[2]
            W_img = x_shape[3]

            # assert Bp == tf.shape(img)[1]


            B_idx = tf.range(0, B)
            B_idx = tf.reshape(B_idx, (B, 1, 1, 1))
            B_idx = tf.tile(B_idx, (1, Bp, H_img, W_img))

            Bp_idx = tf.range(0, Bp)
            Bp_idx = tf.reshape(Bp_idx, (1, Bp, 1, 1))
            Bp_idx = tf.tile(Bp_idx, (B, 1, H_img, W_img))

            indices = tf.stack([B_idx, Bp_idx, y, x], axis=4)
            
            return tf.gather_nd(img, indices)

        else:
            """
             img: (B, H_x, W_x, C)
             x: (B, B', H_img, W_img)
             y: (B, B', H_img, W_img)

             indices : (B, B', H_img, W_img, 3) -> out: (B, B', H_img, W_img, C)
            """
            # assert len(tf.shape(img)) == 4

            x_shape = tf.shape(x)
            B = x_shape[0]
            Bp = x_shape[1]
            H_img = x_shape[2]
            W_img = x_shape[3]

            B_idx = tf.range(0, B)
            B_idx = tf.reshape(B_idx, (B, 1, 1, 1))
            B_idx = tf.tile(B_idx, (1, Bp, H_img, W_img))
            indices = tf.stack([B_idx, y, x], axis=4) #[B, B', H_img, W_img, 3]
            return tf.gather_nd(img, indices)