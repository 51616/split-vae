
import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
import time
import matplotlib.pyplot as plt
from datetime import datetime
import gc
import os
import visualizer
from spair import LGSPAIR, SPAIR

def kl_divergence(z_mean, z_sig):
    z_log_var = tf_safe_log(tf.square(z_sig))
    if (len(z_mean.shape)==2):
        kl_loss = tf.reduce_mean(-0.5 * tf.reduce_sum(1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var),axis=1))
    elif (len(z_mean.shape)==4):
        kl_loss = tf.reduce_mean(-0.5 * tf.reduce_sum(1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var),axis=[1,2,3]))
    else:
        raise NotImplementedError('This KL shape is not implemented')   
    return kl_loss

def kl_divergence_two_gauss(mean1,sig1,mean2,sig2):
    return tf.reduce_mean(tf.reduce_sum(tf_safe_log(sig2) - tf_safe_log(sig1) + ((tf.math.square(sig1) + tf.math.square(mean1-mean2)) / (2*tf.math.square(sig2))) - 0.5, axis=[1,2,3]))


# https://github.com/e2crawfo/auto_yolo/blob/aaai_2019/auto_yolo/models/core.py#L42
def concrete_binary_sample_kl(pre_sigmoid_sample,
                              prior_log_odds, prior_temperature,
                              posterior_log_odds, posterior_temperature,
                              eps=1e-8):
    y = pre_sigmoid_sample

    y_times_prior_temp = y * prior_temperature
    log_prior = tf.math.log(prior_temperature + eps) - y_times_prior_temp + prior_log_odds - \
        2.0 * tf.math.log(1.0 + tf.exp(-y_times_prior_temp + prior_log_odds) + eps)

    y_times_posterior_temp = y * posterior_temperature
    log_posterior = tf.math.log(posterior_temperature + eps) - y_times_posterior_temp + posterior_log_odds - \
        2.0 * tf.math.log(1.0 + tf.exp(-y_times_posterior_temp + posterior_log_odds) + eps)

    return log_posterior - log_prior

#https://github.com/e2crawfo/auto_yolo/blob/aaai_2019/auto_yolo/models/yolo_air.py#L592
def compute_z_pres_kl_yolo_air(z_pres,z_pres_logits,z_pres_pre_sigmoid,prior_prob,temperature):
    H = z_pres.shape[1]
    W = z_pres.shape[2]

    batch_size = z_pres.shape[0]
    count_support = tf.range(H*W+1, dtype=tf.float32)

    count_prior_prob = 1 - prior_prob # tf.nn.sigmoid(self.count_prior_log_odds)
    count_distribution = (1 - count_prior_prob) * (count_prior_prob ** count_support)

    normalizer = tf.reduce_sum(count_distribution)
    count_distribution = count_distribution / tf.maximum(normalizer, 1e-6)
    count_distribution = tf.tile(count_distribution[None, :], (batch_size, 1))
    count_so_far = tf.zeros((batch_size, 1), dtype=tf.float32)

    i = 0

    obj_kl = []
    max_n_objects = H*W
    for h in range(H):
        for w in range(W):
            p_z_given_Cz = tf.maximum(count_support[None, :] - count_so_far, 0) / (max_n_objects - i)

            # Reshape for batch matmul
            _count_distribution = count_distribution[:, None, :]
            _p_z_given_Cz = p_z_given_Cz[:, :, None]

            p_z = tf.matmul(_count_distribution, _p_z_given_Cz)[:, :, 0]

            prior_log_odds = tf_safe_log(p_z) - tf_safe_log(1-p_z)
            _obj_kl = concrete_binary_sample_kl(
                z_pres_pre_sigmoid[:, h, w, :],
                prior_log_odds, temperature,
                z_pres_logits[:, h, w, :], temperature
            )

            obj_kl.append(_obj_kl)

            sample = tf.cast(z_pres[:, h, w, :] > 0.5, tf.float32)
            mult = sample * p_z_given_Cz + (1-sample) * (1-p_z_given_Cz)
            count_distribution = mult * count_distribution
            normalizer = tf.reduce_sum(count_distribution, axis=1, keepdims=True)
            normalizer = tf.maximum(normalizer, 1e-6)
            count_distribution = count_distribution / normalizer

            count_so_far += sample

            i += 1

    return tf_mean_sum(tf.transpose(tf.squeeze(tf.stack([obj_kl])),[1,0])) #[cells, batch] -> [batch,cells]


def tf_safe_log(value, replacement_value=-100.0):
    log_value = tf.math.log(value + 1e-8)
    replace = tf.logical_or(tf.math.is_nan(log_value), tf.math.is_inf(log_value))
    log_value = tf.where(replace, replacement_value * tf.ones_like(log_value), log_value)
    return log_value

def xent_loss(label,pred):
    return -(label * tf_safe_log(pred) + (1. - label) * tf_safe_log(1. - pred))

#https://github.com/e2crawfo/dps/blob/8ff5eab735a1381445e2dbf48c04f7b515ab64bb/dps/utils/tf.py#L272
def tf_mean_sum(t):
    """ Average over batch dimension, sum over all other dimensions """
    return tf.reduce_mean(tf.reduce_sum(tf.reshape(t,[t.shape[0],tf.reduce_prod(t.shape[1:])]), axis=1))


def train_spair(model, optimizer, dataset, train_dataset, test_dataset, config):
    RUN_NAME = datetime.now().strftime("%Y%m%d-%H%M%S")

    data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'output/')
    if not os.path.exists(data_path):
        print('data folder doesn\'t exist, create data folder')
        os.mkdir(data_path)

    RUN_DIR = data_path + RUN_NAME + '/'
    os.mkdir(RUN_DIR)

    train_metric_names = ['x_recon_train_loss','z_zoom_kl_train_loss','z_what_kl_train_loss','z_where_kl_train_loss',
                            'z_depth_kl_train_loss','z_pres_kl_train_loss','z_bg_kl_train_loss','z_l_kl_train_loss','x_hat_recon_train_loss']

    test_metric_names = ['x_recon_test_loss','z_zoom_kl_test_loss','z_what_kl_test_loss','z_where_kl_test_loss',
                            'z_depth_kl_test_loss','z_pres_kl_test_loss','z_bg_kl_test_loss','z_l_kl_test_loss','x_hat_recon_test_loss','MAE test','MAPE test']

    train_metrics = [tf.keras.metrics.Mean(name=loss) for loss in train_metric_names]
    test_metrics = [tf.keras.metrics.Mean(name=loss) for loss in test_metric_names]

    count_acc_test = tf.keras.metrics.Accuracy()



    @tf.function
    def train_step(model, images, optimizer,step):
        with tf.GradientTape() as tape:
            (x_recon, z_what, z_what_mean, z_what_sigma, z_where, z_where_mean, z_where_sigma,
                z_depth, z_depth_mean, z_depth_sigma, z_pres, z_pres_logits, z_pres_pre_sigmoid,
                all_glimpses, obj_recon_unnorm, obj_recon_alpha, obj_full_recon_unnorm, obj_bbox_mask, *more_outputs) = model(images,training=True)

            if config.model=='lg_spair':
                x, x_hat = images[:,:,:,:3], images[:,:,:,3:]
            else:
                x = images

            x_recon_loss = tf_mean_sum(xent_loss(x,x_recon))
            
            prior_z_pres_prob = tf.constant(0.99) * tf.minimum(1.0, (step+1) / config.z_pres_anneal_step) # anneal from 0 to 0.99
            z_pres_kl_loss = compute_z_pres_kl_yolo_air(z_pres,z_pres_logits,z_pres_pre_sigmoid,prior_z_pres_prob,config.tau)

            prior_z_zoom_mean = tf.constant(config.prior_z_zoom,shape=z_where_mean[:,:,:,:2].shape) + (config.prior_z_zoom_start * (1 - (tf.minimum( (step+1) / config.z_pres_anneal_step, 1.0)))) # anneal from 10 to prior_z_zoom
            prior_z_zoom_sig = tf.constant(0.5,shape=z_where_sigma[:,:,:,:2].shape)

            z_where_zoom_kl_loss = kl_divergence_two_gauss(z_where_mean[:,:,:,:2],z_where_sigma[:,:,:,:2],prior_z_zoom_mean,prior_z_zoom_sig)
            z_what_kl_loss = kl_divergence(z_what_mean, z_what_sigma)

            z_where_kl_loss = kl_divergence(z_where_mean[:,:,:,2:], z_where_sigma[:,:,:,2:])
            z_depth_kl_loss = kl_divergence(z_depth_mean, z_depth_sigma)

            losses = [x_recon_loss,z_where_zoom_kl_loss,z_what_kl_loss, z_where_kl_loss, z_depth_kl_loss, z_pres_kl_loss]

            total_loss = config.reconstruction_weight * x_recon_loss +\
                tf.minimum(config.beta,config.beta * (step+1.0)/config.anneal_until) *\
                (config.z_what_beta  * z_what_kl_loss + z_depth_kl_loss + z_where_kl_loss + z_where_zoom_kl_loss + z_pres_kl_loss)

            if config.model=='lg_spair':
                z_bg, z_bg_mean, z_bg_sig, x_hat_recon, z_l, z_l_mean, z_l_sig = more_outputs
                if not config.split_z_l:
                    
                    if config.concat_z_bg:
                        z_bg_kl_loss = kl_divergence(tf.concat([z_bg_mean,z_l_mean],axis=1),tf.concat([z_bg_sig,z_l_sig],axis=1)) # kl_divergence(z_bg_mean, z_bg_sig) #  #  # # #  # k# 
                    else:
                        z_bg_kl_loss = kl_divergence(z_bg_mean, z_bg_sig)
                    if config.concat_z_what:
                        z_what_kl_loss = kl_divergence(tf.concat([z_what_mean,tf.tile(z_l_mean[:,tf.newaxis,tf.newaxis,:],[1,4,4,1])],axis=-1),tf.concat([z_what_sigma,tf.tile(z_l_sig[:,tf.newaxis,tf.newaxis,:],[1,4,4,1])],axis=-1))

                    z_l_kl_loss = kl_divergence(z_l_mean, z_l_sig)
                    # z_what_concat_kl_loss = kl_divergence(tf.concat([z_what_mean,z_l_mean],axis=-1),tf.concat([z_what_sigma,z_l_sig],axis=-1))
                    x_hat_recon_loss = tf_mean_sum(xent_loss(x_hat,x_hat_recon))
                    
                    losses.append(z_bg_kl_loss)
                    losses.append(z_l_kl_loss)
                    losses.append(x_hat_recon_loss)

                    total_loss = config.z_bg_beta * z_bg_kl_loss + config.reconstruction_weight * x_recon_loss + \
                        config.beta * (config.z_what_beta  * z_what_kl_loss + z_depth_kl_loss + z_where_kl_loss + z_where_zoom_kl_loss + z_pres_kl_loss) + x_hat_recon_loss # + z_l_kl_loss
                
                else:
                    z_bg_kl_loss = kl_divergence(z_bg_mean,z_bg_sig) # kl_divergence(z_bg_mean, z_bg_sig) #  #  # # #  # k# 
                    z_l_kl_loss = kl_divergence(z_l_mean, z_l_sig)
                    # z_what_concat_kl_loss = kl_divergence(tf.concat([z_what_mean,z_l_mean],axis=-1),tf.concat([z_what_sigma,z_l_sig],axis=-1))
                    x_hat_recon_loss = tf_mean_sum(xent_loss(x_hat,x_hat_recon))
                    
                    losses.append(z_bg_kl_loss)
                    losses.append(z_l_kl_loss)
                    losses.append(x_hat_recon_loss)
                    total_loss = config.z_bg_beta * z_bg_kl_loss + config.z_l_beta * z_l_kl_loss  + x_hat_recon_loss + config.reconstruction_weight * x_recon_loss + \
                        config.beta * (config.z_what_beta  * z_what_kl_loss + z_depth_kl_loss + z_where_kl_loss + z_where_zoom_kl_loss + z_pres_kl_loss)  
                

            elif config.model=='lg_glimpse_spair':
                z_bg, z_bg_mean, z_bg_sig, x_hat_recon, z_l, z_l_mean, z_l_sig, x_hat = more_outputs
                z_bg_kl_loss = kl_divergence(z_bg_mean, z_bg_sig) # kl_divergence(z_bg_mean, z_bg_sig) #  #  # # #  # k# 
                z_l_kl_loss = kl_divergence(z_l_mean, z_l_sig)
                z_what_concat_kl_loss = kl_divergence(tf.concat([z_what_mean,z_l_mean],axis=-1),tf.concat([z_what_sigma,z_l_sig],axis=-1))
                x_hat_recon_loss = tf_mean_sum(xent_loss(tf.stop_gradient(x_hat),x_hat_recon))

                losses.append(z_bg_kl_loss)
                losses.append(z_l_kl_loss)
                losses.append(x_hat_recon_loss)
                total_loss = config.z_bg_beta * z_bg_kl_loss  + x_hat_recon_loss + config.reconstruction_weight * x_recon_loss + \
                    config.beta * (config.z_what_beta  * z_what_concat_kl_loss + z_depth_kl_loss + z_where_kl_loss + z_where_zoom_kl_loss + z_pres_kl_loss)  
                

            elif config.model=='bg_spair':
                z_bg, z_bg_mean, z_bg_sig = more_outputs
                z_bg_kl_loss = kl_divergence(z_bg_mean, z_bg_sig)
                losses.append(z_bg_kl_loss)

                total_loss = config.z_bg_beta * z_bg_kl_loss + config.reconstruction_weight * x_recon_loss +\
                    tf.minimum(config.beta,config.beta * (step+1.0)/config.anneal_until) *\
                    (config.z_what_beta  * z_what_kl_loss + z_depth_kl_loss + z_where_kl_loss + z_where_zoom_kl_loss + z_pres_kl_loss)

        gradients = tape.gradient(total_loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))

        for (metric,loss) in zip(train_metrics,losses):
            metric(loss)

        return (x_recon, z_what, z_what_mean, z_what_sigma, z_where, z_where_mean, z_where_sigma,
            z_depth, z_depth_mean, z_depth_sigma, z_pres, z_pres_logits, z_pres_pre_sigmoid,
            all_glimpses, obj_recon_unnorm, obj_recon_alpha, obj_full_recon_unnorm, *more_outputs)


    @tf.function
    def test_step(model, images, labels=None):
        (x_recon, z_what, z_what_mean, z_what_sigma, z_where, z_where_mean, z_where_sigma,
            z_depth, z_depth_mean, z_depth_sigma, z_pres, z_pres_logits, z_pres_pre_sigmoid,
            all_glimpses, obj_recon_unnorm, obj_recon_alpha, obj_full_recon_unnorm, obj_bbox_mask, *more_outputs) = model(images,training=True)

        if config.model!='spair':
            x, x_hat = images[:,:,:,:3], images[:,:,:,3:]
        else:
            x = images

        x_recon_loss = tf_mean_sum(xent_loss(x,x_recon))
        
        prior_z_pres_prob = tf.constant(0.99)
        z_pres_kl_loss = compute_z_pres_kl_yolo_air(z_pres,z_pres_logits,z_pres_pre_sigmoid,prior_z_pres_prob,config.tau)

        prior_z_zoom_mean = tf.constant(config.prior_z_zoom,shape=z_where_mean[:,:,:,:2].shape)
        prior_z_zoom_sig = tf.constant(0.5,shape=z_where_sigma[:,:,:,:2].shape)

        z_where_zoom_kl_loss = kl_divergence_two_gauss(z_where_mean[:,:,:,:2],z_where_sigma[:,:,:,:2],prior_z_zoom_mean,prior_z_zoom_sig)
        z_what_kl_loss = kl_divergence(z_what_mean, z_what_sigma)
        z_where_kl_loss = kl_divergence(z_where_mean[:,:,:,2:], z_where_sigma[:,:,:,2:])
        z_depth_kl_loss = kl_divergence(z_depth_mean, z_depth_sigma)

        losses = [x_recon_loss,z_where_zoom_kl_loss,z_what_kl_loss, z_where_kl_loss, z_depth_kl_loss, z_pres_kl_loss]

        if config.model=='lg_spair':
            z_bg, z_bg_mean, z_bg_sig, x_hat_recon, z_l, z_l_mean, z_l_sig = more_outputs
            # z_bg_kl_loss = kl_divergence(z_bg_mean, z_bg_sig)
            z_bg_kl_loss = kl_divergence(tf.concat([z_bg_mean,z_l_mean],axis=1),tf.concat([z_bg_sig,z_l_sig],axis=1))
            z_l_kl_loss = kl_divergence(z_l_mean, z_l_sig)
            x_hat_recon_loss = tf_mean_sum(xent_loss(x_hat,x_hat_recon))
            losses.append(z_bg_kl_loss)
            losses.append(z_l_kl_loss)
            losses.append(x_hat_recon_loss)

        elif config.model=='lg_glimpse_spair':
            z_bg, z_bg_mean, z_bg_sig, x_hat_recon, z_l, z_l_mean, z_l_sig, x_hat = more_outputs
            z_bg_kl_loss = kl_divergence(z_bg_mean, z_bg_sig) # kl_divergence(z_bg_mean, z_bg_sig) #  #  # # #  # k# 
            z_l_kl_loss = kl_divergence(z_l_mean, z_l_sig)
            z_what_concat_kl_loss = kl_divergence(tf.concat([z_what_mean,z_l_mean],axis=-1),tf.concat([z_what_sigma,z_l_sig],axis=-1))
            x_hat_recon_loss = tf_mean_sum(xent_loss(tf.stop_gradient(x_hat),x_hat_recon))

            losses.append(z_bg_kl_loss)
            losses.append(z_l_kl_loss)
            losses.append(x_hat_recon_loss)

        elif config.model=='bg_spair':
            z_bg, z_bg_mean, z_bg_sig = more_outputs
            z_bg_kl_loss = kl_divergence(z_bg_mean, z_bg_sig)
            losses.append(z_bg_kl_loss)
            losses.append(tf.constant(0.0))
            losses.append(tf.constant(0.0))
            

        if labels is not None:
            print('image shape:',images.shape)
            pred_count = tf.reduce_sum(tf.round(tf.sigmoid(z_pres_logits)),axis=[1,2,3]) #[B,]
            print('pred count shape:',pred_count.shape)
            print('label shape:',labels.shape)
            mae = tf.metrics.mean_absolute_error(labels,pred_count)
            mape = tf.metrics.mean_absolute_percentage_error(labels,pred_count)
            losses.append(mae)
            losses.append(mape)
            count_acc_test.update_state(labels,pred_count)

        for (metric,loss) in zip(test_metrics,losses):
            metric(loss)

        return (x_recon, z_what, z_what_mean, z_what_sigma, z_where, z_where_mean, z_where_sigma,
            z_depth, z_depth_mean, z_depth_sigma, z_pres, z_pres_logits, z_pres_pre_sigmoid,
            all_glimpses, obj_recon_unnorm, obj_recon_alpha, obj_full_recon_unnorm, *more_outputs)



    # Train
    start = time.time()
    for step,images in enumerate(train_dataset):
        (x_recon, z_what, z_what_mean, z_what_sigma, z_where, z_where_mean, z_where_sigma,
                z_depth, z_depth_mean, z_depth_sigma, z_pres, z_pres_logits, z_pres_pre_sigmoid,
                all_glimpses, obj_recon_unnorm, obj_recon_alpha, obj_full_recon_unnorm, *more_outputs) = train_step(model, images, optimizer,tf.constant(step,dtype=tf.float32))

        if (step%1000==0):
            print('Training time: {:.2f}'.format(time.time()-start))

            training_log = dict(zip(train_metric_names,[metric.result().numpy() for metric in train_metrics]))
            print('Training step:',step)
            print(training_log)
            print()


            for metric in train_metrics:
                metric.reset_states()

            num_cells = z_where.shape[1]*z_where.shape[2]
            f, ax = plt.subplots(1, 3)
            h,w,channel = images.shape[1:4]
            channel = min(3,channel)
            ax[0].set_xticks(np.arange(0, h*10, w))
            ax[0].set_yticks(np.arange(0, h*(num_cells+2), w))
            ax[1].set_xticks(np.arange(0, h*10, w))
            ax[1].set_yticks(np.arange(0, h*(num_cells+2), w))
            ax[2].set_xticks(np.arange(0, h*10, w))
            ax[2].set_yticks(np.arange(0, h*(num_cells+2), w))


            obj_recon = obj_full_recon_unnorm[:,:,:,:,:channel]
            obj_alpha = obj_full_recon_unnorm[:,:,:,:,channel:]

            z_depth = tf.reshape(z_depth, [32, num_cells, 1, 1, 1])
            z_pres = tf.reshape(z_pres, [32, num_cells, 1, 1, 1])

            canvas = np.empty((h*(num_cells+2), w*10, channel))
            canvas_weighted = np.empty((h*(num_cells+2), w*10, channel))
            canvas_weights_only = np.empty((h*(num_cells+2), w*10, channel)) # only weights of that part

            for i in range(10):
                canvas_weights_only[0:h,i*w:(i+1)*w, :] = canvas_weighted[0:h,i*w:(i+1)*w, :] = canvas[0:h,i*w:(i+1)*w, :] = images[i,:,:,:3]
                canvas_weights_only[h:h*2, i*w:(i+1)*w, :] = canvas_weighted[h:h*2, i*w:(i+1)*w, :] = canvas[h:h*2, i*w:(i+1)*w, :] = x_recon[i].numpy().reshape((h,w,channel))

                canvas[h*2:, i*w:(i+1)*w, :] = obj_recon[i].numpy().reshape((num_cells*h,w,channel))
                canvas_weighted[h*2:, i*w:(i+1)*w, :] = (obj_recon[i]*obj_alpha[i]*z_pres[i]*tf.nn.sigmoid(-z_depth[i])).numpy().reshape((num_cells*h,w,channel))
                canvas_weights_only[h*2:, i*w:(i+1)*w, 0] = (tf.ones(shape=obj_alpha[i].shape)*z_pres[i]).numpy().reshape((num_cells*h,w))  # *tf.nn.sigmoid(-z_depth[i])


            ax[0].imshow(np.squeeze(canvas),cmap='gray')
            ax[0].set_title('reconstruction')
            ax[0].grid(b=True, which='major', color='#ffffff', linestyle='-')
            ax[0].tick_params(top=False, bottom=False, left=False, right=False, labelleft=False, labelbottom=False)


            ax[1].imshow(np.squeeze(canvas_weighted),cmap='gray')
            ax[1].set_title('reconstruction weighted')
            ax[1].grid(b=True, which='major', color='#ffffff', linestyle='-')
            ax[1].tick_params(top=False, bottom=False, left=False, right=False, labelleft=False, labelbottom=False)

            ax[2].imshow(np.squeeze(canvas_weights_only),cmap='inferno')
            ax[2].set_title('weights')
            ax[2].grid(b=True, which='major', color='#ffffff', linestyle='-')
            ax[2].tick_params(top=False, bottom=False, left=False, right=False, labelleft=False, labelbottom=False)

            plt.savefig(RUN_DIR+'train_recon_it_'+str(step)+'.png')

            start = time.time()
            for test_num,test_ds in enumerate(test_dataset):
                # Evaluation
                for test_data in test_ds:
                    if config.label:
                        test_images,labels = test_data[0], test_data[1]
                        test_step(model, test_images, labels)
                    else:
                        test_images = test_data
                        test_step(model, test_images)
                    
                print('Testing time: {:.2f}'.format(time.time()-start))

                test_log = dict(zip([name+str(test_num) for name in test_metric_names],[metric.result().numpy() for metric in test_metrics]))
                print('Count accuracy' + str(test_num) + ': ' + str(count_acc_test.result().numpy()))
                print(test_log)
                print()
                
                count_acc_test.reset_states()
                for metric in train_metrics:
                    metric.reset_states()


                x_recon_plt = visualizer.reconstruction_test(model, test_ds, filename = '_it_'+str(step)+'_'+ str(test_num), filepath = RUN_DIR, label=config.label)
                x_recon_plt.close()
                plt.close()

                visualizer.reconstruction_bbox(model, test_ds, filename = '_it_'+str(step)+'_'+ str(test_num), filepath = RUN_DIR, label=config.label)


                glimpses_recon_plt = visualizer.glimpses_reconstruction_test(model, test_ds, filename = '_it_'+str(step)+'_'+ str(test_num), filepath = RUN_DIR, label=config.label)
                glimpses_recon_plt.close()
                plt.close()
                if isinstance(model,LGSPAIR) :
                    visualizer.x_hat_reconstruction_test(model, test_ds, filename = '_it_'+str(step)+'_'+ str(test_num), filepath = RUN_DIR, label=config.label)
                

                gc.collect()
                start = time.time()

        if (step>=config.training_steps):
            print('Training done!')
            break

    model.save_weights('models/'+RUN_NAME+'.h5')
