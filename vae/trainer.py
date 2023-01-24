import tensorflow as tf
import numpy as np
from datetime import datetime
import time
import visualizer
import gc
import os
from model import Classifier, LGVae, LGGMVae, GMVae
from classifier import main as train_classifier

def kl_divergence(z_mean, z_sig):
    z_log_var = tf.math.log(tf.square(z_sig))
    kl_loss = tf.reduce_mean(-0.5 * tf.reduce_sum(1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var),axis=1))

    return kl_loss

def kl_divergence_two_gauss(mean1,sig1,mean2,sig2):
    return tf.reduce_mean(tf.reduce_sum(tf.math.log(sig2) - tf.math.log(sig1) + ((tf.math.square(sig1) + tf.math.square(mean1-mean2)) / (2*tf.math.square(sig2))) - 0.5, axis=1))


def discretised_logistic_loss(x, m, log_scales):
    # modified from https://github.com/openai/pixel-cnn/blob/master/pixel_cnn_pp/nn.py
    centered_x = x - m
    inv_stdv = tf.exp(-log_scales)
    plus_in = inv_stdv * (centered_x + 1./255.)
    min_in = inv_stdv * (centered_x - 1./255.)
    cdf_plus = tf.nn.sigmoid(plus_in) # 1/(exp(-x)+1)
    cdf_min = tf.nn.sigmoid(min_in)
    cdf_delta = cdf_plus - cdf_min

    mid_in = inv_stdv * centered_x
    log_pdf_mid = mid_in - log_scales - 2.*tf.nn.softplus(mid_in)

    log_cdf_plus = plus_in - tf.nn.softplus(plus_in)
    log_one_minus_cdf_min = -tf.nn.softplus(min_in) # - log(exp(x)+ 1)

    log_prob = tf.where(x < -0.999, log_cdf_plus, tf.where(x > 0.999, log_one_minus_cdf_min, tf.where(cdf_delta > 1e-5, tf.math.log(tf.maximum(cdf_delta, 1e-12)), log_pdf_mid - np.log(127.5))))
    return - log_prob

def linear_assignment(labels,pred):
    num_class = labels.shape[1]
    num_cluster = pred.shape[1]
    # print('labels.shape:',labels.shape)
    # print('pred.shape:',pred.shape)

    # onehot -> int
    labels = tf.argmax(labels,axis=1)
    cluster = tf.argmax(pred,axis=1)
    # print('labels.shape:',labels.shape)
    # print('cluster.shape:',cluster.shape)

    cluster_pred = tf.zeros_like(labels)

    for i in range(num_cluster):
        gt_class,_,count = tf.unique_with_counts(labels[cluster==i])
        if count.shape[0]!=0: # skip if cluster doesn't exists
            maj_class = gt_class[tf.argmax(count)]
            cluster_pred =  tf.where(cluster==i,maj_class,cluster_pred)

            # rep = tf.math.reduce_sum(count)[tf.newaxis]
            # print('maj_class.shape:',maj_class.shape)
            # print('rep.shape:',rep.shape)
            # updates = tf.tile(maj_class,rep)
            # print('update.shape:',updates.shape)
            # cluster_pred = tf.tensor_scatter_nd_update(cluster_pred,tf.where(cluster==i),updates)

    return tf.squeeze(tf.one_hot(cluster_pred,num_class))




def train_local_global_autoencoder(model, optimizer, dataset, train_dataset, test_dataset, config):
    RUN_NAME = datetime.now().strftime("%Y%m%d-%H%M%S")
    # wandb.init(config = config, project = "lg-vae-project", tags = [config.tag], name = RUN_NAME, reinit = True)
    data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'output/')
    if not os.path.exists(data_path):
        print('data folder doesn\'t exist, create data folder')
        os.mkdir(data_path)
    RUN_DIR = 'output/' + RUN_NAME + '/'
    os.mkdir(RUN_DIR)
    if config.label:
        if not os.path.exists('models/svhn_classifier_weights.h5'):
            print('Classifer model not found, training a new classifier')
            train_classifier()
        # Check classifier performance
        classifier = Classifier(target_shape=10)
        classifier(tf.zeros([8,32,32,3])) #build model
        classifier.summary()
        classifier.load_weights('models/svhn_classifier_weights.h5')
        test_acc = tf.keras.metrics.CategoricalAccuracy()
        
        for test_images,labels in test_dataset:
            x = test_images[:,:,:,:3]
            pred = classifier(x)
            test_acc(labels, pred)
        print('Test acc: {:.4f}'.format(test_acc.result()))
        del test_acc
    
    x_recon_train_loss = tf.keras.metrics.Mean(name='x_recon_train_loss')
    x_kl_train_loss = tf.keras.metrics.Mean(name='x_kl_train_loss')
    x_recon_test_loss = tf.keras.metrics.Mean(name='x_recon_test_loss')
    x_kl_test_loss = tf.keras.metrics.Mean(name='x_kl_test_loss')

    total_kl_train_loss = tf.keras.metrics.Mean(name='total_kl_train_loss')
    total_kl_test_loss = tf.keras.metrics.Mean(name='total_test_loss')

    x_hat_recon_train_loss = tf.keras.metrics.Mean(name='x_hat_recon_train_loss')
    x_hat_kl_train_loss = tf.keras.metrics.Mean(name='x_hat_kl_train_loss')
    x_hat_recon_test_loss = tf.keras.metrics.Mean(name='x_hat_recon_test_loss')
    x_hat_kl_test_loss = tf.keras.metrics.Mean(name='x_hat_kl_test_loss')

    y_kl_train_loss = tf.keras.metrics.Mean(name='y_kl_train_loss')
    y_kl_test_loss = tf.keras.metrics.Mean(name='y_kl_test_loss')

    classifier_recon_acc = tf.keras.metrics.CategoricalAccuracy()
    classifier_random_z_l_acc = tf.keras.metrics.CategoricalAccuracy()
    classifier_random_z_g_acc = tf.keras.metrics.CategoricalAccuracy()
    classifier_cluster_acc = tf.keras.metrics.CategoricalAccuracy()

    @tf.function
    def train_step_lg_vae(model, images, optimizer):
        with tf.GradientTape() as tape:
            x_mean, x_log_scale, z_x, z_mean_x, z_sig_x, z_x_hat, x_hat_mean, x_hat_log_scale, z_mean_x_hat, z_sig_x_hat = model(images, training=True)

            x, x_hat = images[:,:,:,:3], images[:,:,:,3:]

            x_recon_loss = tf.reduce_mean(tf.reduce_mean(discretised_logistic_loss(x, x_mean,x_log_scale), axis=[1,2,3]))
            x_hat_recon_loss = tf.reduce_mean(tf.reduce_mean(discretised_logistic_loss(x_hat, x_hat_mean,x_hat_log_scale), axis=[1,2,3]))

            total_kl_loss = config.beta * kl_divergence(tf.concat([z_mean_x,z_mean_x_hat], axis=1), tf.concat([z_sig_x,z_sig_x_hat] ,axis=1))
            x_kl_loss = kl_divergence(z_mean_x, z_sig_x)
            x_hat_kl_loss = kl_divergence(z_mean_x_hat, z_sig_x_hat)


            total_loss = x_recon_loss  + x_hat_recon_loss + total_kl_loss 
        
        gradients = tape.gradient(total_loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))

        x_recon_train_loss(x_recon_loss)
        x_kl_train_loss(x_kl_loss)
        x_hat_recon_train_loss(x_hat_recon_loss)
        x_hat_kl_train_loss(x_hat_kl_loss)
        total_kl_train_loss(total_kl_loss)

    @tf.function
    def train_step_lg_gm_vae(model, images, optimizer):
        with tf.GradientTape() as tape:
            x_mean, x_log_scale, z_x, z_mean_x, z_sig_x, z_x_hat, x_hat_mean, x_hat_log_scale, z_mean_x_hat, z_sig_x_hat, y, y_logits, z_prior_mean, z_prior_sig = model(images,training=True)

            x, x_hat = images[:,:,:,:3], images[:,:,:,3:]

            x_recon_loss = tf.reduce_mean(tf.reduce_mean(discretised_logistic_loss(x, x_mean,x_log_scale), axis=[1,2,3]))
            x_hat_recon_loss = tf.reduce_mean(tf.reduce_mean(discretised_logistic_loss(x_hat, x_hat_mean,x_hat_log_scale), axis=[1,2,3]))


            x_kl_loss = kl_divergence_two_gauss(z_mean_x, z_sig_x, z_prior_mean, z_prior_sig)
            x_hat_kl_loss = kl_divergence_two_gauss(z_mean_x_hat, z_sig_x_hat, 0., 1.)

            py = tf.nn.softmax(y_logits, axis=1)
            y_kl_loss = tf.reduce_mean(tf.reduce_sum(py * (tf.math.log(py + 1e-8) - tf.math.log(1.0/model.y_size)), axis=1))


            total_loss = x_recon_loss  + x_hat_recon_loss + config.beta * (x_kl_loss + x_hat_kl_loss) + config.alpha * y_kl_loss
        
        gradients = tape.gradient(total_loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))

        x_recon_train_loss(x_recon_loss)
        x_kl_train_loss(x_kl_loss)
        x_hat_recon_train_loss(x_hat_recon_loss)
        x_hat_kl_train_loss(x_hat_kl_loss)
        y_kl_train_loss(y_kl_loss)

    @tf.function
    def train_step_gm_vae(model, images, optimizer):
        with tf.GradientTape() as tape:
            x_mean, x_log_scale, z_x, z_mean_x, z_sig_x, y, y_logits, z_prior_mean, z_prior_sig = model(images,training=True)

            x = images[:,:,:,:3]

            x_recon_loss = tf.reduce_mean(tf.reduce_mean(discretised_logistic_loss(x, x_mean,x_log_scale), axis=[1,2,3]))
            x_kl_loss = kl_divergence_two_gauss(z_mean_x, z_sig_x, z_prior_mean, z_prior_sig)

            py = tf.nn.softmax(y_logits, axis=1)
            y_kl_loss = tf.reduce_mean(tf.reduce_sum(py * (tf.math.log(py + 1e-8) - tf.math.log(1.0/model.y_size)), axis=1))

            total_loss = x_recon_loss   + config.beta * x_kl_loss + config.alpha * y_kl_loss
        
        gradients = tape.gradient(total_loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))

        x_recon_train_loss(x_recon_loss)
        x_kl_train_loss(x_kl_loss)
        y_kl_train_loss(y_kl_loss)

    
      
    @tf.function
    def test_step_lg_vae(model, images, labels=None):
        x_mean, x_log_scale, z_x, z_mean_x, z_sig_x, z_x_hat, x_hat_mean, x_hat_log_scale, z_mean_x_hat, z_sig_x_hat = model(images)

        x, x_hat = images[:,:,:,:3], images[:,:,:,3:]

        x_recon_loss = tf.reduce_mean(tf.reduce_mean(discretised_logistic_loss(x, x_mean,x_log_scale), axis=[1,2,3]))
        x_hat_recon_loss = tf.reduce_mean(tf.reduce_mean(discretised_logistic_loss(x_hat, x_hat_mean,x_hat_log_scale), axis=[1,2,3]))

        total_kl_loss = config.beta * kl_divergence(tf.concat([z_mean_x,z_mean_x_hat], axis=1), tf.concat([z_sig_x,z_sig_x_hat] ,axis=1))
        x_kl_loss = kl_divergence(z_mean_x, z_sig_x)
        x_hat_kl_loss = kl_divergence(z_mean_x_hat, z_sig_x_hat)
        

        if labels is not None:
            pred = classifier(x_mean)
            classifier_recon_acc(labels,pred)

            # vary z_l
            random_z_l = np.random.normal(size=[z_x.shape[0], model.local_latent_dims]).astype(np.float32)
            x_recon_with_random_z_l,_ = model.decode(z_x, random_z_l)
            pred_random_z_l = classifier(x_recon_with_random_z_l)
            classifier_random_z_l_acc(labels,pred_random_z_l)
            #vary z_g
            random_z_g = np.random.normal(size=[z_x_hat.shape[0], model.global_latent_dims]).astype(np.float32)
            x_recon_with_random_z_g,_ = model.decode(random_z_g, z_x_hat)
            pred_random_z_g = classifier(x_recon_with_random_z_g)
            classifier_random_z_g_acc(labels,pred_random_z_g)


        x_recon_test_loss(x_recon_loss)
        x_kl_test_loss(x_kl_loss)
        x_hat_recon_test_loss(x_hat_recon_loss)
        x_hat_kl_test_loss(x_hat_kl_loss)
        total_kl_test_loss(total_kl_loss)


    @tf.function
    def test_step_lg_gm_vae(model, images, labels=None):
        x_mean, x_log_scale, z_x, z_mean_x, z_sig_x, z_x_hat, x_hat_mean, x_hat_log_scale, z_mean_x_hat, z_sig_x_hat, y, y_logits, z_prior_mean, z_prior_sig = model(images)

        x, x_hat = images[:,:,:,:3], images[:,:,:,3:]

        x_recon_loss = tf.reduce_mean(tf.reduce_mean(discretised_logistic_loss(x, x_mean,x_log_scale), axis=[1,2,3]))
        x_hat_recon_loss = tf.reduce_mean(tf.reduce_mean(discretised_logistic_loss(x_hat, x_hat_mean,x_hat_log_scale), axis=[1,2,3]))

        x_kl_loss = kl_divergence_two_gauss(z_mean_x, z_sig_x, z_prior_mean, z_prior_sig)
        x_hat_kl_loss = kl_divergence_two_gauss(z_mean_x_hat, z_sig_x_hat, 0., 1.)

        py = tf.nn.softmax(y_logits, axis=1)
        y_kl_loss = tf.reduce_mean(tf.reduce_sum(py * (tf.math.log(py + 1e-8) - tf.math.log(1.0/model.y_size)), axis=1))

        if labels is not None:
            pred = classifier(x_mean)
            classifier_recon_acc(labels,pred)

            # vary z_l
            random_z_l = np.random.normal(size=[z_x_hat.shape[0], model.local_latent_dims]).astype(np.float32)
            x_recon_with_random_z_l,_ = model.decode(z_x, random_z_l)
            pred_random_z_l = classifier(x_recon_with_random_z_l)
            classifier_random_z_l_acc(labels,pred_random_z_l)
            #vary z_g
            random_z_g = z_prior_mean + np.random.normal(size=[z_prior_mean.shape[0], model.global_latent_dims]).astype(np.float32) * z_prior_sig
            x_recon_with_random_z_g,_ = model.decode(random_z_g, z_x_hat)
            pred_random_z_g = classifier(x_recon_with_random_z_g)
            classifier_random_z_g_acc(labels,pred_random_z_g)



        x_recon_test_loss(x_recon_loss)
        x_kl_test_loss(x_kl_loss)
        x_hat_recon_test_loss(x_hat_recon_loss)
        x_hat_kl_test_loss(x_hat_kl_loss)
        y_kl_test_loss(y_kl_loss)

        return x_mean, x_log_scale, z_x, z_mean_x, z_sig_x, z_x_hat, x_hat_mean, x_hat_log_scale, z_mean_x_hat, z_sig_x_hat, y, y_logits, z_prior_mean, z_prior_sig

    @tf.function
    def test_step_gm_vae(model, images, labels=None):
        x_mean, x_log_scale, z_x, z_mean_x, z_sig_x, y, y_logits, z_prior_mean, z_prior_sig = model(images)

        x = images[:,:,:,:3]

        x_recon_loss = tf.reduce_mean(tf.reduce_mean(discretised_logistic_loss(x, x_mean,x_log_scale), axis=[1,2,3]))

        x_kl_loss = kl_divergence_two_gauss(z_mean_x, z_sig_x, z_prior_mean, z_prior_sig)

        py = tf.nn.softmax(y_logits, axis=1)
        y_kl_loss = tf.reduce_mean(tf.reduce_sum(py * (tf.math.log(py + 1e-8) - tf.math.log(1.0/model.y_size)), axis=1))

        x_recon_test_loss(x_recon_loss)
        x_kl_test_loss(x_kl_loss)
        y_kl_test_loss(y_kl_loss)
        return x_mean, x_log_scale, z_x, z_mean_x, z_sig_x, y, y_logits, z_prior_mean, z_prior_sig

    if isinstance(model, LGVae):
        train_step = train_step_lg_vae
        test_step = test_step_lg_vae
    elif isinstance(model, LGGMVae):
        train_step = train_step_lg_gm_vae
        test_step = test_step_lg_gm_vae
    elif isinstance(model, GMVae):
        train_step = train_step_gm_vae
        test_step = test_step_gm_vae

    # Train
    start = time.time()
    for step,train_data in enumerate(train_dataset):
        if config.label:
            images = train_data[0]
        else:
            images = train_data
        train_step(model, images, optimizer)

        if (step%10000==0):
            print('Training time: {:.2f}'.format(time.time()-start))

            start = time.time()
            # Evaluation
            all_labels = []
            all_pred = []
            for test_data in test_dataset:
                if config.label:
                    test_images,labels = test_data[0], test_data[1]
                    
                    if isinstance(model,LGGMVae):
                        (x_mean, x_log_scale, z_x, z_mean_x, z_sig_x, z_x_hat, x_hat_mean, x_hat_log_scale, z_mean_x_hat, z_sig_x_hat,
                            y, y_logits, z_prior_mean, z_prior_sig) = test_step(model, test_images, labels)
                        # cluster_pred = linear_assignment(labels,y_logits)
                        # classifier_cluster_acc(labels,cluster_pred)
                        all_labels += tf.unstack(labels)
                        all_pred += tf.unstack(y_logits)

                    elif isinstance(model,GMVae):
                        (x_mean, x_log_scale, z_x, z_mean_x, z_sig_x, y, y_logits, z_prior_mean, z_prior_sig) = test_step(model, test_images, labels)
                        # cluster_pred = linear_assignment(labels,y_logits)
                        # classifier_cluster_acc(labels,cluster_pred)
                        all_labels += tf.unstack(labels)
                        all_pred += tf.unstack(y_logits)

                    else:
                        test_step(model, test_images, labels)
                else:
                    test_images = test_data
                    test_step(model, test_images)

            if config.label and (isinstance(model,LGGMVae) or isinstance(model,GMVae)):
                all_labels = tf.stack(all_labels)
                all_pred = tf.stack(all_pred)
                cluster_pred = linear_assignment(all_labels,all_pred)
                classifier_cluster_acc(all_labels,cluster_pred)
            print('Testing time: {:.2f}'.format(time.time()-start))

            

            template = 'Training step {}\n\
            X Recon Loss: {:.4f}, X KLD loss: {:.4f}, Total X loss: {:.4f} \n\
            X hat Recon Loss: {:.4f}, X hat KLD loss: {:.4f}, Total X hat loss: {:.4f} \n\
            Test X Recon Loss: {:.4f}, Test X KLD loss: {:.4f}, Test Total X loss: {:.4f} \n\
            Test X hat Recon Loss: {:.4f}, Test X hat KLD loss: {:.4f}, Test Total X hat loss: {:.4f}\n\
            Total KL train loss: {:.4f}, Total KL test loss: {:.4f}\n\
            Classifier recon acc: {:.4f}, Classifier random z_g acc: {:.4f}, Classifier random z_l acc: {:.4f}\n\
            Classifier cluster acc: {:.4f}\n\
            Y KL train loss: {:.4f}, Y KL test loss: {:.4f}'
            print(template.format(step,
                            x_recon_train_loss.result(),
                            x_kl_train_loss.result(),
                            x_recon_train_loss.result() + x_kl_train_loss.result(),

                            x_hat_recon_train_loss.result(),
                            x_hat_kl_train_loss.result(),
                            x_hat_recon_train_loss.result() + x_hat_kl_train_loss.result(),

                            x_recon_test_loss.result(),
                            x_kl_test_loss.result(),
                            x_recon_test_loss.result() + x_kl_test_loss.result(),

                            x_hat_recon_test_loss.result(),
                            x_hat_kl_test_loss.result(),
                            x_hat_recon_test_loss.result() + x_hat_kl_test_loss.result(),
                            total_kl_train_loss.result(), total_kl_test_loss.result(),
                            classifier_recon_acc.result(), classifier_random_z_g_acc.result(), classifier_random_z_l_acc.result(),
                            classifier_cluster_acc.result(),
                            y_kl_train_loss.result(), y_kl_test_loss.result()))


            #VISUALIZATION
            if not isinstance(model, GMVae):
                # FOR LGVAE and LGGMVAE
                visualizer.generate(model, filename='generate_it_' + str(step), filepath = RUN_DIR)
                visualizer.reconstruction_test_lg_vae(model, test_dataset, label=config.label, filename = '_it_'+str(step), filepath = RUN_DIR)
                visualizer.generate_varying_latent(model, vary = 'lower', filename='vary_lower_it_' + str(step), filepath = RUN_DIR)
                visualizer.generate_varying_latent(model, vary = 'upper', filename='vary_upper_it_' + str(step), filepath = RUN_DIR)
                if config.dataset=='svhn':
                    visualizer.style_transfer_test(model, test_dataset, label=config.label, filename = '_it_'+str(step), filepath = RUN_DIR)
                else:
                    visualizer.style_transfer_celeba(model, test_dataset, label=config.label, filename = '_it_'+str(step), filepath = RUN_DIR)

            if config.viz:
                # FOR LGGMVAE ONLY
                if isinstance(model, LGGMVae):
                    visualizer.unseen_cluster_lg(model, test_dataset, label=config.label, filename = '_it_'+str(step), filepath = RUN_DIR)
                    visualizer.generate_cluster(model, vary='zg', filename='generate_cluster_fix_zl_it_' + str(step), filepath = RUN_DIR)
                    visualizer.generate_cluster(model, vary='zg_zl', filename='generate_cluster_it_' + str(step), filepath = RUN_DIR)
                    visualizer.generate_cluster(model, vary='y_zg', filename='generate_multi_cluster_it_' + str(step), filepath = RUN_DIR)
                
            x_recon_train_loss.reset_states()
            x_kl_train_loss.reset_states()
            x_recon_test_loss.reset_states()
            x_kl_test_loss.reset_states()
            total_kl_test_loss.reset_states()
            total_kl_train_loss.reset_states()
            classifier_recon_acc.reset_states()
            classifier_random_z_g_acc.reset_states()
            classifier_random_z_l_acc.reset_states()
            classifier_cluster_acc.reset_states()
            gc.collect()
            start = time.time()
        if (step>=config.training_steps):
            print('Training done!')
            break

    model.save_weights('models/'+RUN_NAME+'.h5')
