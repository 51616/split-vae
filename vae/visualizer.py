import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import tensorflow as tf
from model import LGGMVae
from scipy.io import loadmat
from collections import defaultdict

mpl.use('agg')
mpl.rcParams['figure.dpi'] = 300
mpl.rcParams['savefig.dpi'] = 300

def reconstruction_test_lg_vae(model, test_dataset, label=True, filename = None, filepath = None, n = 10):
	#Get a batch of test images

	for test_data in test_dataset:
		if label:
			images = test_data[0]
		else:
			images = test_data
		x_test = images[:n]
		break
	global h, w, channel
	h,w = x_test.shape[1:3]
	channel = 3
	# x_recon, z_mean_x, z_log_var_x, x_hat_recon, z_mean_x_hat, z_log_var_x_hat = model(x_test)
	z_x, z_x_hat = model.encode(x_test)
	x_recon, x_hat_recon = model.decode(z_x, z_x_hat, True)

	canvas_x = np.empty((h*2, w*n, channel))
	for i in range(n):
		canvas_x[0:h,i*w:(i+1)*w,	:] = x_recon[i].numpy().reshape((h,w,channel))
		canvas_x[h:h*2, i*w:(i+1)*w, :] = (images[i,:,:,:3]+1)*0.5

	plt.figure(figsize=(2*n,2))
	plt.imshow(canvas_x)
	if filename is None:
		plt.savefig(filepath + 'x_reconstruction_test_lg_vae.png')
	else:
		plt.savefig(filepath + 'x_reconstruction_test' + filename + '.png')
	plt.close()

	canvas_x_hat = np.empty((h*2, w*n, channel))
	for i in range(n):
		canvas_x_hat[0:h,i*w:(i+1)*w,	:] = x_hat_recon[i].numpy().reshape((h,w,channel))
		canvas_x_hat[h:h*2, i*w:(i+1)*w, :] = (images[i,:,:,3:]+1)*0.5

	plt.figure(figsize=(2*n,2))
	plt.imshow(canvas_x_hat)
	if filename is None:
		plt.savefig(filepath + 'x_hat_reconstruction_test_lg_vae.png')
	else:
		plt.savefig(filepath + 'x_hat_reconstruction_test' + filename + '.png')
	plt.close()
	return canvas_x, canvas_x_hat

def style_transfer_test(model, test_dataset, label=True, filename = None, filepath = None, n = 10):
	#Hand-picked samples
	idx = tf.constant([26,101,3025,3129,3182,3233,3547,3695,10462,10471,10601,10608,16171,16289,16593,16801,101,326,333,798,841,1189,6186,2651,1437,1826,5536],dtype=tf.int32)
	test_data = loadmat('data/SVHN/test_32x32.mat')['X'].transpose((3,0,1,2)).astype(np.float32)/255.*2 - 1

	rand_x_idx = tf.random.shuffle(idx)[:n]
	rand_x_hat_idx = tf.random.shuffle(idx)[:n]

	x = test_data[rand_x_idx.numpy()]
	x_hat = test_data[rand_x_hat_idx.numpy()]
	x_test = tf.concat([x,x_hat],axis=-1)

	z_x, z_x_hat = model.encode(x_test)
	x_recon, x_hat_recon = model.decode(z_x, z_x_hat, True)

	canvas_1 = np.empty((h*3, w*n, channel))
	for i in range(n):
		canvas_1[0:h, i*w:(i+1)*w, :] = (x_test[i,:,:,:3]+1)*0.5
		canvas_1[h:h*2, i*w:(i+1)*w, :] = (x_test[i,:,:,3:]+1)*0.5
		canvas_1[h*2:h*3,i*w:(i+1)*w,	:] = x_recon[i].numpy().reshape((h,w,channel))

	plt.imshow(canvas_1)
	if filename is None:
		plt.savefig(filepath + 'style_transfer.png')
	else:
		plt.savefig(filepath + 'style_transfer' + filename + '.png')
	plt.close()

	return canvas_1


def style_transfer_celeba(model, test_dataset, label=True, filename = None, filepath = None, n = 10):

	for test_data in test_dataset:
		if label:
			images = test_data[0]
		else:
			images = test_data
		x_test = images # [:n]
		break
	global h, w, channel
	h,w = x_test.shape[1:3]
	channel = 3


	x = x_test[:n,:,:,:3].numpy()
	x_hat = x_test[n:2*n,:,:,:3].numpy()
	x_2 = tf.concat([x,x_hat],axis=-1)

	x_aug = tf.concat([x_test[:n],x_2],axis=0)
	z_x, z_x_hat = model.encode(x_aug)
	x_recon, x_hat_recon = model.decode(z_x, z_x_hat, True)

	canvas_1 = np.empty((h*4, w*n, channel))
	for i in range(n):
		canvas_1[0:h, i*w:(i+1)*w, :] = (x_aug[i,:,:,:3]+1)*0.5
		canvas_1[h:h*2, i*w:(i+1)*w, :] = (x_aug[i+n,:,:,3:]+1)*0.5
		canvas_1[h*2:h*3,i*w:(i+1)*w,	:] = x_recon[i].numpy().reshape((h,w,channel))
		canvas_1[h*3:h*4,i*w:(i+1)*w,	:] = x_recon[n+i].numpy().reshape((h,w,channel))

	# plt.figure(figsize=(2*n,2))
	plt.imshow(canvas_1)
	if filename is None:
		plt.savefig(filepath + 'style_transfer_celeba.png')
	else:
		plt.savefig(filepath + 'style_transfer_celeba' + filename + '.png')
	plt.close()

	return canvas_1


def plot_latent_dims(model, dataset, variational = False):
	z_list = [[] for dim in range(model.latent_dims)]

	for images,_ in dataset:
		if variational:
			z,_,_ = model.encode(images)
		else:
			z = model.encode(images)
		z = z.numpy()
		for dim in range(z.shape[1]):
			z_list[dim].extend(z[:,dim])
	plt.figure()
	plt.scatter(z_list[0],z_list[1],s=1)
	if variational:
		plt.savefig('output/2d_latent_var.png')
	else:
		plt.savefig('output/2d_latent_det.png')
	plt.close()
	for i,z in enumerate(z_list):
		plt.figure()
		plt.hist(z)
		if variational:
			plt.savefig('output/latent_var_'+str(i)+'.png')
		else:
			plt.savefig('output/latent_det_'+str(i)+'.png')
		plt.close()

def generate(model, filename = None, filepath = None):
	if isinstance(model,LGGMVae): #within a cluster
		y = tf.one_hot(tf.random.uniform(shape = [1], minval = 0, maxval = model.y_size, dtype = tf.int32), depth = model.y_size, dtype=tf.float32)
		z_prior_mean, z_prior_sig = model.encode_y(y)
		z_g = tf.random.normal(shape = [100, model.global_latent_dims],mean=z_prior_mean, stddev=z_prior_sig)
		z_l = tf.random.normal(shape = [100, model.local_latent_dims])
	else:
		z_g = tf.random.normal(shape=[100, model.global_latent_dims])
		z_l = tf.random.normal(shape=[100, model.local_latent_dims])

	x_generated,_ = model.decode(z_g,z_l,True)
	h,w, channel = model.image_shape[1:4]

	n = np.sqrt(100).astype(np.int32)
	canvas = np.empty((h*n, w*n, 3))
	for i in range(n):
	    for j in range(n):
	        canvas[i*h:(i+1)*h, j*w:(j+1)*w, :] = x_generated[i*n+j].numpy().reshape(h, w, 3)

	plt.figure(figsize=(8, 8))
	plt.imshow(canvas, cmap='gray')
	if filename is None:
		plt.savefig(filepath + 'generated_image.png')
	else:
		plt.savefig(filepath + filename + '.png')
	plt.close()
	return canvas

def generate_traverse(model):
	if model.latent_dims != 2:
		raise NotImplementedError('Implemented for 2D latent only')
	z1_list = z2_list = np.linspace(-3,3,30)
	z_list = [[z1,z2] for z1 in z1_list for z2 in z2_list]
	generated_img = model.decode(tf.convert_to_tensor(z_list))
	canvas = np.empty((h*30, w*30))
	n = 30
	for i in range(n):
	    for j in range(n):
	        canvas[i*h:(i+1)*h, j*w:(j+1)*w] = generated_img[i*n+j, :].numpy().reshape(h, w)

	plt.figure(figsize=(8, 8))
	plt.imshow(canvas, cmap='gray')
	plt.savefig('output/latent_space.png')
	plt.close()


def generate_varying_latent(model, vary , filename = None, filepath = None):
	# assume top half of z is z_g and bottom is z_l
	z_prior_mean, z_prior_sig = 0, 1
	if isinstance(model,LGGMVae):
		y = tf.one_hot(tf.random.uniform(shape = [1], minval = 0, maxval = model.y_size, dtype = tf.int32), depth = model.y_size, dtype=tf.float32)
		z_prior_mean, z_prior_sig = model.encode_y(y)

		#z2 is global
		if vary=='lower':
			z1 = tf.random.normal(shape = [100,model.local_latent_dims])
			z2 = tf.random.normal(shape=[1, model.global_latent_dims],mean=z_prior_mean,stddev=z_prior_sig) #100 samples of z2s
			z2 = tf.tile(z2, [100,1])  # repeat z1 100 times
			x_generated, x_hat_generated = model.decode(z2,z1, True)
		elif vary=='upper':
			z1 = tf.random.normal(shape = [1,model.local_latent_dims])
			z1 = tf.tile(z1, [100,1])  # repeat z1 100 times
			z2 = tf.random.normal(shape=[100, model.global_latent_dims],mean=z_prior_mean,stddev=z_prior_sig) #100 samples of z2s
			x_generated, x_hat_generated = model.decode(z2,z1, True)


	else:
		# z2 = tf.random.normal(shape = [100, model.latent_dims//2])
		# z1 = tf.random.normal(shape = [1,model.latent_dims//2])
		# z1 = tf.tile(z1, [100,1]) # .transpose(2,0,1).squeeze(1) # repeat z1 100 times
		# z2 is global
		if vary == 'lower':
			# z = np.concatenate([z1,z2], axis=1)
			z2 = tf.random.normal(shape = [1, model.global_latent_dims])
			z1 = tf.random.normal(shape = [100,model.local_latent_dims])
			z2 = tf.tile(z2, [100,1])
			x_generated, x_hat_generated = model.decode(z2,z1, True)
		elif vary == 'upper':
			# z = np.concatenate([z2,z1], axis=1)
			z2 = tf.random.normal(shape = [100, model.global_latent_dims])
			z1 = tf.random.normal(shape = [1,model.local_latent_dims])
			z1 = tf.tile(z1, [100,1])
			x_generated,_ = model.decode(z2,z1, True)

	
	h,w,channel = model.image_shape[1:4]
	n = np.sqrt(100).astype(np.int32)
	canvas_x = np.empty((h*n, w*n, 3))
	for i in range(n):
	    for j in range(n):
	        canvas_x[i*h:(i+1)*h, j*w:(j+1)*w,:] = x_generated[i*n+j].numpy().reshape(h, w, 3)

	plt.figure(figsize=(8, 8))
	plt.imshow(canvas_x, cmap='gray')
	if filename is None:
		plt.savefig(filepath + 'generate_varying_latent_' + vary + '.png')
	else:
		plt.savefig(filepath + filename + '.png')
	plt.close()

	if vary == 'lower':
		canvas_x_hat = np.empty((h*n, w*n, 3))
		for i in range(n):
		    for j in range(n):
		        canvas_x_hat[i*h:(i+1)*h, j*w:(j+1)*w,:] = x_hat_generated[i*n+j].numpy().reshape(h, w, 3)

		plt.figure(figsize=(8, 8))
		plt.imshow(canvas_x_hat, cmap='gray')
		if filename is None:
			plt.savefig(filepath + 'generate_x_hat_' + vary + '.png')
		else:
			plt.savefig(filepath + 'x_hat_' + filename + '.png')
		plt.close()

		return canvas_x, canvas_x_hat
	return canvas_x

def generate_cluster(model, vary, filename = None, filepath = None):
	# For LGGMVAE
	# Default to only vary zg
	y = tf.one_hot(tf.random.uniform(shape = [1], minval = 0, maxval = model.y_size, dtype = tf.int32), depth = model.y_size, dtype=tf.float32)
	z_prior_mean, z_prior_sig = model.encode_y(y)
	if vary=='zg_zl':
		z_g = tf.random.normal(shape = [10, model.global_latent_dims],mean=z_prior_mean, stddev=z_prior_sig)
		z_g = tf.reshape(tf.tile(z_g,tf.convert_to_tensor([1,10])),[10*z_g.shape[0],z_g.shape[1]]) #repeat each elem 10 times
		z_l = tf.random.normal(shape = [10, model.local_latent_dims])
		z_l = tf.tile(z_l,tf.convert_to_tensor([10,1])) # repeat z_l 10 times

	elif vary=='zg':
		z_g = tf.random.normal(shape = [100, model.global_latent_dims],mean=z_prior_mean, stddev=z_prior_sig)
		z_l = tf.random.normal(shape = [1,model.local_latent_dims])
		z_l = tf.tile(z_l, [100,1]) # .transpose(2,0,1).squeeze(1) # repeat z1 100 times

	elif vary=='y_zg':
		y = tf.one_hot(tf.random.shuffle(tf.range(0,model.y_size))[:10], depth=model.y_size, dtype=tf.float32)
		z_prior_mean, z_prior_sig = model.encode_y(y)
		z_g = []
		for mean, sig in zip(z_prior_mean, z_prior_sig):
			z_g.append(tf.random.normal(shape = [10, model.global_latent_dims],mean=mean, stddev=sig))
		z_g = tf.reshape(tf.stack(z_g),[100,model.global_latent_dims])
		z_l = tf.random.normal(shape = [1,model.local_latent_dims])
		z_l = tf.tile(z_l, [100,1])

	x_generated,_ = model.decode(z_g,z_l)

	h,w,channel = model.image_shape[1:4]
	n = np.sqrt(100).astype(np.int32)
	canvas = np.empty((h*n, w*n, 3))
	for i in range(n):
	    for j in range(n):
	        canvas[i*h:(i+1)*h, j*w:(j+1)*w,:] = x_generated[i*n+j].numpy().reshape(h, w, 3)

	plt.figure(figsize=(8, 8))
	plt.imshow(canvas, cmap='gray')
	if filename is None:
		plt.savefig(filepath + 'generate_cluster_' + vary + '.png')
	else:
		plt.savefig(filepath + filename + '.png')
	plt.close()
	return canvas



def unseen_cluster_lg(model, test_dataset, label=True, filename = None, filepath = None, n = 10):

	cluster_dict = defaultdict(list)

	for i,test_data in enumerate(test_dataset):
		if label:
			images = test_data[0]
		else:
			images = test_data
		x_test = images
		_, y_logits = model.get_y(x_test)
		y = tf.nn.softmax(y_logits, axis=1)
		cluster = tf.argmax(y,axis=1)

		for c in range(model.y_size):
			cluster_samples = tf.unstack(x_test[cluster==c][:,:,:,:3])
			score = tf.unstack(y[cluster==c][:,c])
			if len(score)>0:
				cluster_dict[c] += zip(score,cluster_samples)
	global h, w, channel
	h,w = x_test.shape[1:3]
	channel = 3

	for c in range(model.y_size):
		if len(cluster_dict[c])>0:
			# print(cluster_dict[c])
			cluster_dict[c].sort(key=lambda x: x[0], reverse=True)
			cluster_samples = tf.stack([p[1] for p in cluster_dict[c][:7]])
			num_samples = cluster_samples.shape[0]
			# print(f'Cluster: {c}, samples: {num_samples}')
			# if num_samples!=0: #draw only existing cluster
			canvas = np.empty((h, w*num_samples, channel))
			for j in range(num_samples):
				canvas[0:h, j*w:(j+1)*w, :] = (cluster_samples[j]+1)*0.5
			plt.figure()
			plt.imshow(canvas)
			plt.savefig(filepath + 'unseen_cluster_' + filename + '_' + str(c) + '.png')
			plt.close()

	# discrete_y = tf.one_hot(tf.argmax(y_logits,axis=1),model.y_size)


	# z_prior_mean, z_prior_sig = model.encode_y(discrete_y)
	# _, z_x_hat = model.encode(x_test)
	# z_x = []
	# for mean, sig in zip(z_prior_mean, z_prior_sig):
	# 	z_x.append(tf.random.normal(shape = [10, model.global_latent_dims],mean=mean, stddev=sig))

	# z_x = tf.reshape(tf.stack(z_x,axis=0),(10*n,model.global_latent_dims))
	# z_x_hat = tf.reshape(tf.tile(z_x_hat,[1,10]),[10*z_x_hat.shape[0],z_x_hat.shape[1]]) # repeat 10 times
	# x_recon, x_hat_recon = model.decode(z_x, z_x_hat, True)

	# canvas_1 = np.empty((h*n+h, w*n, channel))
	# for i in range(n):
	# 	canvas_1[0:h, i*w:(i+1)*w, :] = (x_test[i,:,:,:3]+1)*0.5
	# 	canvas_1[h:h*n+h,i*w:(i+1)*w,	:] = x_recon[i*n:(i+1)*n].numpy().reshape((h*n,w,channel))

	# plt.figure(figsize=(2*n,2))
	# plt.imshow(canvas_1)
	# if filename is None:
	# 	plt.savefig(filepath + 'unseen_cluster.png')
	# else:
	# 	plt.savefig(filepath + 'unseen_cluster' + filename + '.png')
	# plt.close()

	return canvas

def unseen_cluster_lg_svhn(model, test_dataset, label=True, filename = None, filepath = None, n = 10):

	cluster = defaultdict(list)

	idx = tf.constant([26,101,3025,3129,3182,3233,3547,3695,10462,10471,10601,10608,16171,16289,
		16593,16801,101,326,333,798,841,1189,6186,2651,1437,1826,5536,0,3040,3065,3106,3292,3762,
		10427,10814,16338,16505,16606,16655,16875,16880],dtype=tf.int32)
	test_data = loadmat('data/SVHN/test_32x32.mat')['X'].transpose((3,0,1,2)).astype(np.float32)/255.*2 - 1
	# rand_x_idx = tf.random.shuffle(idx)[:n]

	# x_test = test_data[rand_x_idx.numpy()]
	x_test = test_data[idx]
	h,w, channel = x_test.shape[1:4]
	x_test = tf.tile(x_test,[1,1,1,2])
	
	y, y_logits = model.get_y(x_test)
	cluster = tf.argmax(y_logits,axis=1)

	for i in range(model.y_size):
		cluster_samples = x_test[cluster==i]
		num_samples = cluster_samples.shape[0]
		if num_samples!=0: #draw only existing cluster
			canvas = np.empty((h, w*num_samples, channel))
			for j in range(num_samples):
				canvas[0:h, j*w:(j+1)*w, :] = (cluster_samples[j,:,:,:3]+1)*0.5
			plt.figure()
			plt.imshow(canvas)
			plt.savefig(filepath + 'unseen_cluster_' + filename + '_' + str(i) + '.png')
			plt.close()

	# y, y_logits = model.get_y(x_test)
	# discrete_y = tf.one_hot(tf.argmax(y_logits,axis=1),model.y_size)
	# z_prior_mean, z_prior_sig = model.encode_y(discrete_y)
	# _, z_x_hat = model.encode(x_test)
	# z_x = []
	# for mean, sig in zip(z_prior_mean, z_prior_sig):
	# 	z_x.append(tf.random.normal(shape = [10, model.global_latent_dims],mean=mean, stddev=sig))

	# z_x = tf.reshape(tf.stack(z_x,axis=0),(10*n,model.global_latent_dims))
	# z_x_hat = tf.reshape(tf.tile(z_x_hat,[1,10]),[10*z_x_hat.shape[0],z_x_hat.shape[1]]) # repeat 10 times
	# x_recon, x_hat_recon = model.decode(z_x, z_x_hat, True)

	# canvas_1 = np.empty((h*n+h, w*n, channel))
	# for i in range(n):
	# 	canvas_1[0:h, i*w:(i+1)*w, :] = (x_test[i,:,:,:3]+1)*0.5
	# 	canvas_1[h:h*n+h,i*w:(i+1)*w,	:] = x_recon[i*n:(i+1)*n].numpy().reshape((h*n,w,channel))

	# # plt.figure(figsize=(2*n,2))
	# plt.imshow(canvas_1)
	# if filename is None:
	# 	plt.savefig(filepath + 'unseen_cluster.png')
	# else:
	# 	plt.savefig(filepath + 'unseen_cluster' + filename + '.png')
	# plt.close()

	return canvas

def unseen_cluster(model, test_dataset, label=True, filename = None, filepath = None, n = 10):


	for test_data in test_dataset:
		if label:
			images = test_data[0]
		else:
			images = test_data
		x_test = images[:n]
		break
	global h, w, channel
	h,w = x_test.shape[1:3]
	channel = 3

	y, y_logits = model.get_y(x_test)
	discrete_y = tf.one_hot(tf.argmax(y_logits,axis=1),model.y_size)
	z_prior_mean, z_prior_sig = model.encode_y(discrete_y)
	z_x = []
	for mean, sig in zip(z_prior_mean, z_prior_sig):
		z_x.append(tf.random.normal(shape = [10, model.global_latent_dims],mean=mean, stddev=sig))

	z_x = tf.reshape(z_x,(10*n,model.global_latent_dims))
	x_recon = model.decode(z_x, True)

	canvas_1 = np.empty((h*n+h, w*n, channel))
	for i in range(n):
		canvas_1[0:h, i*w:(i+1)*w, :] = (x_test[i,:,:,:3]+1)*0.5
		canvas_1[h:h*n+h,i*w:(i+1)*w,	:] = x_recon[i*n:(i+1)*n].numpy().reshape((h*n,w,channel))

	# plt.figure(figsize=(2*n,2))
	plt.imshow(canvas_1)
	if filename is None:
		plt.savefig(filepath + 'unseen_cluster.png')
	else:
		plt.savefig(filepath + 'unseen_cluster' + filename + '.png')
	plt.close()

	return canvas_1

def unseen_cluster_svhn(model, test_dataset, label=True, filename = None, filepath = None, n = 10):


	idx = tf.constant([26,101,3025,3129,3182,3233,3547,3695,10462,10471,10601,10608,16171,16289,
		16593,16801,101,326,333,798,841,1189,6186,2651,1437,1826,5536,0,3040,3065,3106,3292,3762,
		10427,10814,16338,16505,16606,16655,16875,16880],dtype=tf.int32)
	test_data = loadmat('data/SVHN/test_32x32.mat')['X'].transpose((3,0,1,2)).astype(np.float32)/255.*2 - 1
	rand_x_idx = tf.random.shuffle(idx)[:n]

	x_test = test_data[rand_x_idx.numpy()]

	h,w, channel = x_test.shape[1:4]

	y, y_logits = model.get_y(x_test)
	discrete_y = tf.one_hot(tf.argmax(y_logits,axis=1),model.y_size)
	z_prior_mean, z_prior_sig = model.encode_y(discrete_y)
	z_x = []
	for mean, sig in zip(z_prior_mean, z_prior_sig):
		z_x.append(tf.random.normal(shape = [10, model.global_latent_dims],mean=mean, stddev=sig))

	z_x = tf.reshape(z_x,(10*n,model.global_latent_dims))
	x_recon = model.decode(z_x, True)

	canvas_1 = np.empty((h*n+h, w*n, channel))
	for i in range(n):
		canvas_1[0:h, i*w:(i+1)*w, :] = (x_test[i,:,:,:3]+1)*0.5
		canvas_1[h:h*n+h,i*w:(i+1)*w,	:] = x_recon[i*n:(i+1)*n].numpy().reshape((h*n,w,channel))

	# plt.figure(figsize=(2*n,2))
	plt.imshow(canvas_1)
	if filename is None:
		plt.savefig(filepath + 'unseen_cluster.png')
	else:
		plt.savefig(filepath + 'unseen_cluster' + filename + '.png')
	plt.close()

	return canvas_1