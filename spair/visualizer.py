import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import tensorflow as tf
import warnings

# plt.tick_params(top=False, bottom=False, left=False, right=False, labelleft=False, labelbottom=False)

mpl.use('agg')
mpl.rcParams['figure.dpi'] = 300
mpl.rcParams['savefig.dpi'] = 300
warnings.filterwarnings("ignore", module="matplotlib")

def reconstruction_test(model, test_dataset, filename = None, filepath = None, label=True, n = 10):
	#Get a batch of test images
	test_ds = test_dataset.take(n).shuffle(n,seed=1)
	for test_data in test_ds:
		if label:
			images = test_data[0]
		else:
			images = test_data
		x_test = images[:n]
		break
	h,w,channel = x_test.shape[1:4]
	channel = min(3,channel)
	(x_recon, z_what, z_what_mean, z_what_sigma, z_where, z_where_mean, z_where_sigma,
		z_depth, z_depth_mean, z_depth_sigma, z_pres, z_pres_logits, z_pres_pre_sigmoid,
		all_glimpses, obj_recon_unnorm, obj_recon_alpha, obj_full_recon_unnorm, obj_bbox_mask, *more_outputs) = model(x_test)

	num_cells = z_where.shape[1]*z_where.shape[2]
	f, ax = plt.subplots(1, 3)
	ax[0].set_xticks(np.arange(0, h*n, w))
	ax[0].set_yticks(np.arange(0, h*(num_cells+2), w))
	ax[1].set_xticks(np.arange(0, h*n, w))
	ax[1].set_yticks(np.arange(0, h*(num_cells+2), w))
	ax[2].set_xticks(np.arange(0, h*n, w))
	ax[2].set_yticks(np.arange(0, h*(num_cells+2), w))
	# num_channel = x_recon.shape[-1]

	obj_recon = obj_full_recon_unnorm[:,:,:,:,:channel]
	obj_alpha = obj_full_recon_unnorm[:,:,:,:,channel:]

	z_depth = tf.reshape(z_depth, [n, num_cells, 1, 1, 1])
	z_pres = tf.reshape(tf.round(tf.sigmoid(z_pres_logits)), [n, num_cells, 1, 1, 1])

	canvas = np.empty((h*(num_cells+2), w*n, channel))
	canvas_weighted = np.empty((h*(num_cells+2), w*n, channel))
	canvas_weights_only = np.empty((h*(num_cells+2), w*n, channel)) # only weights of that part

	for i in range(n):
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


	if filename is None:
		plt.savefig(filepath + 'x_reconstrcution_test_spair.png')
	else:
		plt.savefig(filepath + 'x_reconstrcution_test' + filename + '.png', dpi=300)
	# plt.close()
	return plt


def reconstruction_bbox(model, test_dataset, filename = None, filepath = None, label=True, n = 10):
	#Get a batch of test images
	test_ds = test_dataset.take(n).shuffle(n,seed=1)
	for test_data in test_ds:
		if label:
			images = test_data[0]
		else:
			images = test_data
		x_test = images[:n]
		break
	h,w,channel = x_test.shape[1:4]
	channel = min(3,channel)
	(x_recon, z_what, z_what_mean, z_what_sigma, z_where, z_where_mean, z_where_sigma,
		z_depth, z_depth_mean, z_depth_sigma, z_pres, z_pres_logits, z_pres_pre_sigmoid,
		all_glimpses, obj_recon_unnorm, obj_recon_alpha, obj_full_recon_unnorm, obj_bbox_mask, *more_outputs) = model(x_test)

	num_cells = z_where.shape[1]*z_where.shape[2]
	# f, ax = plt.subplots(1, 1)
	# ax[0].set_xticks(np.arange(0, h*n, w))
	# ax[0].set_yticks(np.arange(0, h*(num_cells+2), w))
	# num_channel = x_recon.shape[-1]
	# print(obj_bbox_mask.numpy())

	z_pres = tf.reshape(tf.round(tf.sigmoid(z_pres_logits)), [n, num_cells, 1])
	colors = tf.constant([[1.0,1.0,1.0,1.0]])
	obj_bbox_mask = obj_bbox_mask * z_pres 
	x_recon_w_bbox = tf.image.draw_bounding_boxes(x_recon,obj_bbox_mask,colors)
	img_w_bbox = tf.image.draw_bounding_boxes(x_test[:,:,:,:3],obj_bbox_mask,colors)


	canvas = np.empty((h*3, w*n, channel))

	for i in range(n):
		canvas[0:h,i*w:(i+1)*w, :] = images[i,:,:,:3]
		canvas[h:h*2, i*w:(i+1)*w, :] = img_w_bbox[i].numpy().reshape((h,w,channel))
		# canvas[h*2:h*3, i*w:(i+1)*w, :] = x_recon[i].numpy().reshape((h,w,channel))
		canvas[h*2:h*3, i*w:(i+1)*w, :] = x_recon_w_bbox[i].numpy().reshape((h,w,channel))
		

	# ax[0].imshow(np.squeeze(canvas),cmap='gray')
	# ax[0].set_title('reconstruction')
	# ax[0].grid(b=True, which='major', color='#ffffff', linestyle='-')
	# ax[0].tick_params(top=False, bottom=False, left=False, right=False, labelleft=False, labelbottom=False)

	plt.imshow(canvas)



	if filename is None:
		plt.savefig(filepath + 'x_reconstrcution_bbox.png')
	else:
		plt.savefig(filepath + 'x_reconstrcution_bbox' + filename + '.png', dpi=300)
	# plt.close()
	return plt


def glimpses_reconstruction_test(model, test_dataset, filename = None, filepath = None, label=True, n = 10):
	# Glimpses
	for test_data in test_dataset:
		if label:
			images = test_data[0]
		else:
			images = test_data
		x_test = images[:n]
		break
	h,w,channel = x_test.shape[1:4]
	channel = min(3,channel)
	
	(x_recon, z_what, z_what_mean, z_what_sigma, z_where, z_where_mean, z_where_sigma,
		z_depth, z_depth_mean, z_depth_sigma, z_pres, z_pres_logits, z_pres_pre_sigmoid,
		all_glimpses, obj_recon_unnorm, obj_recon_alpha, obj_full_recon_unnorm, obj_bbox_mask, *more_outputs) = model(x_test)

	num_cells = z_where.shape[1]*z_where.shape[2]
	object_size = obj_recon_alpha.shape[2]
	f, ax = plt.subplots(1, 3)
	ax[0].set_xticks(np.arange(0, object_size*n, object_size))
	ax[0].set_yticks(np.arange(0, object_size*num_cells, object_size))

	ax[1].set_xticks(np.arange(0, object_size*n, object_size))
	ax[1].set_yticks(np.arange(0, object_size*num_cells, object_size))

	ax[2].set_xticks(np.arange(0, object_size*n, object_size))
	ax[2].set_yticks(np.arange(0, object_size*num_cells, object_size))	

	# plot glimpses
	canvas_glimpses = np.empty((object_size*num_cells, object_size*n, channel))
	canvas_glimpses_recon = np.empty((object_size*num_cells, object_size*n, channel))
	canvas_glimpses_alpha = np.zeros((object_size*num_cells, object_size*n))

	for i in range(n):
		canvas_glimpses[:,i*object_size:(i+1)*object_size,:] = all_glimpses[i].numpy().reshape((num_cells*object_size,object_size,channel))
		canvas_glimpses_recon[:,i*object_size:(i+1)*object_size,:] = obj_recon_unnorm[i].numpy().reshape((num_cells*object_size,object_size,channel))
		canvas_glimpses_alpha[:,i*object_size:(i+1)*object_size] = obj_recon_alpha[i].numpy().reshape((num_cells*object_size,object_size))


	ax[0].imshow(np.squeeze(canvas_glimpses),cmap='gray')
	ax[0].set_title('Glimpses')
	ax[0].grid(b=True, which='major', color='#ffffff', linestyle='-')
	ax[0].tick_params(top=False, bottom=False, left=False, right=False, labelleft=False, labelbottom=False)
	

	ax[1].imshow(np.squeeze(canvas_glimpses_recon),cmap='gray')
	ax[1].set_title('Glimpses reconstruction')
	ax[1].grid(b=True, which='major', color='#ffffff', linestyle='-')
	ax[1].tick_params(top=False, bottom=False, left=False, right=False, labelleft=False, labelbottom=False)

	ax[2].imshow(np.squeeze(canvas_glimpses_alpha), cmap='viridis') #,cmap='gray'
	ax[2].set_title('Glimpses alpha')
	ax[2].grid(b=True, which='major', color='#ffffff', linestyle='-')
	ax[2].tick_params(top=False, bottom=False, left=False, right=False, labelleft=False, labelbottom=False)


	if filename is None:
		plt.savefig(filepath + 'glimpses.png')
	else:
		plt.savefig(filepath + 'glimpses' + filename + '.png', dpi=300)
	# plt.close()

	return plt

def glimpses_local_reconstruction_test(model, test_dataset, filename = None, filepath = None, label=True, n = 10):
	# Glimpses
	for test_data in test_dataset:
		if label:
			images = test_data[0]
		else:
			images = test_data
		x_test = images[:n]
		break
	h,w,channel = x_test.shape[1:4]
	channel = min(3,channel)
	
	(x_recon, z_what, z_what_mean, z_what_sigma, z_where, z_where_mean, z_where_sigma,
            z_depth, z_depth_mean, z_depth_sigma, z_pres, z_pres_logits, z_pres_pre_sigmoid, all_glimpses,
            obj_recon_unnorm, obj_recon_alpha, obj_full_recon_unnorm, obj_bbox_mask, z_bg, z_bg_mean, z_bg_sig, x_hat_recon, z_l, z_l_mean, z_l_sig, x_hat) = model(x_test)

	num_cells = z_where.shape[1]*z_where.shape[2]
	object_size = obj_recon_alpha.shape[2]
	f, ax = plt.subplots(1, 2)
	ax[0].set_xticks(np.arange(0, object_size*n, object_size))
	ax[0].set_yticks(np.arange(0, object_size*num_cells, object_size))

	ax[1].set_xticks(np.arange(0, object_size*n, object_size))
	ax[1].set_yticks(np.arange(0, object_size*num_cells, object_size))

	# plot glimpses
	canvas_glimpses = np.empty((object_size*num_cells, object_size*n, channel))
	canvas_glimpses_recon = np.empty((object_size*num_cells, object_size*n, channel))

	for i in range(n):
		canvas_glimpses[:,i*object_size:(i+1)*object_size,:] = x_hat[i].numpy().reshape((num_cells*object_size,object_size,channel))
		canvas_glimpses_recon[:,i*object_size:(i+1)*object_size,:] = x_hat_recon[i].numpy().reshape((num_cells*object_size,object_size,channel))


	ax[0].imshow(np.squeeze(canvas_glimpses),cmap='gray')
	ax[0].set_title('Glimpses')
	ax[0].grid(b=True, which='major', color='#ffffff', linestyle='-')
	ax[0].tick_params(top=False, bottom=False, left=False, right=False, labelleft=False, labelbottom=False)
	

	ax[1].imshow(np.squeeze(canvas_glimpses_recon),cmap='gray')
	ax[1].set_title('Glimpses reconstruction')
	ax[1].grid(b=True, which='major', color='#ffffff', linestyle='-')
	ax[1].tick_params(top=False, bottom=False, left=False, right=False, labelleft=False, labelbottom=False)



	if filename is None:
		plt.savefig(filepath + 'glimpses_local.png')
	else:
		plt.savefig(filepath + 'glimpses_local' + filename + '.png', dpi=300)
	# plt.close()

	return plt

def x_hat_reconstruction_test(model, test_dataset, filename = None, filepath = None, label=True, n = 10):
	for test_data in test_dataset:
		if label:
			images = test_data[0]
		else:
			images = test_data
		x_test = images[:n]
		break
	h,w,channel = x_test.shape[1:4]

	channel = min(3,channel)
	(x_recon, z_what, z_what_mean, z_what_sigma, z_where, z_where_mean, z_where_sigma,
            z_depth, z_depth_mean, z_depth_sigma, z_pres, z_pres_logits, z_pres_pre_sigmoid, all_glimpses,
            obj_recon_unnorm, obj_recon_alpha, obj_full_recon_unnorm, obj_bbox_mask, *_, x_hat_recon, z_l, z_l_mean, z_l_sig) = model(x_test)
	canvas_x_hat = np.empty((h*2, w*n, channel))
	for i in range(n):
		canvas_x_hat[0:h,i*w:(i+1)*w,	:] = x_hat_recon[i].numpy().reshape((h,w,channel))
		canvas_x_hat[h:h*2, i*w:(i+1)*w, :] = images[i,:,:,3:]

	plt.figure(figsize=(2*n,2))
	plt.imshow(canvas_x_hat)
	if filename is None:
		plt.savefig(filepath + 'x_hat_reconstrcution_test_lg_vae.png')
	else:
		plt.savefig(filepath + 'x_hat_reconstrcution_test' + filename + '.png')
	plt.close()
	return canvas_x_hat
