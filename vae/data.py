import tensorflow as tf
import os
import numpy as np
from pathlib import Path
from scipy.io import loadmat
from glob import glob
import wget
import zipfile
from utils import download_file_from_google_drive

def get_dataset(dataset='mnist', get_label=False):
	if dataset.upper()=='SVHN':
		return get_svhn(get_label)
	elif dataset.upper()=='SVHN_NO_EXTRA':
		return get_svhn(get_label,extra=False)
	elif dataset.upper()=='CELEBA128':
		return get_celeba_tfrec(size=128)
	elif dataset.upper()=='CELEBA64':
		return get_celeba_tfrec(size=64)
	elif dataset.upper()=='CELEBAHQ':
		return get_celebahq_tfrec()
	else:
		raise NotImplementedError('Dataset doesn\'t exit')

def get_svhn(get_label=False,extra=True):
	data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data/SVHN/')
	training_data_path = os.path.join(data_path, 'train_32x32.mat')
	test_data_path = os.path.join(data_path, 'test_32x32.mat')
	extra_data_path = os.path.join(data_path, 'extra_32x32.mat')

	if not os.path.exists(data_path):
		print('data folder doesn\'t exist, create data folder')
		# os.mkdir('data')
		# os.mkdir(data_path)
		Path(data_path).mkdir(parents=True, exist_ok=True)
	if not glob(training_data_path):
		print('Downloading SVHN training dataset')
		wget.download('http://ufldl.stanford.edu/housenumbers/train_32x32.mat','data/SVHN/train_32x32.mat')
	if not glob(extra_data_path):
		print('Downloading SVHN extra dataset')
		wget.download('http://ufldl.stanford.edu/housenumbers/extra_32x32.mat','data/SVHN/extra_32x32.mat')
	if not glob(test_data_path):
		print('Downloading SVHN test dataset')
		wget.download('http://ufldl.stanford.edu/housenumbers/test_32x32.mat','data/SVHN/test_32x32.mat')

	train_data = loadmat(training_data_path)
	x_train, y_train = train_data['X'].transpose((3,0,1,2)), train_data['y']
	extra_data = loadmat(extra_data_path)
	x_extra, y_extra = extra_data['X'].transpose((3,0,1,2)), extra_data['y']

	test_data = loadmat(test_data_path)
	x_test, y_test = test_data['X'].transpose((3,0,1,2)), test_data['y']

	x_train, x_test = (x_train / 255.0 * 2 - 1).astype(np.float32), (x_test / 255.0 * 2 - 1).astype(np.float32)
	x_extra = (x_extra / 255.0 * 2 - 1).astype(np.float32)

	if get_label:
		y_train = tf.squeeze(tf.one_hot(y_train-1,10)) #0 onehot at the last index
		y_extra = tf.squeeze(tf.one_hot(y_extra-1,10))
		y_test = tf.squeeze(tf.one_hot(y_test-1,10))
		if extra:
			train_dataset = tf.data.Dataset.from_tensor_slices((np.concatenate([x_train,x_extra]),np.concatenate([y_train,y_extra])))
		else:
			train_dataset = tf.data.Dataset.from_tensor_slices((x_train,y_train))
		test_dataset = tf.data.Dataset.from_tensor_slices((x_test,y_test))
	else:
		if extra:
			train_dataset = tf.data.Dataset.from_tensor_slices(np.concatenate([x_train,x_extra]))
		else:
			train_dataset = tf.data.Dataset.from_tensor_slices(x_train)
		test_dataset = tf.data.Dataset.from_tensor_slices(x_test)
		
	print("Training Set", x_train.shape, y_train.shape)
	print("Extra Set", x_extra.shape, y_extra.shape)
	print("Test Set", x_test.shape, y_test.shape)

	return train_dataset, test_dataset, [-1] + [shape for shape in x_train.shape[1:]]

def create_celeba_tfrec():
	def load_and_preprocess_image(path):
		image = tf.io.read_file(path)
		return preprocess_image(image)

	def preprocess_image(image):
		image = tf.image.decode_jpeg(image, channels=3)
		image = tf.image.resize_with_crop_or_pad(image,178,178)
		image = tf.image.resize(image, [64, 64])
		image = image/255. * 2 -1
		return image

	files = glob('data/celeba/img_align_celeba/*')
	test_files = files[:len(files)//10]
	train_files = files[len(files)//10:]

	path_train = tf.data.Dataset.from_tensor_slices(train_files).map(load_and_preprocess_image).map(tf.io.serialize_tensor)
	path_test = tf.data.Dataset.from_tensor_slices(test_files).map(load_and_preprocess_image).map(tf.io.serialize_tensor)

	tfrec = tf.data.experimental.TFRecordWriter('data/celeba/train_64x64.tfrec')
	tfrec.write(path_train)

	tfrec = tf.data.experimental.TFRecordWriter('data/celeba/test_64x64.tfrec')
	tfrec.write(path_test)

def get_celeba_tfrec(size):
	data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data/celeba/')
	zip_data_path = os.path.join(data_path, 'img_align_celeba.zip')
	raw_data_path = os.path.join(data_path, 'img_align_celeba/')
	training_data_path = os.path.join(data_path, 'train_64x64.tfrec')
	test_data_path = os.path.join(data_path, 'test_64x64.tfrec')

	if not os.path.exists(data_path):
		print('data folder doesn\'t exist, create data folder')
		Path(data_path).mkdir(parents=True, exist_ok=True)
	if not glob(zip_data_path):
		print('Downloading CelebA dataset')
		download_file_from_google_drive('0B7EVK8r0v71pZjFTYXZWM3FlRnM', zip_data_path)
	if not glob(raw_data_path):
		print('Extracting CelebA dataset')
		with zipfile.ZipFile(zip_data_path, 'r') as zip_ref:
			zip_ref.extractall('data/celeba/')
	if not glob(training_data_path) or not glob(test_data_path):
		print('Creating CelebA TFrecord')
		create_celeba_tfrec()
	
	def parse(x):
		result = tf.io.parse_tensor(x, out_type=tf.float32)
		result = tf.reshape(result, [size, size, 3])
		return result
	if size==128:
		train_dataset = tf.data.TFRecordDataset('data/celeba/train_128x128.tfrec').map(parse, num_parallel_calls=tf.data.experimental.AUTOTUNE)
		test_dataset = tf.data.TFRecordDataset('data/celeba/test_128x128.tfrec').map(parse, num_parallel_calls=tf.data.experimental.AUTOTUNE)
	elif size==64:
		train_dataset = tf.data.TFRecordDataset('data/celeba/train_64x64.tfrec').map(parse, num_parallel_calls=tf.data.experimental.AUTOTUNE)
		test_dataset = tf.data.TFRecordDataset('data/celeba/test_64x64.tfrec').map(parse, num_parallel_calls=tf.data.experimental.AUTOTUNE)

	return train_dataset, test_dataset, [-1,size,size,3]


def create_celebahq_tfrec():
	def load_and_preprocess_image(path):
		image = tf.io.read_file(path)
		return preprocess_image(image)

	def preprocess_image(image):
		image = tf.image.decode_jpeg(image, channels=3)
		# image = tf.image.resize_with_crop_or_pad(image,178,178)
		image = tf.image.resize(image, [256, 256])
		image = image/255. * 2 -1
		return image

	files = glob('data/celebahq/CelebAMask-HQ/CelebA-HQ-img/*.jpg')
	test_files = files[:len(files)//10]
	train_files = files[len(files)//10:]

	path_train = tf.data.Dataset.from_tensor_slices(train_files).map(load_and_preprocess_image).map(tf.io.serialize_tensor)
	path_test = tf.data.Dataset.from_tensor_slices(test_files).map(load_and_preprocess_image).map(tf.io.serialize_tensor)

	tfrec = tf.data.experimental.TFRecordWriter('data/celebahq/CelebAMask-HQ/train_256x256.tfrec')
	tfrec.write(path_train)

	tfrec = tf.data.experimental.TFRecordWriter('data/celebahq/CelebAMask-HQ/test_256x256.tfrec')
	tfrec.write(path_test)

def get_celebahq_tfrec():
	size = 256
	data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data/celebahq/')
	zip_data_path = os.path.join(data_path, 'CelebAMask-HQ.zip')
	raw_data_path = os.path.join(data_path, 'CelebAMask-HQ/CelebA-HQ-img/')
	training_data_path = os.path.join(data_path, 'CelebAMask-HQ/train_256x256.tfrec')
	test_data_path = os.path.join(data_path, 'CelebAMask-HQ/test_256x256.tfrec')

	if not os.path.exists(data_path):
		print('data folder doesn\'t exist, create data folder')
		Path(data_path).mkdir(parents=True, exist_ok=True)
	if not glob(zip_data_path):
		import gdown
		print('Downloading CelebA dataset')
		gdown.download(id='1badu11NqxGf6qM3PTTooQDJvQbejgbTv',output=zip_data_path, quiet=False)
	if not glob(raw_data_path):
		print('Extracting CelebA dataset')
		with zipfile.ZipFile(zip_data_path, 'r') as zip_ref:
			zip_ref.extractall('data/celebahq/')
	if not glob(training_data_path) or not glob(test_data_path):
		print('Creating CelebA TFrecord')
		create_celebahq_tfrec()
	
	def parse(x):
		result = tf.io.parse_tensor(x, out_type=tf.float32)
		result = tf.reshape(result, [size, size, 3])
		return result
	train_dataset = tf.data.TFRecordDataset('data/celebahq/CelebAMask-HQ/train_256x256.tfrec').map(parse, num_parallel_calls=tf.data.experimental.AUTOTUNE)
	test_dataset = tf.data.TFRecordDataset('data/celebahq/CelebAMask-HQ/test_256x256.tfrec').map(parse, num_parallel_calls=tf.data.experimental.AUTOTUNE)
	
	return train_dataset, test_dataset, [-1,size,size,3]

if __name__=='__main__':
	get_svhn()
	get_celeba_tfrec(64)
 