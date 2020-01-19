import tensorflow as tf
import argparse
from utils import dotdict
from model import Classifier
import data
import os
import time


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'



def main():
	parser = argparse.ArgumentParser()
	parser.add_argument('--latent_dims', type=int, nargs='?', default=256)
	parser.add_argument('--learning_rate', type=float, nargs='?', default=1e-4)
	parser.add_argument('--dataset', type=str, nargs='?', default = 'svhn')
	parser.add_argument('--epochs', type=int, nargs='?', default = 20)
	parser.add_argument('--model', type=str, nargs='?', default='variational')
	parser.add_argument('--type', type=str, nargs='?', default='conv')
	parser.add_argument('--batch_size', type=int, nargs='?', default=32)
	parser.add_argument('--runs', type=int, nargs='?', default=1)

	args = parser.parse_args()
	BATCH_SIZE = args.batch_size
	AUTOTUNE = tf.data.experimental.AUTOTUNE
	
	config = dotdict({'learning_rate':args.learning_rate, 'latent_dims':args.latent_dims,
						'dataset':args.dataset, 'epochs':args.epochs, 'batch_size':args.batch_size})
	print('Config:',config)

	train_dataset, test_dataset, input_shape = data.get_dataset(dataset=args.dataset, get_label=True)
	train_dataset = train_dataset.concatenate(test_dataset)
	train_dataset = train_dataset.shuffle(20000).batch(BATCH_SIZE).prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
	test_dataset = test_dataset.batch(BATCH_SIZE).prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

	model = Classifier(latent_dims = config.latent_dims, target_shape=10)
	optimizer = tf.keras.optimizers.Adam(learning_rate = config.learning_rate, amsgrad=True)
	print('Model output shape:',model(tf.zeros([8,32,32,3])).shape)
	model.summary()

	print('Training a classifier')

	cross_ent = tf.keras.losses.CategoricalCrossentropy()
	train_loss = tf.keras.metrics.Mean(name='train_loss')
	test_loss = tf.keras.metrics.Mean(name='test_loss')
	train_acc = tf.keras.metrics.CategoricalAccuracy()
	test_acc = tf.keras.metrics.CategoricalAccuracy()


	@tf.function
	def train_step(images, labels, model, optimizer):
		with tf.GradientTape() as tape:
			pred = model(images,training=True)
			loss = tf.nn.softmax_cross_entropy_with_logits(labels,pred)

		gradients = tape.gradient(loss, model.trainable_variables)
		optimizer.apply_gradients(zip(gradients, model.trainable_variables))
		train_loss(loss)
		train_acc(labels,pred)
	  
	@tf.function
	def test_step(images, labels, model):
	  pred = model(images)
	  loss = tf.nn.softmax_cross_entropy_with_logits(labels,pred)

	  test_loss(loss)
	  test_acc(labels, pred)


	for epoch in range(1,config.epochs+1):
		start = time.time()
		for images,labels in train_dataset:
			train_step(images,labels,model,optimizer)

		for images,labels in test_dataset:
			test_step(images,labels,model)

		template = 'Epoch {}, Train Loss: {:.4f}, Train acc {:.4f}, Test Loss: {:.4f}, Test acc: {:.4f}'
		print(template.format(epoch,train_loss.result(),train_acc.result(),test_loss.result(),test_acc.result()))
		print('Epoch time: {:.2f}'.format(time.time()-start))
		train_loss.reset_states()
		test_loss.reset_states()
		test_acc.reset_states()
		train_acc.reset_states()

	model.save_weights('models/' + config.dataset + '_classifier_weights.h5')
	# tf.keras.experimental.export_saved_model(model, 'models/classifier.h5')

	# Create new model
	model = Classifier(latent_dims = config.latent_dims, target_shape=10)
	print('Test untrained model')
	for images,labels in test_dataset:
		pred = model(images)
		loss = tf.nn.softmax_cross_entropy_with_logits(labels,pred)
		test_acc(labels, pred)
	print('Test acc: {:.4f}'.format(test_acc.result()))
	test_acc.reset_states()

	print('Test model with trained weights')
	model.load_weights('models/' + config.dataset + '_classifier_weights.h5')
	for images,labels in test_dataset:
		pred = model(images)
		loss = tf.nn.softmax_cross_entropy_with_logits(labels,pred)
		test_acc(labels, pred)
	print('Test acc: {:.4f}'.format(test_acc.result()))
	test_acc.reset_states()
if __name__=='__main__':
	main()
