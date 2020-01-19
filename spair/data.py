import tensorflow as tf
import tensorflow_addons as tfa
import numpy as np
from random import randint, choice, shuffle
import os
from glob import glob
from PIL import Image
import math
from pathlib import Path
import matplotlib.pyplot as plt


def load_cub_masked():
    train_images = np.load('data/cub_train_seg_14x14_pad_20_masked.npy')
    test_images = np.load('data/cub_test_seg_14x14_pad_20_masked.npy')
    return train_images, None, test_images, None

def calculateIntersection(a0, a1, b0, b1):
    if a0 >= b0 and a1 <= b1: # Contained
        intersection = a1 - a0
    elif a0 < b0 and a1 > b1: # Contains
        intersection = b1 - b0
    elif a0 < b0 and a1 > b0: # Intersects right
        intersection = a1 - b0
    elif a1 > b1 and a0 < b1: # Intersects left
        intersection = b1 - a0
    else: # No intersection (either side)
        intersection = 0
    return intersection

def calculate_overlap(rand_x,rand_y,drawn_boxes):
    # check if a new box is overlapped with drawn boxes more than 15% or not
    for box in drawn_boxes:
        x,y = box[0], box[1]
        if calculateIntersection(rand_x,rand_x+14,x,x+14) * calculateIntersection(rand_y,rand_y+14,y,y+14) / 14**2 > 0.15:
            return True
    return False

class MultiCUB:
    def __init__(self, data, reshape=True):
        self.num_channel = data[0].shape[-1]
        self.train_x = data[0]
        self.train_y = data[1]
        self.test_x = data[2]
        self.test_y = data[3]
        if reshape:
            self.train_x = tf.image.resize(self.train_x,(14,14)).numpy() #[28,28] -> [14,14]
            self.test_x = tf.image.resize(self.test_x,(14,14)).numpy()
        self.bg_list = glob('data/kylberg/*.png')

        #triad hard
        self.train_colors_triad = [(195,135,255),(193,255,135),(255,165,135),(81,197,255),(255,229,81),(255,81,139)]
        self.test_colors_triad = [(255,125,227),(125,255,184),(255,205,125)]

        #easy colors
        self.train_colors = [(100, 209, 72) , (209, 72, 100) , (209, 127, 72), (72, 129, 209) , (84, 184, 209), (209, 109, 84), (184, 209, 84), (109, 84, 209)]
        self.test_colors = [(222, 222, 102),(100,100,219),(219,100,219),(100,219,100)]

    def create_sample(self, n, width, height, bg = None, test=False):

        canvas = np.zeros([width, height, self.num_channel], np.float32)
        if bg=='solid_random':
            brightness = randint(0,255)
            r = randint(0,brightness)/255.
            g = randint(0,brightness)/255.
            b = randint(0,brightness)/255.
            canvas[:,:,0] = r
            canvas[:,:,1] = g
            canvas[:,:,2] = b
        elif bg=='solid_fixed':
            color = choice(self.train_colors)
            canvas[:,:,0] = color[0]/255.
            canvas[:,:,1] = color[1]/255.
            canvas[:,:,2] = color[2]/255.
        elif bg=='unseen_solid_fixed':
            color = choice(self.test_colors)    
            canvas[:,:,0] = color[0]/255.
            canvas[:,:,1] = color[1]/255.
            canvas[:,:,2] = color[2]/255.
        elif bg=='white':
            canvas[:,:,:] = np.ones_like(canvas)
        elif bg=='texture':
            img_name = np.random.choice(self.bg_list)
            # print(img_name)
            img = np.tile(np.array(Image.open(img_name))[:,:,np.newaxis]/255.,[1,1,3])
            # print(img.shape)
            canvas[:,:,:] = tf.image.resize(img, size=[width,height] )
        
        if 'rot' in bg: #ckb_rot_6
            temp_canvas = np.zeros([width*4, height*4, self.num_channel], np.float32)
            if 'unseen' in bg:
                shuffle(self.test_colors_triad)
                colors = self.test_colors_triad[:2]
            else:
                shuffle(self.train_colors_triad)
                colors = self.train_colors_triad[:2]
            cell_size = int(bg[-1])
            num_ckb = (height*4)//cell_size
            for i in range(num_ckb):
                for j in range(num_ckb):
                    temp_canvas[i*cell_size:(i+1)*cell_size,j*cell_size:(j+1)*cell_size,0] = colors[(i+j)%2][0]/255.
                    temp_canvas[i*cell_size:(i+1)*cell_size,j*cell_size:(j+1)*cell_size,1] = colors[(i+j)%2][1]/255.
                    temp_canvas[i*cell_size:(i+1)*cell_size,j*cell_size:(j+1)*cell_size,2] = colors[(i+j)%2][2]/255.
            rot_image = tfa.image.rotate(tf.convert_to_tensor(temp_canvas),tf.constant(tf.random.uniform([],-1,1)*math.pi/2,dtype=tf.float32),interpolation='BILINEAR')
            canvas = tf.image.central_crop(rot_image,0.25).numpy()
            # plt.imshow(canvas)
            # plt.show()

        elif 'ckb' in bg:
            if 'unseen' in bg:
                shuffle(self.test_colors)
                colors = self.test_colors[:2]
            else:
                shuffle(self.train_colors)
                colors = self.train_colors[:2]
            num_ckb = int(bg[0])
            h = height//num_ckb; w = width//num_ckb
            for i in range(num_ckb):
                for j in range(num_ckb):
                    canvas[i*h:(i+1)*h,j*w:(j+1)*w,0] = colors[(i+j)%2][0]/255.
                    canvas[i*h:(i+1)*h,j*w:(j+1)*w,1] = colors[(i+j)%2][1]/255.
                    canvas[i*h:(i+1)*h,j*w:(j+1)*w,2] = colors[(i+j)%2][2]/255.
            
        drawn_boxes = [] #x,y

        for i in range(n):
            rand_x = np.random.randint(0, width-14)
            rand_y = np.random.randint(0, height-14)
            while calculate_overlap(rand_x,rand_y,drawn_boxes):
                rand_x = np.random.randint(0, width-14)
                rand_y = np.random.randint(0, height-14)
            drawn_boxes.append((rand_x,rand_y))

            if not test:
                rand_img = self.train_x[np.random.randint(0, self.train_x.shape[0])]
            else:
                rand_img = self.test_x[np.random.randint(0, self.test_x.shape[0])]
            # rand_img = rand_img/255.
            # print(rand_img)
            # plt.imshow(rand_img)
            # plt.show()
            # rand_img = cv2.cvtColor(rand_img, cv2.COLOR_RGB2RGBA)
            alpha_img = np.where(np.max(rand_img,axis=-1)>0,1.0,0.0)
            rand_img = rand_img/255.

            alpha_bg = 1.0 - alpha_img
            alpha_img = alpha_img[:,:,np.newaxis]
            alpha_bg = alpha_bg[:,:,np.newaxis]
            # print('alpha_img.shape',alpha_img.shape)
            # print('alpha_bg.shape',alpha_bg.shape)

            canvas[rand_x:rand_x+14 , rand_y:rand_y+14] = alpha_img * rand_img + alpha_bg * canvas[rand_x:rand_x+14 , rand_y:rand_y+14]
            # plt.imshow(canvas)
            # plt.show()
            # canvas[rand_x:rand_x+14 , rand_y:rand_y+14, :] = \
            #         np.clip(canvas[rand_x:rand_x+14 , rand_y:rand_y+14, :] + rand_img/255. , 0., 1.)

        return canvas

    def create_dataset(self, nsamples, digits, size,bg=None, test=False):
        dataset_buffer = np.zeros([nsamples, size, size, self.num_channel])
        if test:
            count = np.zeros([nsamples])
        for i in range(nsamples):
            # print('sample no:',i)
            rand_n = np.random.randint(digits[0], digits[1]+1)
            # print('rand_n:',rand_n)
            if test:
                count[i] = rand_n
            sample = self.create_sample(rand_n, size, size,bg, test)        
            dataset_buffer[i] = sample
        if test:
            return dataset_buffer.astype(np.float32), count
        return dataset_buffer.astype(np.float32)


def random_crop(input_image,crop_size):
    cropped_image = tf.image.random_crop(input_image, size=[crop_size, crop_size, input_image.shape[2]])

    return cropped_image

def resize_random_crop(img,crop_size,resize_size):
    return tf.image.resize(random_crop(img,crop_size),resize_size)


def _bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    if isinstance(value, type(tf.constant(0))):
        value = value.numpy() # BytesList won't unpack a string from an EagerTensor.
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _float_feature(value):
  """Returns a float_list from a float / double."""
  return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

def _int64_feature(value):
  """Returns an int64_list from a bool / enum / int / uint."""
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))



def serialize_images_and_labels(images,labels):

    feature = {'image': _bytes_feature(tf.io.serialize_tensor(images))  ,'label': _int64_feature(labels)}

    # Create an example protocol buffer
    example = tf.train.Example(features=tf.train.Features(feature=feature))
    return example.SerializeToString()

def tf_serialize_images_and_labels(images,labels):
    tf_string = tf.py_function(
        serialize_images_and_labels,
        (images,labels),  # pass these args to the above function.  
        tf.string)      # the return type is `tf.string`.
    return tf.reshape(tf_string, ()) # The result is a scalar

def parse_48_with_label(example_proto,size,channel):
    feature_description = {
            'image': tf.io.FixedLenFeature([], tf.string, default_value=''),
            'label': tf.io.FixedLenFeature([], tf.int64, default_value=0)
            }

    example = tf.io.parse_single_example(example_proto, feature_description)
    image = tf.io.parse_tensor(example['image'], out_type=tf.float32)
    image = tf.reshape(image, [size,size,channel])
    label = example['label']
    return image,label

def create_cub_tfrec(name):
    data =  load_cub_masked()
    multi_cub = MultiCUB(data,reshape=False)
    if (name != 'cub_solid_fixed') and (name != 'cub_ckb_rot_6'):
        print(name)
        raise NotImplementedError('Undefined dataset')

    # name = 'cub_solid_fixed' #'cub_ckb_rot_6', 'cub_16x16_ckb'
    Path('data/multi_cub/').mkdir(parents=True, exist_ok=True)

    numpy_test_unseen_dataset, count_test_unseen_dataset = multi_cub.create_dataset(1000,digits=[0,5],size=48,bg= 'unseen_' + name[4:] , test=True) #16x16_ckb_unseen
    test_dataset_unseen = tf.data.Dataset.from_tensor_slices((numpy_test_unseen_dataset,count_test_unseen_dataset))

    test_dataset_unseen = test_dataset_unseen.map(tf_serialize_images_and_labels)
    tfrec = tf.data.experimental.TFRecordWriter('data/multi_cub/test_unseen_' + name + '.tfrec')
    tfrec.write(test_dataset_unseen)

    numpy_train_dataset = multi_cub.create_dataset(100000,digits=[0,5],size=48,bg=name[4:])
    numpy_test_dataset, count_test_dataset = multi_cub.create_dataset(1000,digits=[0,5],size=48,bg=name[4:],test=True)

    test_dataset = tf.data.Dataset.from_tensor_slices((numpy_test_dataset,count_test_dataset)).map(tf_serialize_images_and_labels)
    tfrec = tf.data.experimental.TFRecordWriter('data/multi_cub/test_' + name + '.tfrec')
    tfrec.write(test_dataset)

    train_dataset = tf.data.Dataset.from_tensor_slices(numpy_train_dataset).map(tf.io.serialize_tensor)
    tfrec = tf.data.experimental.TFRecordWriter('data/multi_cub/train_' + name + '.tfrec')
    tfrec.write(train_dataset)


def get_cub_dataset(name,size=48,channel=3):
    return get_cub_tfrec(name,size,channel)


def get_cub_tfrec(name,size=48,channel=3):
    def parse(x):
        result = tf.io.parse_tensor(x, out_type=tf.float32)
        result = tf.reshape(result, [size, size, channel])
        return result
    train_path = 'data/multi_cub/train_' + name + '.tfrec'
    test_path = 'data/multi_cub/test_' + name + '.tfrec'
    test_unseen_path = 'data/multi_cub/test_unseen_' + name + '.tfrec'
    if (not os.path.exists(train_path)) or (not os.path.exists(test_path)) or (not os.path.exists(test_unseen_path)):
        print('TFRecord files not found, creating TFRecord files. This might take a while.')
        create_cub_tfrec(name)

    train_dataset = tf.data.TFRecordDataset(train_path).map(parse, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    test_dataset = tf.data.TFRecordDataset(test_path).map(lambda x: parse_48_with_label(x,size,channel), num_parallel_calls=tf.data.experimental.AUTOTUNE)
    test_dataset_unseen = tf.data.TFRecordDataset(test_unseen_path).map(lambda x: parse_48_with_label(x,size,channel), num_parallel_calls=tf.data.experimental.AUTOTUNE)

    return train_dataset, [test_dataset,test_dataset_unseen], [-1,size,size,channel], [-1,size,size,channel]

if __name__ == "__main__":
    get_cub_tfrec(name='cub_ckb_rot_6')
    # os.environ["CUDA_VISIBLE_DEVICES"]="-1"
    # data =  load_cub_masked()
    # multi_cub = MultiCUB(data,reshape=False)
    # name = 'cub_ckb_rot_6' #'cub_ckb_rot_6', 'cub_16x16_ckb'

    # numpy_test_unseen_dataset, count_test_unseen_dataset = multi_cub.create_dataset(1000,digits=[0,5],size=48,bg='unseen_cub_ckb_rot_6',test=True) #16x16_ckb_unseen
    # for img in numpy_test_unseen_dataset:
    #     plt.imshow(img)
    #     plt.show()
    # test_dataset_unseen = tf.data.Dataset.from_tensor_slices((numpy_test_unseen_dataset,count_test_unseen_dataset))

    # test_dataset_unseen = test_dataset_unseen.map(tf_serialize_images_and_labels)
    # tfrec = tf.data.experimental.TFRecordWriter('data/multi_cub/test_unseen_' + name + '.tfrec')
    # tfrec.write(test_dataset_unseen)

    # numpy_train_dataset = multi_cub.create_dataset(100000,digits=[0,5],size=48,bg='cub_ckb_rot_6')
    # numpy_test_dataset, count_test_dataset = multi_cub.create_dataset(1000,digits=[0,5],size=48,bg='cub_ckb_rot_6',test=True)

    # test_dataset = tf.data.Dataset.from_tensor_slices((numpy_test_dataset,count_test_dataset)).map(tf_serialize_images_and_labels)
    # tfrec = tf.data.experimental.TFRecordWriter('data/multi_cub/test_' + name + '.tfrec')
    # tfrec.write(test_dataset)

    # train_dataset = tf.data.Dataset.from_tensor_slices(numpy_train_dataset).map(tf.io.serialize_tensor) # .map(random_crop)
    # tfrec = tf.data.experimental.TFRecordWriter('data/multi_cub/train_' + name + '.tfrec')
    # tfrec.write(train_dataset)

    #######################################################################################################

    # train_dataset, test_dataset, input_shape, test_size = get_cub_dataset('cub_solid_fixed_triad')
    # for i, train_data in enumerate(train_dataset):
    #     print(train_data.shape)
    #     plt.imshow( tf.squeeze(train_data) ,cmap='gray')
    #     plt.show()
    #     if i ==3:
    #         break
    # for i, (image,count) in enumerate(test_dataset[1]):
    #     print(image.shape)
    #     plt.imshow( tf.squeeze(image) ,cmap='gray')
    #     plt.show()
    #     if i ==3:
    #         break
