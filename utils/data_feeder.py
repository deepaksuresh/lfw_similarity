#filter images from lfw
#load filtered set
#augment on the fly
#yield a set of images and labels

import tensorflow as tf
from .model import Input_data
from .pair_gen import Pair 

class Data(object):
    img1 = "first"
    img2 = "second"
    label = ""

    def __init__(self, img_generator=Pair()):
        self.next_set = self.img_iterator(img_generator)
        self.iterator = None
    def img_iterator(self, img_gen):
        batch_size = 16
        prefetch_batch_buffer = 8

        dataset = tf.data.Dataset.from_generator(img_gen.get_pair, output_types = {Pair.first:tf.string, Pair.second:tf.string, Pair.label:tf.bool})
        dataset = dataset.map(self.read_augment)
        dataset = dataset.batch(batch_size)
        dataset = dataset.prefetch(prefetch_batch_buffer)
        iterator = dataset.make_one_shot_iterator()
        self.iterator = iterator
        next_set = iterator.get_next()
        return Input_data(next_set[self.img1], next_set[self.img2], next_set[self.label])

    def read_augment(self, pair):
        im_size = [299,299]
        im1 = tf.io.read_file(pair[Pair.first])
        im2 = tf.io.read_file(pair[Pair.second])

        img1 = tf.image.decode_image(im1)
        img2 = tf.image.decode_image(im2)
        
        img1.set_shape([250,250,3])
        img2.set_shape([250,250,3])

        im1_resized = tf.image.resize_images(img1, im_size)
        im2_resized = tf.image.resize_images(img2, im_size)
        #Augmentation
        im1_resized = tf.image.random_flip_left_right(im1_resized)
        im1_resized = tf.image.random_brightness(im1_resized, max_delta=0.3)
        im1_resized = tf.image.random_contrast(im1_resized, lower=0.2, upper=1.8)
        im1_resized = tf.contrib.image.rotate(im1_resized, angles=0.15)
        im1_resized = tf.contrib.image.translate(im1_resized, translations=[5,5])
        
        im2_resized = tf.image.random_flip_left_right(im2_resized)
        im2_resized = tf.image.random_brightness(im2_resized, max_delta=0.3)
        im2_resized = tf.image.random_contrast(im2_resized, lower=0.2, upper=1.8)
        im2_resized = tf.contrib.image.rotate(im2_resized, angles=0.15)
        im2_resized = tf.contrib.image.translate(im2_resized, translations=[5,5])

        pair[self.img1] = im1_resized
        pair[self.img2] = im2_resized
        pair[self.label] = tf.cast(pair[Pair.label], tf.float32)
        return pair
