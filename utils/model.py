import tensorflow as tf
from tensorflow import Tensor
import numpy as np

class Input_data(object):

    def __init__(self, img1, img2, label):
        self.img1 = img1
        self.img2 = img2
        self.label = label

    def feed_input(self, input_img1, input_img2, input_label = None):
        feed_dict = {self.img1:input_img1, self.img2:input_img2}

        if input_label is not None:
            feed_dict[self.label] = input_label
        return feed_dict

class Dumb_model(object):
    def __init__(self, inputs):
        self.inputs = inputs
        self.predictions = self.predict(inputs)
        self.loss = self.calculate_loss(inputs, self.predictions)
        self.opt_step = tf.train.AdamOptimizer(learning_rate=0.001).minimize(self.loss)

    def predict(self, inputs):
        img_diff = (inputs.img1 - inputs.img2)
        x = img_diff
        for conv_layer_i in range(6):
            x = tf.layers.conv2d(x, filters=20*(conv_layer_i+1), kernel_size=3, activation=tf.nn.relu)
            x = tf.layers.max_pooling2d(x, pool_size=3, strides=2)

        x = tf.layers.dense(x, units=100, activation=tf.nn.relu)
        logits = tf.layers.dense(x,1,activation=None)
        return tf.squeeze(logits)

    def calculate_loss(self, inputs, logits):
        print(inputs.label)
        return tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=inputs.label, logits=logits))

class Inception_ResNetV2_model(object):
    def __init__(self, inputs):
        self.pre_process = tf.keras.applications.inception_resnet_v2.preprocess_input
        self.inc_res = tf.keras.applications.InceptionResNetV2(input_shape=(299,299,3), include_top=False, weights='imagenet')
        self.inc_res.trainable = False
        self.global_average_layer = tf.keras.layers.GlobalAveragePooling2D()
        self.predictions = self.predict(inputs)
        self.loss = self.calculate_loss(inputs, self.predictions)
        self.opt_step = tf.train.AdamOptimizer(learning_rate=0.001).minimize(self.loss)

    def predict(self, inputs):
        with tf.variable_scope("pre_trained") as scope:
        #img_diff = (inputs.img1-inputs.img2)
        #x = img_diff
            first_img = tf.identity(inputs.img1, name="first_image")
            second_img = tf.identity(inputs.img2, name="second_image")
            x1 = self.inc_res(self.pre_process(first_img))
            print(x1.shape)
            x1 = tf.layers.conv2d(x1,filters=1024,kernel_size=3,activation=tf.nn.relu, name="conv1")
            #x1 = tf.layers.dense(x1, units=1024,activation = tf.nn.relu,name="firstFCC")
            x1 = self.global_average_layer(x1)
            scope.reuse_variables()
            x2 = self.inc_res(self.pre_process(second_img))
            x2 = tf.layers.conv2d(x2, filters=1024,kernel_size=3,activation=tf.nn.relu, name="conv1", reuse=True)
            #x2 = tf.layers.dense(x2, units=1024, activation=tf.nn.relu, name="firstFCC", reuse=True)
            x2 = self.global_average_layer(x2)
            print(x2.shape)
        with tf.variable_scope("logits") as scope:
            x3 = tf.math.abs(x1-x2)
            x = tf.layers.dense(x3, units = 512, activation=tf.nn.relu)
            x = tf.layers.dense(x, units = 128, activation=tf.nn.relu)
            x = tf.layers.dense(x,1,activation=None, name="score")
            print("score = ",x)
        return x

    def calculate_loss(self, inputs, logits):
        return tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=inputs.label, logits=logits))
