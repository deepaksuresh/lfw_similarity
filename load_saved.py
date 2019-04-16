import tensorflow as tf
import numpy as np

sess = tf.Session()
saver = tf.train.import_meta_graph('model.ckpt.meta')
saver.restore(sess, tf.train.latest_checkpoint('./'))
pre_processor = tf.keras.applications.inception_resnet_v2.preprocess_input

def get_score(image_path1, image_path2, session = sess):
    """Input: paths to two images
    output: similarity score and binary classification
    """
    im_size = [299,299]
    path1 = tf.placeholder(dtype=tf.string)
    path2 = tf.placeholder(dtype=tf.string)

    im1 = tf.io.read_file(path1)
    im2 = tf.io.read_file(path2)

    img1 = tf.image.decode_image(im1)
    img2 = tf.image.decode_image(im2)

    img1.set_shape([250,250,3])
    img2.set_shape([250,250,3])

    im1_resized = tf.image.resize_images(img1, im_size)
    im2_resized = tf.image.resize_images(img2, im_size)

    im1_processed = pre_processor(im1_resized)
    im2_processed = pre_processor(im2_resized)

    im1_processed = tf.reshape(im1_processed, [1,299,299,3])
    im2_processed = tf.reshape(im2_processed, [1,299,299,3])

    first_image = sess.run(im1_processed, feed_dict = {path1:image_path1})
    second_image = sess.run(im2_processed, feed_dict = {path2:image_path2})

    first_img = sess.graph.get_tensor_by_name("pre_trained/first_image:0")
    second_img = sess.graph.get_tensor_by_name("pre_trained/second_image:0")

    result = sess.graph.get_tensor_by_name("logits/score/BiasAdd:0")
    similarity_score = tf.nn.sigmoid(result)

    final_score = sess.run(similarity_score, feed_dict = {first_img:first_image, second_img:second_image})

    print("score = ", final_score)
    print("is_similar? ", final_score[0][0]>0.5)
    return None
