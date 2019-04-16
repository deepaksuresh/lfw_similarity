from utils.model import Input_data
from utils.pair_gen import Pair
from utils.data_feeder import Data
from utils.model import Inception_ResNetV2_model
import tensorflow as tf

generator = Pair()
ds = Data(generator)
model_input = ds.next_set
model = Inception_ResNetV2_model(model_input)
saver = tf.train.Saver()

with tf.Session() as sess:
    #sess.run(ds.iterator.initializer)
    sess.run(tf.global_variables_initializer())
    saver.save(sess,"/home/nanonets/model.ckpt")
    for step in range(200):
        (_, current_loss) = sess.run([model.opt_step, model.loss])
        print(step," ", current_loss)
