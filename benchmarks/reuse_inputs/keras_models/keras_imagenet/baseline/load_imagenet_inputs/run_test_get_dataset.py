"""test_get_dataset.py

This test script could be used to verify either the 'train' or
'validation' dataset, by visualizing data augmented images on
TensorBoard.

Examples:
$ cd ${HOME}/project/keras_imagenet
$ python3 test_get_dataset.py train
$ tensorboard --logdir logs/train
"""

import os
import shutil
import argparse

import tensorflow as tf

from dataset import get_dataset

#DATASET_DIR = os.path.join(os.environ['HOME'], 'data/ILSVRC2012/tfrecords')
DATASET_DIR = "/data/google_imagenet/raw_data/tf_records/train"

#parser = argparse.ArgumentParser()
#parser.add_argument('subset', type=str, choices=['train', 'validation'])
#args = parser.parse_args()

#log_dir = os.path.join('logs', args.subset)
#shutil.rmtree(log_dir, ignore_errors=True)  # clear prior log data

dataset = get_dataset(DATASET_DIR, 'train', batch_size=64)
#dataset = get_dataset(DATASET_DIR, args.subset, batch_size=64)
iterator = dataset.make_initializable_iterator()
batch_xs, batch_ys = iterator.get_next()
"""
ret_x = sess.run(batch_xs)
ret_y = sess.run(batch_ys)
print(ret_x.shape)
print(ret_y.shape)

batch_xs: Tensor
batch_ys: Tensor

(64, 224, 224, 3)
(64, 1000)
"""

#mean_rgb = tf.reduce_mean(batch_xs, axis=[0, 1, 2])

# convert normalized image back: [-1, 1] -> [0, 1]
#batch_imgs = tf.multiply(batch_xs, 0.5)
#batch_imgs = tf.add(batch_imgs, 0.5)

#summary_op = tf.summary.image('image_batch', batch_imgs, max_outputs=64)

with tf.Session() as sess:
  #writer = tf.summary.FileWriter(log_dir, sess.graph)
  sess.run(iterator.initializer)
  ret_x = sess.run(batch_xs)
  print('>>> ', ret_x.shape)
  ret_y = sess.run(batch_ys)
  print('>>> ', ret_y.shape)
  #rgb = sess.run(mean_rgb)
  #print('Mean RGB (-1.0~1.0):', rgb)

  #summary = sess.run(summary_op)
  #writer.add_summary(summary)
  #writer.close()