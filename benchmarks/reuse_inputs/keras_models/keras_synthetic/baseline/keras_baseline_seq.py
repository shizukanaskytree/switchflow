import os
#os.environ['TF_CPP_MIN_VLOG_LEVEL']='1'

import random
import string
import time
import datetime
import numpy as np
import threading

import tensorflow as tf
from tensorflow.python.keras import backend
from tensorflow.python.keras import initializers
from tensorflow.python.keras import layers
from tensorflow.python.keras import regularizers
from tensorflow.python.keras.optimizer_v2 import gradient_descent as gradient_descent_v2
from tensorflow.python.keras.applications import densenet, inception_resnet_v2
from tensorflow.python.keras.applications import inception_v3, mobilenet, mobilenet_v2
from tensorflow.python.keras.applications import nasnet, resnet50, vgg16, vgg19, xception
from tensorflow.python.keras.callbacks import ModelCheckpoint
from tensorflow.python.keras.models import load_model
from tensorflow.python.framework import graph_io
from tensorflow.python.keras.backend import categorical_crossentropy
from tensorflow.python.client import timeline

###############################################################
# N.B.
# Manage and switch to the right python env: base
###############################################################

model_name = 'ResNet50'
num_steps = 53
BATCH_SIZE = 256
HEIGHT = 224
WIDTH = 224
NUM_CHANNELS = 3
data_format = 'channels_first'
shape = (NUM_CHANNELS, HEIGHT, WIDTH)  # Channel GPU prefers channels_first.
#DO_TRAIN = True
dtype = tf.float32
NUM_CLASSES = 1000
inter_op_threads_num = 32

models = {
  'DenseNet121': densenet.DenseNet121,
  'DenseNet169': densenet.DenseNet169,
  'DenseNet201': densenet.DenseNet201,
  'InceptionResNetV2': inception_resnet_v2.InceptionResNetV2,
  'InceptionV3': inception_v3.InceptionV3,
  'MobileNet': mobilenet.MobileNet,
  'MobileNetV2': mobilenet_v2.MobileNetV2,
  'NASNetLarge': nasnet.NASNetLarge,
  'NASNetMobile': nasnet.NASNetMobile,
  'ResNet50': resnet50.ResNet50,
  'VGG16': vgg16.VGG16,
  'VGG19': vgg19.VGG19,
}

tf.keras.backend.set_image_data_format(data_format)

config = tf.ConfigProto(inter_op_parallelism_threads=inter_op_threads_num)

sess = tf.Session(config=config)
# Set the global Session.
tf.keras.backend.set_session(session=sess)

# input
dummy_input_X = np.ones(shape=[BATCH_SIZE, NUM_CHANNELS, HEIGHT, WIDTH])
dummy_input_y = np.ones(shape=[BATCH_SIZE, NUM_CLASSES])

def CreateGraph(graph_name, X_name, y_name):
  X = tf.placeholder(tf.float32,
                     [None, NUM_CHANNELS, HEIGHT, WIDTH],
                     name=X_name)
  y = tf.placeholder(tf.float32, [None, NUM_CLASSES], name=y_name)

  #with tf.name_scope(name=graph_name):
  #model = mobilenet_v2.MobileNetV2(weights=None, input_tensor=X)
  #model = resnet50.ResNet50(weights=None, input_tensor=X)
  model = models[model_name](weights=None, input_tensor=X)

  # 2. Forward pass
  # Evaluate model
  correct_pred = tf.equal(tf.argmax(model.output, 1), tf.argmax(y, 1))
  accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32), name=graph_name+'accuracy')
  init_op = tf.global_variables_initializer()

  return X, y, init_op, accuracy

X00, y00, init_00, accuracy_00 = \
  CreateGraph(graph_name='graph_00', X_name='XX00', y_name='yy00')

X01, y01, init_01, accuracy_01 = \
  CreateGraph(graph_name='graph_01', X_name='XX01', y_name='yy01')

#current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
#train_log_dir = 'logs_baseline/' + current_time + '/train'

# Start training
#train_writer = tf.summary.FileWriter(train_log_dir, sess.graph)

# Run the initializer
sess.run(init_00)
sess.run(init_01)
#sess.run(init_03)
#sess.run(init_04)

# collect runtime statistics
#run_options = None
#run_metadata = None

sess_run_time_total = 0
sess_run_time_start = 0
sess_run_time_end = 0

start_01 = 0
for step in range(num_steps):
  if step == 3:
    # warm up
    print('start timer')
    start_01 = time.time()

  # mnist input
  # 1.
  # batch_x = ndarray: (128, 784)
  # batch_y = ndarray: (128, 10)
  # where batch size == 128.

  #if step == 10:
  #  # collect execution stats
  #  run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
  #  run_metadata = tf.RunMetadata()
  #  sess.run(accuracy_01, feed_dict={X01: batch_x, Y01: batch_y},
  #           options=run_options,
  #           run_metadata=run_metadata)
  #  train_writer.add_run_metadata(run_metadata, 'step%d' % step)
  #  continue

  if step >= 3:
    sess_run_time_start = time.time()

  sess.run(accuracy_00, feed_dict={X00: dummy_input_X, y00: dummy_input_y})

  if step >= 3:
    sess_run_time_end = time.time()

  sess_run_time_total += (sess_run_time_end - sess_run_time_start)
end_01 = time.time()

start_02 = 0
for step in range(num_steps):
  if step == 3:
    # warm up
    print('start timer')
    start_02 = time.time()

  if step >= 3:
    sess_run_time_start = time.time()

  sess.run(accuracy_01, feed_dict={X01: dummy_input_X, y01: dummy_input_y})

  if step >= 3:
    sess_run_time_end = time.time()

  sess_run_time_total += (sess_run_time_end - sess_run_time_start)
end_02 = time.time()

#  start_03 = 0
#  for step in range(num_steps):
#    if step == 3:
#      # warm up
#      print('start timer')
#      start_03 = time.time()
#
#    # mnist input
#    batch_x, batch_y = mnist.train.next_batch(batch_size)
#    # numpy random inputs
#    #batch_x, batch_y = random_inputs(batch_size)
#
#    if step >= 3:
#      sess_run_time_start = time.time()
#
#    sess.run(accuracy_03, feed_dict={X03: batch_x, Y03: batch_y})
#
#    if step >= 3:
#      sess_run_time_end = time.time()
#
#    sess_run_time_total += (sess_run_time_end - sess_run_time_start)
#  end_03 = time.time()
#
#  start_04 = 0
#  for step in range(num_steps):
#    if step == 3:
#      # warm up
#      print('start timer')
#      start_04 = time.time()
#
#    # mnist input
#    batch_x, batch_y = mnist.train.next_batch(batch_size)
#    # numpy random inputs
#    #batch_x, batch_y = random_inputs(batch_size)
#
#    if step >= 3:
#      sess_run_time_start = time.time()
#
#    sess.run(accuracy_04, feed_dict={X04: batch_x, Y04: batch_y})
#
#    if step >= 3:
#      sess_run_time_end = time.time()
#
#    sess_run_time_total += (sess_run_time_end - sess_run_time_start)
#  end_04 = time.time()

#print('Baseline, seq, total time: ', end_01 - start_01)
print('Baseline, seq, total time: ', end_01 - start_01 + end_02 - start_02)
#print('Baseline, seq, total time: ', end_01 - start_01 + end_02 - start_02 + end_03 - start_03)
#print('Baseline, seq, total time: ', end_01 - start_01 + end_02 - start_02 + end_03 - start_03 + end_04 - start_04)
print('Baseline, seq, sess total time: ', sess_run_time_total)

#  # Create the Timeline object, and write it to a json
#  tl = timeline.Timeline(run_metadata.step_stats)
#  ctf = tl.generate_chrome_trace_format()
#
#  timeline_folder = 'timeline_baseline_logs'
#  timeline_filename = 'baseline_timeline_' + current_time + '.json'
#  if not os.path.exists(timeline_folder):
#    os.makedirs(timeline_folder)
#  timeling_logs_path_file = os.path.join(os.getcwd(), timeline_folder, timeline_filename)
#  with open(timeling_logs_path_file, 'w') as f:
#    f.write(ctf)

print("Finished!")