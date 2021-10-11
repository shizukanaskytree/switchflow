import os
os.environ['TF_CPP_MIN_VLOG_LEVEL']='0'

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
from tensorflow.python.profiler.profile_context import ProfileContext

from load_imagenet_inputs.dataset import get_dataset, get_dataset_jpeg

model_name_01 = 'MobileNetV2'
model_name_02 = 'MobileNetV2'
#model_name_03 = 'NASNetMobile'
task_name = 'infer' # or 'train' or 'infer'
num_steps = 203 # extra 3 steps for warmup
TIMELINE_SAMPLE_STEP = int(num_steps / 10) # sample 10 times, so step is 203/10 ~= 20 as an example
BATCH_SIZE = 128
HEIGHT = 224
WIDTH = 224
NUM_CHANNELS = 3
data_format = 'channels_first'
shape = (NUM_CHANNELS, HEIGHT, WIDTH)  # Channel GPU prefers channels_first.
DO_TRAIN = True
dtype = tf.float32
NUM_CLASSES = 1000
inter_op_threads_num = 32  # wxf:debug, set 1

models = {
  'ResNet50': resnet50.ResNet50,
  'VGG16': vgg16.VGG16,
  'VGG19': vgg19.VGG19,
  'MobileNetV2': mobilenet_v2.MobileNetV2,
  'MobileNet': mobilenet.MobileNet,
  'DenseNet121': densenet.DenseNet121,
  'DenseNet169': densenet.DenseNet169,
  'DenseNet201': densenet.DenseNet201,
  'InceptionResNetV2': inception_resnet_v2.InceptionResNetV2,
  'InceptionV3': inception_v3.InceptionV3,
  'NASNetLarge': nasnet.NASNetLarge,
  'NASNetMobile': nasnet.NASNetMobile,
}

config = tf.ConfigProto(inter_op_parallelism_threads=inter_op_threads_num)
sess = tf.Session(config=config)
#tf.keras.backend.set_session(session=sess)

tf.keras.backend.set_image_data_format(data_format)

DATASET_DIR = "/data1/google_imagenet/raw_data/tf_records/train"
JPEG_DATASET_DIR  = '/data1/google_imagenet/raw_data/train_slices'
LABEL_FILENAME= '/data1/google_imagenet/raw_data/folder_label_mapping.txt'

# ===------===
# import data for graph 01
# ===------===
#dataset_01 = get_dataset(DATASET_DIR, 'train', batch_size=BATCH_SIZE)
dataset_01 = get_dataset_jpeg(JPEG_DATASET_DIR, LABEL_FILENAME, 'train', batch_size=BATCH_SIZE)
#dataset_01 = get_dataset_jpeg(JPEG_DATASET_DIR, LABEL_FILENAME, 'infer', batch_size=BATCH_SIZE)
iterator_01 = dataset_01.make_initializable_iterator()
batch_xs_01, batch_ys_01 = iterator_01.get_next()

# ===------===
# import data for graph 02
# ===------===
#dataset_02 = get_dataset(DATASET_DIR, 'train', batch_size=BATCH_SIZE)
dataset_02 = get_dataset_jpeg(JPEG_DATASET_DIR, LABEL_FILENAME, 'train', batch_size=BATCH_SIZE)
#dataset_02 = get_dataset_jpeg(JPEG_DATASET_DIR, LABEL_FILENAME, 'infer', batch_size=BATCH_SIZE)
iterator_02 = dataset_02.make_initializable_iterator()
batch_xs_02, batch_ys_02 = iterator_02.get_next()
# comment: separate load graph for different graphs.
# 1.
# shape of batch_xs= (?, 3, 224, 224)
# NHWC (N, Height, width, channel) is the TensorFlow default and
# NCHW is the optimal format to use for NVIDIA cuDNN.

## ===------===
## import data for graph 03 and graph 03 construct
## ===------===
#dataset_03 = get_dataset_jpeg(JPEG_DATASET_DIR, LABEL_FILENAME, 'train', batch_size=BATCH_SIZE)
#iterator_03 = dataset_03.make_initializable_iterator()
#batch_xs_03, batch_ys_03 = iterator_03.get_next()

## ===------===
## import data for graph 04 and graph 04 construct
## ===------===
#dataset_04 = get_dataset(DATASET_DIR, 'train', batch_size=BATCH_SIZE)
#iterator_04 = dataset_04.make_initializable_iterator()
#batch_xs_04, batch_ys_04 = iterator_04.get_next()
#
#model_04 = models[model_name](weights=None, input_tensor=batch_xs_04)
#loss_04 = tf.reduce_mean(categorical_crossentropy(batch_ys_04, model_04.output))
#train_step_op_04 = tf.train.GradientDescentOptimizer(0.5).minimize(loss_04)
#target_op_04 = train_step_op_04
#
## ===------===
## import data for graph 05 and graph 05 construct
## ===------===
#dataset_05 = get_dataset(DATASET_DIR, 'train', batch_size=BATCH_SIZE)
#iterator_05 = dataset_05.make_initializable_iterator()
#batch_xs_05, batch_ys_05 = iterator_05.get_next()
#
#model_05 = models[model_name](weights=None, input_tensor=batch_xs_05)
#loss_05 = tf.reduce_mean(categorical_crossentropy(batch_ys_05, model_05.output))
#train_step_op_05 = tf.train.GradientDescentOptimizer(0.5).minimize(loss_05)
#target_op_05 = train_step_op_05

# =======================================================================
# Graph construction 01
# =======================================================================
# if DO_TRAIN:
#  model = resnet50.ResNet50(weights=None, input_tensor=batch_xs)
# else:
#  model = resnet50.ResNet50(weights='imagenet',
#                            include_top=True, # default is also True...
#                            input_shape=shape)

model_01 = models[model_name_01](weights=None, input_tensor=batch_xs_01)

# 1. training network
loss_01 = tf.reduce_mean(categorical_crossentropy(batch_ys_01, model_01.output))
# 1.
# https://gombru.github.io/2018/05/23/cross_entropy_loss/
train_step_op_01 = tf.train.GradientDescentOptimizer(0.5).minimize(loss_01)

# 2. Forward pass: accuracy
correct_pred_01 = tf.equal(tf.argmax(model_01.output, 1), tf.argmax(batch_ys_01, 1))
accuracy_01 = tf.reduce_mean(tf.cast(correct_pred_01, tf.float32), name='accuracy_01')

target_op_01 = accuracy_01
# ===================================
# ~Graph construction 01
# ===================================

# =======================================================================
# Graph construction 02
# =======================================================================
model_02 = models[model_name_02](weights=None, input_tensor=batch_xs_02)

# 1. training network
loss_02 = tf.reduce_mean(categorical_crossentropy(batch_ys_02, model_02.output))
train_step_op_02 = tf.train.GradientDescentOptimizer(0.5).minimize(loss_02)

# 2. forward pass only: accuracy
correct_pred_02 = tf.equal(tf.argmax(model_02.output, 1), tf.argmax(batch_ys_02, 1))
accuracy_02 = tf.reduce_mean(tf.cast(correct_pred_02, tf.float32), name='accuracy_02')

target_op_02 = accuracy_02
# ===================================
# ~Graph construction 02
# ===================================

## =======================================================================
## Graph construction 03
## =======================================================================
#model_03 = models[model_name_03](weights=None, input_tensor=batch_xs_03)
## 1.train
#loss_03 = tf.reduce_mean(categorical_crossentropy(batch_ys_03, model_03.output))
#train_step_op_03 = tf.train.GradientDescentOptimizer(0.5).minimize(loss_03)
#
## 2.infer
#correct_pred_03 = tf.equal(tf.argmax(model_03.output, 1), tf.argmax(batch_ys_03, 1))
#accuracy_03 = tf.reduce_mean(tf.cast(correct_pred_03, tf.float32), name='accuracy_03')
#
#target_op_03 = accuracy_03
## ===================================
## ~Graph construction 03
## ===================================

## tfboard
#current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
#train_log_dir = 'tfboard_logs/' + current_time + '/train'
#
## collect runtime statistics
#run_options = None
#run_metadata = None
#train_writer = tf.summary.FileWriter(train_log_dir, sess.graph)

init_op = tf.global_variables_initializer()
sess.run(init_op)

sess.run(iterator_01.initializer)
sess.run(iterator_02.initializer)

#######################################################################
# wxf: graph 03 - 0N
#sess.run(iterator_03.initializer)
#sess.run(iterator_04.initializer)
#sess.run(iterator_05.initializer)
#~wxf
#######################################################################

sess_run_time_total = 0
sess_run_time_start = 0
sess_run_time_end = 0

start_01 = 0
for step in range(num_steps):
  if step == 3:
    # warm up
    print('start timer')
    start_01 = time.time()

  #if step == 40:
  #  # collect execution stats
  #  run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
  #  run_metadata = tf.RunMetadata()

  #  sess.run(target_op_01,
  #           options=run_options,
  #           run_metadata=run_metadata)

  #  train_writer.add_run_metadata(run_metadata, 'step%d' % step)
  #  continue

  if step >= 3:
    sess_run_time_start = time.time()

  # Create a profiling context, set constructor argument `trace_steps`,
  # `dump_steps` to empty for explicit control.
  #with ProfileContext('./profile_logs',
  #                    trace_steps=[],
  #                    dump_steps=[]
  #                    ) as pctx:
  #  # Create options to profile the time and memory information.
  #  opts = tf.profiler.ProfileOptionBuilder.time_and_memory()

  #  # Enable tracing for next session.run.
  #  pctx.trace_next_step()
  #  # Dump the profile to './xxx_dir' after the step.
  #  pctx.dump_next_step()

  #print('start step: ', step, " ", datetime.datetime.now())
  print('start step:', step, '01_py_start_time:', time.time())
  sess.run(target_op_01)
  print('end   step:', step, '01_py_end_time:', time.time())
  #print('end   step: ', step, " ", datetime.datetime.now())

  #  # Run online profiling with 'op' view and 'opts' options at step 15, 18, 20.
  #  pctx.add_auto_profiling('op', opts, [21, 23, 24])
  #  pctx.profiler.profile_operations(options=opts)

  if step >= 3:
    sess_run_time_end = time.time()
    sess_run_time_total += (sess_run_time_end - sess_run_time_start)
    print('step:', step, '01_py_step_time:', "%.6f" % (sess_run_time_end - sess_run_time_start))

end_01 = time.time()

start_02 = 0
for step in range(num_steps):
  if step == 3:
    # warm up
    print('start timer')
    start_02 = time.time()

  if step >= 3:
    sess_run_time_start = time.time()

  #if step % TIMELINE_SAMPLE_STEP == TIMELINE_SAMPLE_STEP - 1:
  #  # collect execution stats
  #  run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
  #  run_metadata = tf.RunMetadata()

  #  sess.run(target_op_02,
  #           options=run_options,
  #           run_metadata=run_metadata)

  #  train_writer.add_run_metadata(run_metadata, 'step%d' % step)

  #  # Create the Timeline object, and write it to a json
  #  tl = timeline.Timeline(run_metadata.step_stats)
  #  ctf = tl.generate_chrome_trace_format()
  #  timeline_folder = 'timeline_baseline_logs/' + task_name + '/' + model_name_02.lower()
  #  timeline_filename = model_name_02.lower() + '_' + 'step' + '_' + str(step) + '_' + 'baseline_timeline_' + current_time + '.json'
  #  if not os.path.exists(timeline_folder):
  #    os.makedirs(timeline_folder)
  #  timeling_logs_path_file = os.path.join(os.getcwd(), timeline_folder, timeline_filename)
  #  with open(timeling_logs_path_file, 'w') as f:
  #    f.write(ctf)

  #  continue

  #print('start step: ', step, " ", datetime.datetime.now())
  print('start step:', step, '02_py_start_time:', time.time())
  sess.run(target_op_02)
  print('end   step:', step, '02_py_end_time:', time.time())
  #print('end   step: ', step, " ", datetime.datetime.now())

  if step >= 3:
    sess_run_time_end = time.time()
    sess_run_time_total += (sess_run_time_end - sess_run_time_start)
    print('step:', step, '02_py_step_time:', "%.6f" % (sess_run_time_end - sess_run_time_start))

end_02 = time.time()

#######################################################################

#start_03 = 0
#for step in range(num_steps):
#  if step == 3:
#    # warm up
#    print('start timer')
#    start_03 = time.time()
#
#  if step >= 3:
#    sess_run_time_start = time.time()
#
#  #print('start step: ', step, " ", datetime.datetime.now())
#  print('start step:', step, '03_py_start_time:', time.time())
#  sess.run(target_op_03)
#  print('end   step:', step, '03_py_end_time:', time.time())
#  #print('end   step: ', step, " ", datetime.datetime.now())
#
#  if step >= 3:
#    sess_run_time_end = time.time()
#    sess_run_time_total += (sess_run_time_end - sess_run_time_start)
#    print('step:', step, '03_py_step_time:', "%.6f" % (sess_run_time_end - sess_run_time_start))
#
#end_03 = time.time()

#######################################################################

#start_04 = 0
#for step in range(num_steps):
#  if step == 3:
#    # warm up
#    print('start timer')
#    start_04 = time.time()
#
#  if step >= 3:
#    sess_run_time_start = time.time()
#
#  #print('start step: ', step, " ", datetime.datetime.now())
#  print('start step:', step, '04_py_start_time:', time.time())
#  sess.run(target_op_04)
#  print('end   step:', step, '04_py_end_time:', time.time())
#  #print('end   step: ', step, " ", datetime.datetime.now())
#
#  if step >= 3:
#    sess_run_time_end = time.time()
#    sess_run_time_total += (sess_run_time_end - sess_run_time_start)
#    print('step: ', step, ' step_time: ', "%.6f" % (sess_run_time_end - sess_run_time_start))
#
#end_04 = time.time()

#######################################################################

#start_05 = 0
#for step in range(num_steps):
#  if step == 3:
#    # warm up
#    print('start timer')
#    start_05 = time.time()
#
#  if step >= 3:
#    sess_run_time_start = time.time()
#
#  #print('start step: ', step, " ", datetime.datetime.now())
#  print('start step:', step, '05_py_start_time:', time.time())
#  sess.run(target_op_05)
#  print('end   step:', step, '05_py_end_time:', time.time())
#  #print('end   step: ', step, " ", datetime.datetime.now())
#
#  if step >= 3:
#    sess_run_time_end = time.time()
#    sess_run_time_total += (sess_run_time_end - sess_run_time_start)
#    print('step: ', step, ' step_time: ', "%.6f" % (sess_run_time_end - sess_run_time_start))
#end_05 = time.time()

#######################################################################

#print('Baseline, seq, total time: ', end_01 - start_01)
print('Baseline, seq, total time: ', end_01 - start_01 + end_02 - start_02)
#print('Baseline, seq, total time: ', end_01 - start_01 + end_02 - start_02 + end_03 - start_03)
#print('Baseline, seq, total time: ', end_01 - start_01 + end_02 - start_02 + end_03 - start_03 + end_04 - start_04)
#print('Baseline, seq, total time: ',
#      end_01 - start_01 +
#      end_02 - start_02 +
#      end_03 - start_03 +
#      end_04 - start_04 +
#      end_05 - start_05
#      )
print('Baseline, seq, sess total time: ', sess_run_time_total)

print("Finished!")