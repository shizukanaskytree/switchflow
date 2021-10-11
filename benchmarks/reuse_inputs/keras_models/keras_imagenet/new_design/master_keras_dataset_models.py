import os
#os.environ['TF_CPP_MIN_VLOG_LEVEL']='0'

#os.environ['TF_SET_REUSE_INPUTS_FLAG'] = '1'
#os.environ['TF_REUSE_INPUT_OP_NAME_MASTER_X'] = 'IteratorGetNext/_23'
#os.environ['TF_REUSE_INPUT_OP_NAME_MASTER_y'] = 'IteratorGetNext/_21'
#os.environ['TF_REUSE_INPUT_OPS_NAME_SUB_X'] = 'X01'
#os.environ['TF_REUSE_INPUT_OPS_NAME_SUB_y'] = 'y01'

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

from load_imagenet_inputs.dataset import get_dataset, get_dataset_jpeg

# new tf execution priority
LOW = 1
HIGH = 2

graph_session_id = {
  'graph_00': 0, # master
  'graph_01': 1, # sub 01
  'graph_02': 2, # sub 02
  'graph_03': 3, # sub 03
}
#######################################
# number of threads == num_sessions !!!
num_sessions = 2
#######################################

class MasterModel:
  def __init__(self):
    self.model_name = 'NASNetMobile'
    self.task_name = 'train'
    self.train_steps = 103 # extra 3 steps for skipping warmup
    #self.stop_at_step = 103 # extra 3 steps for skipping warmup
    self.BATCH_SIZE = 128
    self.HEIGHT = 224
    self.WIDTH = 224
    self.NUM_CHANNELS = 3
    self.data_format = 'channels_first'
    self.shape = (self.NUM_CHANNELS, self.HEIGHT, self.WIDTH) # Channel GPU prefers channels_first.
    self.DO_TRAIN = True
    self.dtype = tf.float32
    self.NUM_CLASSES = 1000
    self.inter_op_threads_num = 32 # wxf:debug, set 1

    self.models = {
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

  def BuildAndRunGraph(self, graph_name, X_name, y_name):
    tf.set_execution_priority(HIGH) # a must for new design for tf Dataset

    # Create Session since keras model will get or create if
    # not existing Session instance for backend tf Dataset usage
    config = tf.ConfigProto(inter_op_parallelism_threads=self.inter_op_threads_num,
                            session_id=graph_session_id[graph_name],
                            num_sessions=num_sessions)
    # comment: token turn: 2 graphs, token id = 0, 1, i.e., session run token id
    # I assign token 0 to master; 1 to subsidiary.

    # wxf: for debug, one graph
    #config = tf.ConfigProto(inter_op_parallelism_threads=self.inter_op_threads_num,
    #                        session_id=0,
    #                        num_sessions=1)

    sess = tf.Session(config=config)
    # A must if use tf Dataset, set the session as the global session
    tf.keras.backend.set_session(session=sess)
    tf.keras.backend.set_image_data_format(self.data_format)

    # ===------===
    # import data
    # ===------===
    #DATASET_DIR = "/data/google_imagenet/raw_data/tf_records/train"
    #dataset = get_dataset(DATASET_DIR, 'train', batch_size=self.BATCH_SIZE)

    JPEG_DATASET_DIR = '/data1/google_imagenet/raw_data/train_slices'
    LABEL_FILENAME = '/data1/google_imagenet/raw_data/folder_label_mapping.txt'
    dataset = get_dataset_jpeg(JPEG_DATASET_DIR, LABEL_FILENAME, 'train', batch_size=self.BATCH_SIZE)
    #dataset = get_dataset_jpeg(JPEG_DATASET_DIR, LABEL_FILENAME, 'infer', batch_size=self.BATCH_SIZE)

    iterator = dataset.make_initializable_iterator()
    batch_xs_t, batch_ys_t = iterator.get_next()

    #######################################################################
    # rename, but it is optimized out in the execution graph backend.
    batch_xs = tf.identity(batch_xs_t, name=X_name)
    batch_ys = tf.identity(batch_ys_t, name=y_name)
    #######################################################################
    # 1.
    # shape of batch_xs= (?, 3, 224, 224)
    # NHWC (N, Height, width, channel) is the TensorFlow default and
    # NCHW is the optimal format to use for NVIDIA cuDNN.
    # If TensorFlow is compiled with the Intel MKL optimizations,
    # many operations will be optimized and support NCHW.
    # Otherwise, some operations are not supported on CPU when using NCHW.
    # On GPU, NCHW is faster.Mar 7, 2017

    # 2.
    # change the name of the tensor in tensorflow
    # https://stackoverflow.com/questions/38626424/how-to-assign-new-name-or-rename-an-existing-tensor-in-tensorflow
    # figure: https://keep.google.com/u/1/#NOTE/1iq5oI5w1gUXcbdzEf7Dx_AhCrY20TMH2XCUao7-y11OHzmOidPf15m_fV1s2wA
    # ===------===
    # ~import data
    # ===------===

    #if DO_TRAIN:
    #  model = resnet50.ResNet50(weights=None, input_tensor=batch_xs)
    #else:
    #  model = resnet50.ResNet50(weights='imagenet',
    #                            include_top=True, # default is also True...
    #                            input_shape=shape)

    # This function will create Session!
    #model = mobilenet_v2.MobileNetV2(weights=None, input_tensor=batch_xs)
    model = self.models[self.model_name](weights=None, input_tensor=batch_xs)

    # 1. training network
    loss = tf.reduce_mean(categorical_crossentropy(batch_ys, model.output))
    # 1.
    # https://gombru.github.io/2018/05/23/cross_entropy_loss/
    train_step_op = tf.train.GradientDescentOptimizer(0.5).minimize(loss)

    # 2. Forward pass
    # Evaluate model
    correct_pred = tf.equal(tf.argmax(model.output, 1), tf.argmax(batch_ys, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32), name='accuracy')

    target_op = accuracy
    # 1. ResNet model
    # /home/wxf/.local/lib/python3.6/site-packages/keras_applications/resnet50.py

    # 2.
    # model.output
    # https://blog.keras.io/keras-as-a-simplified-interface-to-tensorflow-tutorial.html

    # 2.1
    # vars(model)
    # [<tf.Tensor 'fc1000/Softmax:0' shape=(?, 1000) dtype=float32>]
    # How to get all members of a class in python?
    # https://stackoverflow.com/questions/5969806/print-all-properties-of-a-python-class

    #callbacks = [tf.keras.callbacks.TensorBoard(log_dir='./logs'),
    #             MyCustomCallback()]
    #callbacks = None

    ## tfboard
    #current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    #train_log_dir = 'logs/' + current_time + '/train'

    ## collect runtime statistics
    #run_options = None
    #run_metadata = None
    #train_writer = tf.summary.FileWriter(train_log_dir, sess.graph)

    init_op = tf.global_variables_initializer()

    sess.run(init_op)
    sess.run(iterator.initializer)

    # timing temp variables
    t_sess_s = 0
    t_sess_e = 0
    t_step_s = 0
    t_step_e = 0
    t_step_total = 0

    for step in range(self.train_steps):
      print(threading.currentThread().getName(), '**********************************')
      #time.sleep(5)

      if step == 3:
        t_sess_s = time.time()
        print('start timer')

      if step >= 3:
        t_step_s = time.time()

      #if step == 10:
      #  # collect execution stats
      #  run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
      #  run_metadata = tf.RunMetadata()
      #  sess.run(target_op,
      #           options=run_options,
      #           run_metadata=run_metadata)
      #  train_writer.add_run_metadata(run_metadata, 'step%d' % step)
      #  # Create the Timeline object, and write it to a json
      #  tl = timeline.Timeline(run_metadata.step_stats)
      #  ctf = tl.generate_chrome_trace_format()
      #  timeline_folder = 'timeline_logs'
      #  timeline_filename = 'master_timeline_' + current_time + '.json'
      #  if not os.path.exists(timeline_folder):
      #    os.makedirs(timeline_folder)
      #  timeling_logs_path_file = os.path.join(os.getcwd(), timeline_folder, timeline_filename)
      #  with open(timeling_logs_path_file, 'w') as f:
      #    f.write(ctf)
      #  continue

      #print('master start step: ', step, " ", datetime.datetime.now())
      # new API setting for token turn step counting when we start doing real train or inference
      step_run_options = tf.RunOptions(real_step_start=True)
      sess.run([target_op], options=step_run_options)
      #print('master end   step: ', step, " ", datetime.datetime.now())

      if step >= 3:
        t_step_e = time.time()
        t_step_total += (t_step_e - t_step_s)
        print('>>> ', threading.currentThread().getName(), 'step', step, "%.6f" % (t_step_e - t_step_s))

      #if step == self.stop_at_step:
      #  print('>>> ', threading.currentThread().getName(),
      #        'Total step time: ', t_step_total,
      #        'Time stamp: ', time.time())

    t_sess_e = time.time()
    print('>>> ', threading.currentThread().getName(),
          ': Done!', ' Total Time:', t_sess_e - t_sess_s,
          'Total step time: ', t_step_total,
          'Time stamp: ', time.time())


if __name__ == '__main__':
  master = MasterModel()
  master.BuildAndRunGraph('graph_00', 'X00', 'y00')