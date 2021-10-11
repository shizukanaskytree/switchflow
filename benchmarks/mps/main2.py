import random
import os
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

import string
import time
import datetime
import numpy as np

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

from tensorflow.python.client import device_lib
print('GPU device info: \n', device_lib.list_local_devices())

class Benchmark():
    def __init__(self, model_name, height=224, width=224, num_channels=3,
        batch_size=32, log_steps=10, data_format='channels_first',
        skip_eval=True, only_inference=False, inter_op_threads=36,
        train_steps=100, inference_steps=10, create_frozen_model=False):
        """
        Args:
            model_name: string; The model name.
            is_vanilla: boolean; Whether to use vanilla TF or not.
            priority: 1/2; Set 1 (low) or 2 (high) when we test our design.
            height: int;
            width: int;
            num_channels: int;
            batch_size: int;
            log_steps: int; interval steps between logging
            data_format: string; 'channels_first' or 'channels_last'
            skip_eval: boolean; skip evaluation or not. To skip model.evaluate(...) or not.
            only_inference: If we set only_inference as True, we don't do training but only predict.
            arrival_time_interval: Time interval simulation arrival distribution, unit: second
        """
        self.model_name = model_name
        # https://www.tensorflow.org/api_docs/python/tf/keras/applications
        self.models = {
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
            'VGG19': vgg19.VGG19
        }
        self.HEIGHT = height
        self.WIDTH = width
        self.NUM_CHANNELS = num_channels
        self.NUM_CLASSES = 1000
        self.BATCH_SIZE = batch_size
        self.inter_op_threads = inter_op_threads
        self.only_inference = only_inference
        self.train_steps = train_steps
        self.inference_steps = inference_steps
        self.create_frozen_model = create_frozen_model
        self.NUM_EPOCHS = 1
        self.NUM_IMAGES = {'train': 50000,
                    'validation': 10000}

        self.L2_WEIGHT_DECAY = 2e-4
        self.BATCH_NORM_DECAY = 0.997
        self.BATCH_NORM_EPSILON = 1e-5

        self.skip_eval = skip_eval
        self.epochs_between_evals = 1
        self.log_steps = log_steps
        self.TRAIN_TOP_1_ = 'training_accuracy_top_1'
        self.data_format = data_format


    def get_synth_input_fn(self, height, width, num_channels, num_classes,
                            dtype=tf.float32, drop_remainder=True):
        """Returns an input function that returns a dataset with random data.
        This input_fn returns a data set that iterates over a set of random data and
        bypasses all preprocessing, e.g. jpeg decode and copy. The host to device
        copy is still included. This used to find the upper throughput bound when
        tuning the full input pipeline.
        Args:
        height: Integer height that will be used to create a fake image tensor.
        width: Integer width that will be used to create a fake image tensor.
        num_channels: Integer depth that will be used to create a fake image tensor.
        num_classes: Number of classes that should be represented in the fake labels
            tensor
        dtype: Data type for features/images.
        drop_remainder: A boolean indicates whether to drop the remainder of the
            batches. If True, the batch dimension will be static.
        Returns:
        An input_fn that can be used in place of a real one to return a dataset
        that can be used for iteration.
        """
        def input_fn(is_training, data_dir, batch_size, *args, **kwargs):
            # Xception is only be used with the data format `(width, height, channels)`.
            if self.data_format == 'channels_first':
                shape = [num_channels, height, width]
            else:
                # 'channels_last'
                shape = [height, width, num_channels]

            inputs = tf.random.truncated_normal(shape=shape,
                                                dtype=dtype,
                                                mean=127,
                                                stddev=60,
                                                name='synthetic_inputs')
            labels = tf.random.uniform([1],
                                        minval=0,
                                        maxval=num_classes - 1,
                                        dtype=tf.int32,
                                        name='synthetic_labels')
            labels = tf.cast(labels, dtype=tf.float32)
            data = tf.data.Dataset.from_tensors((inputs, labels)).repeat()
            # `drop_remainder`  will make dataset produce outputs with known shapes
            data = data.batch(batch_size, drop_remainder=drop_remainder)
            data = data.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
            return data

        return input_fn

    def get_optimizer(self, learning_rate=0.1):
        return gradient_descent_v2.SGD(learning_rate=learning_rate, momentum=0.9)

    def get_callbacks(self):
        time_callback = TimeHistory(self.model_name, self.BATCH_SIZE, self.log_steps)
        callbacks = [time_callback]
        return callbacks

    def build_stats(self, history, eval_output, callbacks):
        """Normalizes and returns dictionary of stats.

        Args:
            history: Results of the training step. Supports both categorical_accuracy
                and sparse_categorical_accuracy.
            eval_output: Output of the eval step. Assumes first value is eval_loss and
                second value is accuracy_top_1.
            callbacks: a list of callbacks which might include a time history callback
                used during keras.fit.

        Returns:
            Dictionary of normalized results.
        """
        stats = {}
        if eval_output:
            stats['accuracy_top_1'] = eval_output[1].item()
            stats['eval_loss'] = eval_output[0].item()

        if history and history.history:
            train_hist = history.history
            # Gets final loss from training.
            stats['loss'] = train_hist['loss'][-1].item()
            # Gets top_1 training accuracy.
            if 'categorical_accuracy' in train_hist:
                stats[self.TRAIN_TOP_1_] = train_hist['categorical_accuracy'][-1].item()
            elif 'sparse_categorical_accuracy' in train_hist:
                stats[self.TRAIN_TOP_1_] = train_hist['sparse_categorical_accuracy'][-1].item()

        if not callbacks:
            return stats

        # Look for the time history callback which was used duriing keras.fit
        for callback in callbacks:
            if isinstance(callback, TimeHistory):
                timestamp_log = callback.timestamp_log
                stats['step_timestamp_log'] = timestamp_log
                stats['train_finish_time'] = callback.train_finish_time
                if len(timestamp_log) > 1:
                    stats['avg_exp_per_second'] = (
                        callback.batch_size * callback.log_steps *
                        (len(callback.timestamp_log) - 1) /
                        (timestamp_log[-1].timestamp - timestamp_log[0].timestamp))

        return stats

    def freeze_graph(self, graph, session, output):
        """
        :param graph: The target graph to be frozen
        :param session: The current session
        :param output: Nodes of the graph that need to be frozen
        :return:
        """
        with graph.as_default():
            graphdef_inf = tf.graph_util.remove_training_nodes(graph.as_graph_def())
            graphdef_frozen = tf.graph_util.convert_variables_to_constants(session, graphdef_inf, output)

            if self.is_vanilla:
                frozen_model_exported_path = './frozen_model'
            else:
                frozen_model_exported_path = './frozen_model_new_tf'

            frozen_model_name = self.model_name + '_frozen_model.pb'
            graph_io.write_graph(graphdef_frozen,
                                frozen_model_exported_path,
                                frozen_model_name,
                                as_text=False)

    #############################################################

    def run_training(self):
        config = tf.ConfigProto(inter_op_parallelism_threads=self.inter_op_threads)
        config.gpu_options.allow_growth=True
        sess = tf.Session(config=config)
        tf.keras.backend.set_session(sess)

        if self.create_frozen_model:
            # Ignore dropout at inference, this line is important
            tf.keras.backend.set_learning_phase(0)

        tf.keras.backend.set_image_data_format(self.data_format)

        # 单机单卡模式, 没有设置分布式的逻辑
        # - strategy = self.get_distribution_stragegy()
        # - strategy_scope = self.get_strategy_scope(strategy)

        input_fn = self.get_synth_input_fn(height=self.HEIGHT,
                                        width=self.WIDTH,
                                        num_channels=self.NUM_CHANNELS,
                                        num_classes=self.NUM_CLASSES,
                                        dtype=tf.float32,
                                        drop_remainder=True)

        train_input_dataset = input_fn(
            is_training=True,
            data_dir="",  # synthetic dataset has no filepath.
            batch_size=self.BATCH_SIZE,
            num_epochs=self.NUM_EPOCHS,
            dtype=tf.float32,
            drop_remainder=True)

        eval_input_dataset = input_fn(
            is_training=False,
            data_dir="", # synthetic dataset has no filepath.
            batch_size=self.BATCH_SIZE,
            num_epochs=self.NUM_EPOCHS,
            dtype=tf.float32,
            drop_remainder=True)

        optimizer = self.get_optimizer()

        model = self.models[self.model_name](weights=None)

        model.compile(
            loss='categorical_crossentropy',
            optimizer=optimizer,
            metrics=(['categorical_accuracy'])
        )

        callbacks = self.get_callbacks()

        # We can also specify steps here:
        train_steps = self.train_steps
        train_epochs = self.NUM_EPOCHS

        num_eval_steps = self.NUM_IMAGES['validation'] // self.BATCH_SIZE
        validation_data = eval_input_dataset

        tf.logging.set_verbosity(v=2)

        history = model.fit(train_input_dataset,
                            epochs=train_epochs,
                            steps_per_epoch=train_steps,
                            callbacks=callbacks,
                            validation_steps=num_eval_steps,
                            validation_data=validation_data,
                            validation_freq=self.epochs_between_evals,
                            verbose=2)

        if self.create_frozen_model:
            #INPUT_NODE = model.inputs[0].op.name
            #OUTPUT_NODE = model.outputs[0].op.name
            self.freeze_graph(sess.graph, sess,
                            [out.op.name for out in model.outputs])

        eval_output = None
        if not self.skip_eval:
            eval_output = model.evaluate(eval_input_dataset,
                                        steps=num_eval_steps,
                                        verbose=2)

        print('Done!')


    # Inference Benchmark routine
    def run_frozen_model_inference_helper(self):
        # Specify TF backend data format
        tf.keras.backend.set_image_data_format(self.data_format)
        # Restore graph from .pb file
        graph_def = tf.GraphDef()

        if self.is_vanilla:
            frozen_file = './frozen_model/' + self.model_name + '_frozen_model.pb'
        else:
            frozen_file = './frozen_model_new_tf/' + self.model_name + '_frozen_model.pb'

        with tf.gfile.GFile(frozen_file, "rb") as f:
            graph_def.ParseFromString(f.read())

        graph = tf.Graph()

        with graph.as_default():
            net_input, net_out = tf.import_graph_def(graph_def,
                #return_elements=["input_1", "fc1000/Softmax"] # this is for ResNet50 only
                return_elements=[graph_def.node[0].name, graph_def.node[-1].name])

        # Input data
        if self.data_format == 'channels_first':
            shape = [self.NUM_CHANNELS, self.HEIGHT, self.WIDTH]
        else:
            # 'channels_last'
            shape = [self.HEIGHT, self.WIDTH, self.NUM_CHANNELS]

        inputs = np.random.rand(self.BATCH_SIZE, shape[0], shape[1], shape[2])

        config = tf.ConfigProto(inter_op_parallelism_threads=self.inter_op_threads)
        sess = tf.Session(graph=graph, config=config)
        tf.keras.backend.set_session(sess)

        # only 1 step for experiment is bad since cold-start overhead
        num_warmup = 3
        for n in range(num_warmup):
            sess.run(net_out.outputs[0], feed_dict={net_input.outputs[0]:inputs})
        print('Warmup Done!')

        return sess, net_input, net_out, inputs

    # Do the inference for benchmarking
    def do_run_inference(self, sess, net_input, net_out, inputs):
        tail95_done_at = 0
        for i in range(self.inference_steps):
            out = sess.run(net_out.outputs[0], feed_dict={net_input.outputs[0]:inputs})
            #print(np.argmax(out))
            if i == (self.inference_steps*0.95 - 1):
                tail95_done_at = time.time()
                #print('95% percentile time done at: ', )

        # No need since we don't use keras, so no keras session
        #if not self.is_vanilla:
        #  tf.keras.backend.clear_only_session()
        sess.close()
        sess = None

        infer_done_at = time.time()
        print('Inference Done at: ', infer_done_at)
        return infer_done_at, tail95_done_at

    def run_inference(self):
        graph, net_input, net_out, inputs = self.run_frozen_model_inference_helper()
        self.do_run_inference(graph, net_input, net_out, inputs)

##################################################################
# Auxiliary Classes
##################################################################
class SyntheticDataset(object):
    def __init__(self, dataset, split_by=1):
        self._input_data = {}
        # Since dataset.take(1) doesn't have GPU kernel.
        with tf.device('device:CPU:0'):
            tensor = tf.data.experimental.get_single_element(dataset.take(1))
        flat_tensor = tf.nest.flatten(tensor)
        variable_data = []
        self._initializers = []
        for t in flat_tensor:
            rebatched_t = tf.split(t, num_or_size_splits=split_by, axis=0)[0]
            assert rebatched_t.shape.is_fully_defined(), rebatched_t.shape
            v = tf.compat.v1.get_local_variable(self.random_name(), initializer=rebatched_t)
            variable_data.append(v)
            self._initializers.append(v.initializer)
        self._input_data = tf.nest.pack_sequence_as(tensor, variable_data)

    def get_next(self):
        return self._input_data

    def initialize(self):
        if tf.executing_eagerly():
            return tf.no_op()
        else:
            return self._initializers

    def random_name(self, size=10, chars=string.ascii_uppercase + string.digits):
        return ''.join(random.choice(chars) for _ in range(size))

class BatchTimestamp(object):
    """A structure to store batch time stamp."""

    def __init__(self, batch_index, timestamp):
        self.batch_index = batch_index
        self.timestamp = timestamp

    def __repr__(self):
        return "'BatchTimestamp<batch_index: {}, timestamp: {}>'".format(
        self.batch_index, self.timestamp)


class TimeHistory(tf.keras.callbacks.Callback):
    """Callback for Keras models."""

    def __init__(self, model_name, batch_size, log_steps):
        """Callback for logging performance.
        Args:
        batch_size: Total batch size.
        log_steps: Interval of steps between logging of batch level stats.
        """
        self.model_name = model_name
        self.batch_size = batch_size
        super(TimeHistory, self).__init__()
        self.log_steps = log_steps
        self.global_steps = 0

        # Logs start of step 1 then end of each step based on log_steps interval.
        self.timestamp_log = []

        # Records the time each epoch takes to run from start to finish of epoch.
        self.epoch_runtime_log = []

    # The following functions follow the logic execution order
    def on_train_end(self, logs=None):
        self.train_finish_time = time.time()

    def on_epoch_begin(self, epoch, logs=None):
        self.epoch_start = time.time()

    def on_batch_begin(self, batch, logs=None):
        self.global_steps += 1
        if self.global_steps == 1:
            self.start_time = time.time()
            self.timestamp_log.append(BatchTimestamp(self.global_steps, self.start_time))

    def on_batch_end(self, batch, logs=None):
        """Records elapse time of the batch and calculates examples per second."""
        if self.global_steps % self.log_steps == 0:
            timestamp = time.time()
            elapsed_time = timestamp - self.start_time
            examples_per_second = (self.batch_size * self.log_steps) / elapsed_time
            self.timestamp_log.append(BatchTimestamp(self.global_steps, timestamp))
            tf.compat.v1.logging.info(
                "BenchmarkMetric: {%s\t'global step':%d, 'time_taken': %f,"
                "'examples_per_second': %f, timestamp: %f}" %
                (self.model_name, self.global_steps,
                elapsed_time, examples_per_second, timestamp))
            self.start_time = timestamp

    def on_epoch_end(self, epoch, logs=None):
        epoch_run_time = time.time() - self.epoch_start
        self.epoch_runtime_log.append(epoch_run_time)
        tf.compat.v1.logging.info(
            "BenchmarkMetric: {'epoch':%d, 'time_taken': %f}" %
            (epoch, epoch_run_time))

# This class is used in the model.predict metrics measurement
class InferenceCallback(tf.keras.callbacks.Callback):
    def __init__(
        self,
        model_name,
        batch_size,
        log_steps,
    ):
        self.start_time = 0
        self.end_time = 0
        self.global_steps = 0
        self.log_steps = log_steps
        self.batch_size = batch_size
        self.model_name=model_name

    def on_predict_batch_begin(self, batch, logs=None):
        #print('Predicting: batch {} begins at {}'.format(batch, datetime.datetime.now().time()))
        self.global_steps += 1
        if self.global_steps == 1:
            self.start_time = time.time()

    def on_predict_batch_end(self, batch, logs=None):
        #print('Predicting: batch {} ends at {}'.format(batch, datetime.datetime.now().time()))
        if self.global_steps % self.log_steps == 0:
            timestamp = time.time()
            elapsed_time = timestamp - self.start_time
            self.time_interval = 0
            examples_per_second = (self.batch_size * self.log_steps) / elapsed_time
            tf.compat.v1.logging.info(
                "BenchmarkMetric: {'%s\t global step':%d, 'infer time_taken': %f,"
                "'examples_per_second': %f}" %
                (self.model_name, self.global_steps, elapsed_time, examples_per_second))
            self.start_time = timestamp

def next_time(poisson_rate):
    return random.expovariate(poisson_rate)

def run_resnet50_training():
    net= 'InceptionV3'

    Benchmark(model_name=net,
              batch_size=32,
              train_steps=1000,
              height=224, width=224,
              create_frozen_model=False,
              data_format='channels_first' if net != 'MobileNetV2' else 'channels_last',
              ).run_training()

if __name__ == '__main__':
    run_resnet50_training()