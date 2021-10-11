"""dataset.py

This module implements functions for reading ImageNet (ILSVRC2012)
dataset in TFRecords format.
"""


import os
from functools import partial

import tensorflow as tf
import pathlib

from .image_processing import preprocess_image, resize_and_rescale_image

# Number of parallel works for generating training/validation data
NUM_DATA_WORKERS = 32

# Do image data augmentation or not
DATA_AUGMENTATION = True

def decode_jpeg(image_buffer, scope=None):
  """Decode a JPEG string into one 3-D float image Tensor.

  Args:
      image_buffer: scalar string Tensor.
      scope: Optional scope for name_scope.
  Returns:
      3-D float Tensor with values ranging from [0, 1).
  """
  with tf.name_scope(values=[image_buffer], name=scope,
                     default_name='decode_jpeg'):
    # Decode the string as an RGB JPEG.
    # Note that the resulting image contains an unknown height
    # and width that is set dynamically by decode_jpeg. In other
    # words, the height and width of image is unknown at compile-i
    # time.
    image = tf.image.decode_jpeg(image_buffer, channels=3)

    # After this point, all image pixels reside in [0,1)
    # until the very end, when they're rescaled to (-1, 1).
    # The various adjust_* ops all require this range for dtype
    # float.
    image = tf.image.convert_image_dtype(image, dtype=tf.float32)
    return image


def _parse_fn(example_serialized, is_training):
  """Helper function for parse_fn_train() and parse_fn_valid()

  Each Example proto (TFRecord) contains the following fields:

  image/height: 462
  image/width: 581
  image/colorspace: 'RGB'
  image/channels: 3
  image/class/label: 615
  image/class/synset: 'n03623198'
  image/class/text: 'knee pad'
  image/format: 'JPEG'
  image/filename: 'ILSVRC2012_val_00041207.JPEG'
  image/encoded: <JPEG encoded string>

  Args:
      example_serialized: scalar Tensor tf.string containing a
      serialized Example protocol buffer.

  Returns:
      image_buffer: Tensor tf.string containing the contents of
      a JPEG file.
      label: Tensor tf.int32 containing the label.
      text: Tensor tf.string containing the human-readable label.
  """
  feature_map = {
    'image/encoded': tf.FixedLenFeature([], dtype=tf.string,
                                        default_value=''),
    'image/class/label': tf.FixedLenFeature([], dtype=tf.int64,
                                            default_value=-1),
    'image/class/text': tf.FixedLenFeature([], dtype=tf.string,
                                           default_value=''),
  }
  parsed = tf.parse_single_example(example_serialized, feature_map)
  image = decode_jpeg(parsed['image/encoded'])
  if DATA_AUGMENTATION:
    image = preprocess_image(image, 224, 224, is_training=is_training)
  else:
    image = resize_and_rescale_image(image, 224, 224)
  # The label in the tfrecords is 1~1000 (0 not used).
  # So I think the minus 1 is needed below.
  label = tf.one_hot(parsed['image/class/label'] - 1, 1000, dtype=tf.float32)
  # wxf: from NHWC to NCHW
  image = tf.transpose(image, [2, 0, 1])
  #~wxf
  return (image, label)

def get_dataset(tfrecords_dir, subset, batch_size):
  """Read TFRecords files and turn them into a TFRecordDataset.
  Args:
    tfrecords_dir: e.g., '/data/google_imagenet/raw_data/tf_records/train'
    subset: e.g., 'train'
    batch_size: e.g., 64
  """
  files = tf.matching_files(os.path.join(tfrecords_dir, '%s-*' % subset))
  shards = tf.data.Dataset.from_tensor_slices(files)
  shards = shards.shuffle(tf.cast(tf.shape(files)[0], tf.int64))
  shards = shards.repeat()
  dataset = shards.interleave(tf.data.TFRecordDataset, cycle_length=4)
  dataset = dataset.shuffle(buffer_size=8192)
  parser = partial(
    _parse_fn, is_training=True if subset == 'train' else False)
  #dataset = dataset.apply(
  #    tf.data.experimental.map_and_batch(
  #        map_func=parser,
  #        batch_size=batch_size,
  #        num_parallel_calls=NUM_DATA_WORKERS))
  dataset = dataset.map(map_func=parser,
                        num_parallel_calls=NUM_DATA_WORKERS)
  dataset = dataset.batch(batch_size=batch_size)
  dataset = dataset.prefetch(batch_size)
  return dataset

"""
sess.run(files)

[b'/data/google_imagenet/raw_data/tf_records/train/train-00000-of-01024'
 b'/data/google_imagenet/raw_data/tf_records/train/train-00001-of-01024'
 b'/data/google_imagenet/raw_data/tf_records/train/train-00002-of-01024'
 ...
 b'/data/google_imagenet/raw_data/tf_records/train/train-01021-of-01024'
 b'/data/google_imagenet/raw_data/tf_records/train/train-01022-of-01024'
 b'/data/google_imagenet/raw_data/tf_records/train/train-01023-of-01024']
"""

# wxf: parse JPEG images
def _parse_fn_jpeg(filename, label, is_training = True):
  image = tf.io.read_file(filename)
  image = decode_jpeg(image)
  if DATA_AUGMENTATION:
    image = preprocess_image(image, 224, 224, is_training=is_training)
  else:
    image = resize_and_rescale_image(image, 224, 224)
  # from NHWC to NCHW
  # (4, 224, 224, 3) ==> (4, 3, 224, 224)
  image = tf.transpose(image, [2, 0, 1])
  label = tf.one_hot(label - 1, 1000, dtype=tf.float32)
  return (image, label)

def get_dataset_jpeg(train_dataset_path, label_filename, subset, batch_size):
  """Get dataset from jpeg and convert to TF Dataset.

  :param train_dataset_path: jpeg files dir
  :param label_filename: label dir
  :param subset: 'train' or 'valid' ?
  :param batch_size: batch size
  :return: Dataset instance dataset
  """
  train_dataset_path = pathlib.Path(train_dataset_path)
  train_dataset_folders = train_dataset_path.glob('*')
  train_dataset_folder_name_list = []
  for folder_name in train_dataset_folders:
    # print(folder_name)
    name = folder_name.name
    # print(name_)
    train_dataset_folder_name_list.append(name)

  foldernames_in_label_file = []
  folder_label_mappings = {}
  with open(label_filename, 'r') as f:
    for line in f:
      split_line = line.split()
      foldernames_in_label_file.append(split_line[0])
      # print(split_line[0])
      folder_label_mappings[split_line[0]] = split_line[1]

  #print(len(train_dataset_folder_name_list))
  #print(len(foldernames_in_label_file))

  train_dataset_folder_name_list.sort()
  foldernames_in_label_file.sort()

  # for folder_name, label_folder_name in zip(train_dataset_folder_name_list, foldernames_in_label_file):
  #  assert folder_name == label_folder_name
  # print(folder_name, ':', label_folder_name)

  glob_train_filenames = pathlib.Path(train_dataset_path).glob('*/*')
  train_filenames = []
  train_labels = []
  for file_path in glob_train_filenames:
    train_filenames.append(str(file_path))
    folder_name = (str(file_path).split('/'))[-2]
    # print(folder_name)
    train_labels.append(int(folder_label_mappings[folder_name]))

  # print(list_ds)
  # print(len(train_filenames))
  # print(len(train_labels))

  # files = tf.data.Dataset.list_files(str(train_dataset_path/'*/*'))
  # iter = files.make_initializable_iterator()
  # el = iter.get_next()
  # sess.run(iter.initializer)
  # print(sess.run(el))
  # print(sess.run(el))
  # print(sess.run(el))

  image_label_dataset = tf.data.Dataset.from_tensor_slices(
    (tf.constant(train_filenames), tf.constant(train_labels)))

  parser = partial(
    _parse_fn_jpeg,
    is_training=True if subset == 'train' else False)

  dataset = image_label_dataset.map(map_func=parser,
                              num_parallel_calls=NUM_DATA_WORKERS)
  dataset = dataset.repeat()
  dataset = dataset.batch(batch_size=batch_size)
  dataset = dataset.prefetch(batch_size)
  return dataset