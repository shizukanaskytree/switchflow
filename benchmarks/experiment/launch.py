import os
# Normal one

os.environ['TF_CPP_MIN_VLOG_LEVEL']='0'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
# Note:
# Env: 1080, 2080 and cpu
# if os.environ['CUDA_VISIBLE_DEVICES'] = '1' : GeForce GTX 1080 Ti

##########################################################################
os.environ['OMP_NUM_THREADS'] = '32'
##########################################################################

import threading
import time
from benchmarks import Benchmark
from benchmarks import next_time

HIGH = 2
LOW = 1

##########################################################################
# training(1 thread) speed: avg images per second
# Experiment 1
##########################################################################
def run_experiment1():
  is_vanilla = True

  net_list = ['MobileNetV2']
  ##net_list = ['ResNet50', 'VGG16', 'VGG19', 'DenseNet121',
  ##            'DenseNet169', 'DenseNet201', 'InceptionResNetV2',
  ##            'InceptionV3', 'MobileNetV2']
  for net in net_list:
    Benchmark(model_name=net,
              is_vanilla=is_vanilla,
              priority=None,
              batch_size=32,
              train_steps=120,
              #data_format='channels_first' if net != 'MobileNetV2' else 'channels_last', # training should channels_first.
              ).run()
    print('============= ' + net+ ' ============= Done!')


##########################################################################
# Co-run training - vanilla:
# Experiment 2 - 1
##########################################################################
def experiment2_1(low_model_name, high_model_name):
  low = Benchmark(model_name=low_model_name, is_vanilla=True, priority=None,
                  train_steps=1000,
                  data_format='channels_first' if low_model_name != 'MobileNetV2' else 'channels_last',
                  )
  t2 = threading.Thread(target=low.run, name="low_t2")
  t2.start()

  # Simulation of preemption of high priority task.
  time.sleep(55)

  high = Benchmark(model_name=high_model_name, is_vanilla=True, priority=None,
                   train_steps=100,
                   data_format='channels_first' if high_model_name != 'MobileNetV2' else 'channels_last',
                   )
  t1 = threading.Thread(target=high.run, name="high_t1")
  t1.start()

  t1.join()
  t2.join()

def run_experiment2_1():
  low_model_name = 'VGG16'
  net_list = ['DenseNet169']
  #net_list = ['InceptionV3', 'MobileNetV2']
  ##net_list = ['ResNet50', 'VGG16', 'VGG19', 'DenseNet121', 'DenseNet169',
  ##            'DenseNet201', 'InceptionResNetV2', 'InceptionV3', 'MobileNetV2']
  for net in net_list:
    experiment2_1(low_model_name=low_model_name, high_model_name=net)


##########################################################################
# Co-run training - new:
# Experiment 2 - 2
##########################################################################
def experiment2_2(background_model_name, high_model_name):
  is_vanilla = False
  low = Benchmark(model_name=background_model_name, is_vanilla=is_vanilla, priority=LOW,
                  train_steps=3660,
                  #data_format='channels_first' if background_model_name!= 'MobileNetV2' else 'channels_last',
                  data_format='channels_first',
                  #data_format='channels_last',
                  log_steps=10)
  # Note:
  # 1.
  # cpu + gpu : 2020-05-19 20:46:47.376633: E tensorflow/core/common_runtime/executor.cc:436] Executor failed to create kernel. Invalid argument: Default MaxPoolingOp only supports NHWC on device type CPU
  # build TF with mkl!

  t2 = threading.Thread(target=low.run, name="low_t2")
  t2.start()

  # Simulation of preemption of high priority task.
  time.sleep(71)

  high = Benchmark(model_name=high_model_name, is_vanilla=is_vanilla, priority=HIGH,
                   train_steps=500, height=299, width=299,
                   data_format='channels_first' if high_model_name != 'MobileNetV2' else 'channels_last',
                   log_steps=10)
  t1 = threading.Thread(target=high.run, name="high_t1")
  t1.start()

  t1.join()
  t2.join()

def run_experiment2_2():
  #low_net_list = ['ResNet50', 'VGG16', 'VGG19', 'DenseNet121', 'DenseNet169',
  #                'DenseNet201', 'InceptionResNetV2', 'InceptionV3', 'MobileNetV2']
  #high_net_list = ['ResNet50', 'VGG16', 'VGG19', 'DenseNet121', 'DenseNet169', 'DenseNet201', 'InceptionResNetV2', 'InceptionV3', 'MobileNetV2']
  low_net_list = ['ResNet50']
  high_net_list = ['InceptionV3']
  for background in low_net_list:
    for high in high_net_list:
      if background == high:
        continue
      else:
        experiment2_2(background_model_name=background, high_model_name=high)
        print('====================== The Above Done =========================')


##########################################################################
# Co-run training(1 thread) and inference(1 thread):
# Experiment 3
##########################################################################
def experiment3(net, batch_size, inference_steps, is_vanilla):
  # -----------------------------------------------------------------------
  # Training task
  # -----------------------------------------------------------------------
  low = Benchmark(model_name='ResNet50',
                  is_vanilla=is_vanilla,
                  priority=LOW,
                  train_steps=320,
                  data_format='channels_first',
                  skip_eval=True)

  t2 = threading.Thread(target=low.run, name="low_t2")
  t2.start()
  # -----------------------------------------------------------------------

  # -----------------------------------------------------------------------
  # warm up training
  # -----------------------------------------------------------------------
  time.sleep(48)
  # 1.
  # batch size = 1 :
  # 48 is good for ResNet50 , vanilla TF
  # 20, 15 are good for VGG16, vanilla TF
  # 25 is good for VGG16, new TF. NOTE: terminate after half and continue
  # 20 is good for MobileNetV2, vanilla TF, 320 training steps
  # 30 is good for MobileNetV2, new TF, 320 steps

  # 2.
  # batch size = 16 :
  # VGG16 is the background training task
  # 30 is good for vanilla TF most except InceptionResNetV2, which needs 22 , 320 steps is good
  # 30 is good for new TF.

  # 3.
  # batch size = 32;
  # VGG16 as the background trainig task
  # 22 is good, 320 steps is good for vanilla TF
  # 30 for new TF, 320 steps for new TF

  # 4.
  # batch size = 64
  # VGG16 as the background trainig task
  # 22 is good, 320 steps for vinalla TF
  # 30 is good, 320 steps for new TF
  # -----------------------------------------------------------------------

  # -----------------------------------------------------------------------
  # Inference requests
  # -----------------------------------------------------------------------
  infer = \
    Benchmark(model_name=net,
              is_vanilla=is_vanilla,
              priority=HIGH,
              batch_size=batch_size,
              data_format='channels_first' if net != 'MobileNetV2' else 'channels_last',
              inference_steps=inference_steps,
              inter_op_threads=36,
              )

  # Prepare inference data and warmup it to get rid of long warmup time
  sess, net_input, net_out, inputs = infer.run_frozen_model_inference_helper()

  # Do benchmarking
  start_inference_time = time.time()
  print('Inference Requests Start at: ', time.time())

  end_inference_time, tail95_done_at = infer.do_run_inference(sess, net_input, net_out, inputs)

  print('====================================================')
  print(net,
        ', batch size: ', batch_size,
        ', inference steps: ', inference_steps,
        ', total time: ', end_inference_time - start_inference_time,
        ', 95% total time: ', tail95_done_at - start_inference_time)
  print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')

  t2.join()
  print('========================The Above Done============================')
  # -----------------------------------------------------------------------


def run_experiment3():
  is_vanilla = False
  #is_vanilla = True

  net_list = ['ResNet50', 'VGG16', 'VGG19', 'DenseNet121', 'DenseNet169',
              'InceptionResNetV2', 'InceptionV3', 'MobileNetV2']
  #net_list = ['InceptionResNetV2', 'InceptionV3', 'MobileNetV2'] # debug
  #net_list = ['InceptionV3', 'MobileNetV2'] # debug

  #batch_size_list = [1, 16, 32, 64, 128]
  batch_size_list = [64] # debug

  inference_steps = 100
  # 1.
  # inference step, I want it to be 1 for the setting of batch size=1, inference steps=1

  for net in net_list:
    for batch_size in batch_size_list:
      experiment3(net=net, batch_size=batch_size,
                  inference_steps=inference_steps,
                  is_vanilla=is_vanilla)
# -----------------------------------------------------------------------

##########################################################################
# Training time measurement
# Experiment 4
##########################################################################
def run_experiment4_vanilla_training():
  net_list = ['ResNet50', 'VGG16', 'VGG19', 'DenseNet121', 'DenseNet169',
              'DenseNet201', 'InceptionResNetV2', 'InceptionV3', 'MobileNetV2']
  is_vanilla = True
  for net in net_list:
    Benchmark(model_name=net, is_vanilla=is_vanilla, priority=None,
              train_steps=150, log_steps=1).run()

def run_experiment4_vanilla_co_run_training():
  pass


##########################################################################
# Experiment 5:
# Get weights size
##########################################################################
def run_experiment5():
  is_vanilla = False
  net = 'MobileNetV2'
  high_model_name = 'VGG16'

  low = Benchmark(model_name=net, is_vanilla=is_vanilla, priority=LOW,
                  train_steps=300,
                  data_format='channels_first' if net!= 'MobileNetV2' else 'channels_last',
                  log_steps=1)
  t2 = threading.Thread(target=low.run, name="low_t2")
  t2.start()

  time.sleep(42)

  high = Benchmark(model_name=high_model_name, is_vanilla=is_vanilla, priority=HIGH,
                   train_steps=2,
                   data_format='channels_first' if high_model_name != 'MobileNetV2' else 'channels_last',
                   log_steps=1)
  t1 = threading.Thread(target=high.run, name="high_t1")
  t1.start()
  t1.join()

  t2.join()

# -----------------------------------------------------------------------

if __name__ == "__main__":

  ##########################################################################
  # training(1 thread) speed: avg images per second
  # Experiment 1
  ##########################################################################
  #run_experiment1()

  ##########################################################################
  # Co-run training - vanilla:
  # Experiment 2
  ##########################################################################
  #run_experiment2_1()

  ##########################################################################
  # Co-run training - new:
  # Experiment 2
  ##########################################################################
  # run_experiment2_2()

  ##########################################################################
  # Co-run training(1 thread) and inference(1 thread):
  # Experiment 3
  ##########################################################################
  run_experiment3()

  ##########################################################################
  # Training time:
  # 1. Vanilla run one instance of training
  # 2. Co-run 2 training instances by 2 threads.
  # Experiment 4
  ##########################################################################
  #run_experiment4_vanilla_training()
  #run_experiment4_vanilla_co_run_training()

  ##########################################################################
  # Experiment 5:
  # Get weights size
  ##########################################################################
  #run_experiment5()


  del os.environ['OMP_NUM_THREADS']
  del os.environ['CUDA_VISIBLE_DEVICES']