import os
os.environ['TF_CPP_MIN_VLOG_LEVEL']='0'

os.environ['TF_SET_REUSE_INPUTS_FLAG'] = '1'
os.environ['TF_REUSE_INPUT_OP_NAME_MASTER_X'] = 'X00'
os.environ['TF_REUSE_INPUT_OP_NAME_MASTER_y'] = 'y00'
os.environ['TF_REUSE_INPUT_OPS_NAME_SUB_X'] = 'X01'
os.environ['TF_REUSE_INPUT_OPS_NAME_SUB_y'] = 'y01'
# N.B. splitter is "," NO SPACE BETWEEN!!!
# subsidiary_input_names

import threading
import time

from keras_placeholder_models import Model

"""
Description
1. Create Graphs
2. define func
3. launch threads, thread_N : func_N
"""

def TEST_WithoutThreads():
  # Description: Test without threads
  pass
  # to test time

def TEST_2_threads():
  user_00 = Model()
  user_01 = Model()

  # passing args
  # https://stackoverflow.com/questions/30913201/pass-keyword-arguments-to-target-function-in-python-threading-thread
  #user_00.BuildGraph(graph_name='graph_00', X_name='X00', y_name='y00')
  #user_01.BuildGraph(graph_name='graph_01', X_name='X01', y_name='y01')

  t0 = threading.Thread(name='t0', target=user_00.BuildAndRunGraph,
                        args=('graph_00', 'X00', 'y00')
                        )

  t1 = threading.Thread(name='t1', target=user_01.BuildAndRunGraph,
                        args=('graph_01', 'X01', 'y01')
                        )

  #t_s = time.time()
  t0.start()
  t1.start()

  t0.join()
  t1.join()
  #t_e = time.time()
  #t_total = t_e - t_s
  # It times extra 1.62 s to construct, so don't measure this time!
  #print('>>> Total time: ', t_total)  # 1.62 s to construct

def TEST_3_threads():
  pass
  #user_00 = MNIST_CNN()
  #user_01 = MNIST_CNN()
  #user_02 = MNIST_CNN()

  #user_00.BuildGraph(graph_name='graph_00', X_name='X00', y_name='y00')
  #user_01.BuildGraph(graph_name='graph_01', X_name='X01', y_name='y01')
  #user_02.BuildGraph(graph_name='graph_02', X_name='X02', y_name='y02')

  #t0 = threading.Thread(name='t0', target=user_00.RunGraph)
  #t1 = threading.Thread(name='t1', target=user_01.RunGraph)
  #t2 = threading.Thread(name='t2', target=user_02.RunGraph)

  ##t_s = time.time()
  #t0.start()
  #t1.start()
  #t2.start()

  #t0.join()
  #t1.join()
  #t2.join()
  ##t_e = time.time()
  ##t_total = t_e - t_s
  ## It times extra 1.62 s to construct, so don't measure this time!
  ##print('>>> Total time: ', t_total)

#######################################
# N.B.
# 1. don't forget to update num_sessions, number of threads == num_sessions !!!
# 2. os.environ['TF_REUSE_INPUT_OPS_NAME_SUB_X'] = 'X01,X02'
#######################################
#TEST_WithoutThreads()
TEST_2_threads()
#TEST_3_threads()

# clear before set
#del os.environ['TF_SET_REUSE_INPUTS_FLAG']
#del os.environ['TF_REUSE_INPUT_OP_NAME_MASTER_X']
#del os.environ['TF_REUSE_INPUT_OP_NAME_MASTER_y']
#del os.environ['TF_REUSE_INPUT_OPS_NAME_SUB_X']
#del os.environ['TF_REUSE_INPUT_OPS_NAME_SUB_y']