import os
os.environ['TF_CPP_MIN_VLOG_LEVEL']='0'

os.environ['TF_SET_REUSE_INPUTS_FLAG'] = '1'
os.environ['TF_REUSE_INPUT_OP_NAME_MASTER_X'] = 'IteratorGetNext/_5'
os.environ['TF_REUSE_INPUT_OP_NAME_MASTER_y'] = 'IteratorGetNext/_3'
# reuse names id: https://gist.github.com/shizukanaskytree/801cc88a9f19476c9f12630942cdb3e3

# for 1 Master and 1 subsidiaries(X01,y01)
os.environ['TF_REUSE_INPUT_OPS_NAME_SUB_X'] = 'X01'
os.environ['TF_REUSE_INPUT_OPS_NAME_SUB_y'] = 'y01'

## for 1 Master and 2 subsidiaries(X01,y01;X02,y02)
#os.environ['TF_REUSE_INPUT_OPS_NAME_SUB_X'] = 'X01,X02'
#os.environ['TF_REUSE_INPUT_OPS_NAME_SUB_y'] = 'y01,y02'
# N.B. splitter is "," NO SPACE BETWEEN!!!
# subsidiary_input_names

import threading
import time

from master_keras_dataset_models import MasterModel
from subsidiary_keras_placeholder_models import SubModel
from subsidiary_keras_placeholder_models_2nd import SubModel as SubModel_2nd

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
  user_00 = MasterModel()
  user_01 = SubModel()

  # passing args
  # https://stackoverflow.com/questions/30913201/pass-keyword-arguments-to-target-function-in-python-threading-thread
  #user_00.BuildGraph(graph_name='graph_00', X_name='X00', y_name='y00')
  #user_01.BuildGraph(graph_name='graph_01', X_name='X01', y_name='y01')

  t0 = threading.Thread(name='t0', target=user_00.BuildAndRunGraph,
                        args=('graph_00', 'X00', 'y00')
                        )

  # 1st subsidiary graph
  t1 = threading.Thread(name='t1', target=user_01.BuildAndRunGraph,
                        args=('graph_01', 'X01', 'y01')
                        )
  # wxf: check corresponding graph name and its sess.run token turn id
  #graph_session_id = {
  #  'graph_00': 0,  # master
  #  'graph_01': 1,  # sub 01
  #  'graph_02': 2,  # sub 02
  #  'graph_03': 3,  # sub 03
  #}

  #t_s = time.time()
  t0.start()
  time.sleep(30)
  t1.start()

  t0.join()
  t1.join()
  #t_e = time.time()
  #t_total = t_e - t_s
  # It times extra 1.62 s to construct, so don't measure this time!
  #print('>>> Total time: ', t_total)  # 1.62 s to construct

def TEST_3_threads():
  user_00 = MasterModel()
  user_01 = SubModel()
  user_02 = SubModel_2nd()

  # passing args
  # https://stackoverflow.com/questions/30913201/pass-keyword-arguments-to-target-function-in-python-threading-thread
  # user_00.BuildGraph(graph_name='graph_00', X_name='X00', y_name='y00')
  # user_01.BuildGraph(graph_name='graph_01', X_name='X01', y_name='y01')

  t0 = threading.Thread(name='t0', target=user_00.BuildAndRunGraph,
                        args=('graph_00', 'X00', 'y00')
                        )

  # 1st subsidiary graph
  t1 = threading.Thread(name='t1', target=user_01.BuildAndRunGraph,
                        args=('graph_01', 'X01', 'y01')
                        )

  # 2nd subsidiary graph
  t2 = threading.Thread(name='t2', target=user_02.BuildAndRunGraph,
                        args=('graph_02', 'X02', 'y02')
                        )

  #######################################
  # 01:
  # wxf: check corresponding graph name and its sess.run token turn id
  # graph_session_id = {
  #  'graph_00': 0,  # master
  #  'graph_01': 1,  # sub 01
  #  'graph_02': 2,  # sub 02
  #  'graph_03': 3,  # sub 03
  # }
  #
  # 02:
  # number of threads == num_sessions !!!
  #num_sessions = 3
  #######################################

  # t_s = time.time()
  t0.start()
  t1.start()
  time.sleep(30)
  t2.start()

  t0.join()
  t1.join()
  t2.join()


#######################################
# N.B.
# 1. don't forget to update num_sessions, number of threads == num_sessions !!!
# 2. os.environ['TF_REUSE_INPUT_OPS_NAME_SUB_X'] = 'X01,X02'
#######################################
#TEST_WithoutThreads()
TEST_2_threads()
#TEST_3_threads()

# clear before set
del os.environ['TF_SET_REUSE_INPUTS_FLAG']
del os.environ['TF_REUSE_INPUT_OP_NAME_MASTER_X']
del os.environ['TF_REUSE_INPUT_OP_NAME_MASTER_y']
del os.environ['TF_REUSE_INPUT_OPS_NAME_SUB_X']
del os.environ['TF_REUSE_INPUT_OPS_NAME_SUB_y']