# 1. Starting MPS control daemon
# single GPU
export CUDA_VISIBLE_DEVICES=0

# Select a location that’s accessible to the given $UID
export CUDA_MPS_PIPE_DIRECTORY=/tmp/nvidia-mps

# Select a location that’s accessible to the given $UID
export CUDA_MPS_LOG_DIRECTORY=/tmp/nvidia-log

# Start the daemon.
nvidia-cuda-mps-control -d