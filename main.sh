GPUS=$1
torchrun --nproc_per_node=$GPUS main.py ${@:2}