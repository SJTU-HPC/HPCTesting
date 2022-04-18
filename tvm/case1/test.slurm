#!/bin/bash

#SBATCH -N 1
#SBATCH -n 1 
#SBATCH -p dgx2
#SBATCH --gres gpu:1


module purge
IMAGE_PATH=/lustre/home/acct-hpc/hpchxj/tvm2.sif

singularity run --nv --env TVM_LOG_DEBUG=1 $IMAGE_PATH python test.py
