#!/bin/bash

#SBATCH -p dgx2
#SBATCH -n 1
#SBATCH --ntasks-per-node 1
#SBATCH --cpus-per-task 6 
#SBATCH --gres gpu:1
#SBATCH -N 1



singularity  run --nv   /lustre/share/singularity/modules/tensorflow/2.2.0.sif python tf_test.py
