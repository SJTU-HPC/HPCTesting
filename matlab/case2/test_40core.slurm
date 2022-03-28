#!/bin/bash
#SBATCH -J matlab_test
#SBATCH -p cpu
#SBATCH -o %j.out
#SBATCH -e %j.err
#SBATCH -n 1
#SBATCH --exclusive
#SBATCH --ntasks-per-node=40

IMAGE_PATH=/lustre/share/img/matlab_latest.sif

ulimit -s unlimited
ulimit -l unlimited
singularity exec $IMAGE_PATH matlab -r multicore
