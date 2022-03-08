#!/bin/bash
#SBATCH -J matlab_test
#SBATCH -p small
#SBATCH -o %j.out
#SBATCH -e %j.err
#SBATCH -n 1
#SBATCH --ntasks-per-node=1

IMAGE_PATH=/lustre/share/img/matlab_latest.sif

ulimit -s unlimited
ulimit -l unlimited
cd ~/HPCTesting/matlab
singularity exec $IMAGE_PATH matlab -r test
