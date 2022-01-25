#!/bin/bash
#SBATCH -J run_lammps
#SBATCH -p cpu
#SBATCH -o %j.out
#SBATCH -e %j.err
#SBATCH -N 2
#SBATCH --ntasks-per-node=40

module load oneapi/2021

# test with user intel package
# KMP_BLOCKTIME=0 mpirun -n 80 singularity run $YOUR_IMAGE_PATH  lmp -pk intel 0 omp 1 -sf intel -i in.eam

# test without user intel package
KMP_BLOCKTIME=0 mpirun -n 80 singularity run $YOUR_IMAGE_PATH  lmp -i in.eam