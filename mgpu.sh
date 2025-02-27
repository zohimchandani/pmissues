#!/bin/bash
#SBATCH -N 1
#SBATCH --gpus-per-task=1
#SBATCH --ntasks-per-node=4
#SBATCH --gpu-bind=none
#SBATCH -t 00:10:00
#SBATCH -q debug
#SBATCH -A m4955
#SBATCH -C gpu
#SBATCH --image=docker:nvcr.io/nvidia/nightly/cuda-quantum:cu11-latest
#SBATCH --module=cuda-mpich

export CUDAQ_MPI_COMM_LIB=${HOME}/distributed_interfaces/libcudaq_distributed_interface_mpi.so

srun --mpi=pmix -N 1 -n 4 shifter bash -l launch.sh ansys.py
