#!/bin/bash
export OMPI_CXX=/grand/IMEXLBM/czhao/Kokkos/kokkos/bin/nvcc_wrapper
export OMP_PROC_BIND=spread
export OMP_PLACES=threads
NODES='cat $COBALT_NODEFILE|wc-|'
PROCS=$((NODES*8))
make -j KOKKOS_DEVICES=Cuda
mpirun -f $COBALT_NODEFILE -n $PROCS LBM.exe


