#!/bin/bash -l
# Frontier equivalent of set_affinity_gpu_polaris.sh.
#
# Polaris: 4 A100s, CUDA_VISIBLE_DEVICES, rank id from $PMI_LOCAL_RANK, and the
# reversed mapping (num_gpus-1-rank) because of Polaris' NIC/GPU topology.
#
# Frontier: 4 MI250X = 8 GCDs, ROCR_VISIBLE_DEVICES, rank id from
# $SLURM_LOCALID. The mapping is NOT identity and NOT reversed — it comes from
# the node's L3/NUMA-to-GCD table (OLCF Frontier User Guide). With srun -c7,
# local rank i lands on L3 region i, whose closest GCD is:
#
#   L3 region (cores)   0-7  8-15 16-23 24-31 32-39 40-47 48-55 56-63
#   closest GCD          4     5     2     3     6     7     0     1
#
# We set this by hand instead of using `--gpu-bind=closest` because that Slurm
# option is known to hang GPU-aware MPI on Frontier — and this code is entirely
# GPU-aware-MPI dependent (exchange_f passes HIPSpace pointers to MPI).
#
# Usage (from the sbatch script):
#   srun ... ./set_affinity_gpu_frontier.sh ./LBM3D_frontier

gpu_map=(4 5 2 3 6 7 0 1)
num_gpus=${#gpu_map[@]}

lrank=${SLURM_LOCALID:-0}
export ROCR_VISIBLE_DEVICES=${gpu_map[$((lrank % num_gpus))]}

exec "$@"
