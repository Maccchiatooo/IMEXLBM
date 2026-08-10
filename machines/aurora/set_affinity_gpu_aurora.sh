#!/bin/bash -l
# Aurora equivalent of set_affinity_gpu_polaris.sh.
#
# Polaris:  4 A100s,  CUDA_VISIBLE_DEVICES, rank from $PMI_LOCAL_RANK
# Frontier: 8 GCDs,   ROCR_VISIBLE_DEVICES, rank from $SLURM_LOCALID
# Aurora:   6 PVCs x 2 tiles = 12 tiles, ZE_AFFINITY_MASK, rank from
#           $PALS_LOCAL_RANKID (Cray PALS launcher, not PMI/Slurm).
#
# Mapping is compact: consecutive local ranks fill a GPU's two tiles before
# moving to the next GPU.
#
#   local rank   0   1   2   3   4   5   6   7   8   9  10  11
#   ZE mask     0.0 0.1 1.0 1.1 2.0 2.1 3.0 3.1 4.0 4.1 5.0 5.1
#   socket       <-------- 0 -------->  <-------- 1 -------->
#
# That pairs with the --cpu-bind list in submit_aurora.sh, which puts local
# ranks 0-5 on socket 0 (cores 1-48) and 6-11 on socket 1 (cores 53-100).
# Each socket hosts 3 of the 6 PVCs, so rank i and GPU i/2 land on the same
# socket and MPI/GPU traffic stays off the inter-socket link.
#
# The "<gpu>.<tile>" mask syntax requires the COMPOSITE device hierarchy;
# env_aurora.sh pins ZE_FLAT_DEVICE_HIERARCHY=COMPOSITE for exactly this reason.
#
# This is functionally the same as ALCF's centrally installed
# gpu_tile_compact.sh (`which gpu_tile_compact.sh`) — swap it in if you prefer
# the supported script; it is kept local here to mirror the Polaris/Frontier
# layout and to keep the tree self-contained.
#
# Usage (from the PBS script):
#   mpiexec ... ./set_affinity_gpu_aurora.sh ./LBM2D_aurora

num_tiles=2

lrank=${PALS_LOCAL_RANKID:-0}
gpu_id=$((lrank / num_tiles))
tile_id=$((lrank % num_tiles))

export ZE_AFFINITY_MASK=${gpu_id}.${tile_id}

exec "$@"
