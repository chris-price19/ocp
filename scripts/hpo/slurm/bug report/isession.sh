#!/bin/bash

#SBATCH --job-name=h1c2
#SBATCH --time=0-02:00:00
#SBATCH --nodes=1
#SBATCH -e err.%j
##SBATCH -C "[ib1|ib2|ib3|ib4]"
##SBATCH --partition g_vsheno
#SBATCH --partition debug
#SBATCH --ntasks-per-node=1
##SBATCH --gpus-per-task=8
#SBATCH --cpus-per-task=32
##SBATCH --mem-per-cpu=DefMemPerCPU

### This script works for any number of nodes, Ray will find and manage all resources
### Give all resources to a single Ray task, ray can manage the resources internally

# Load modules or your own conda environment here
# e.g. conda activate ocp-models
# module load intel/17.0.3
module purge

module load gcc-9.2.0/9.2.0
# module load gpu/cuda/10.2

ulimit -s unlimited
ulimit -n 4096

source activate ocp-models

ray stop

################# DON NOT CHANGE THINGS HERE UNLESS YOU KNOW WHAT YOU ARE DOING ###############
# This script is a modification to the implementation suggest by gregSchwartz18 here:
# https://github.com/ray-project/ray/issues/826#issuecomment-522116599
redis_password=$(uuidgen)
export redis_password

nodes=$(scontrol show hostnames $SLURM_JOB_NODELIST) # Getting the node names
nodes_array=( $nodes )

node_1=${nodes_array[0]}
ip=$(srun --nodes=1 --ntasks=1 -w $node_1 hostname --ip-address) # making redis-address
port=16379
ip_head=$ip:$port
export ip_head
echo "IP Head: $ip_head"

echo "STARTING HEAD at $node_1"
srun --nodes=1 --ntasks=1 -w $node_1 start-head.sh $ip $redis_password $port &
sleep 45

python -u bug_tune.py
