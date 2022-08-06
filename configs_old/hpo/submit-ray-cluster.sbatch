#!/bin/bash

#SBATCH --job-name=h1c2
#SBATCH --time=06-02:00:00
#SBATCH --nodes=1
#SBATCH -e err.%j
#SBATCH --partition g_vsheno
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=8
#SBATCH --cpus-per-task=32

### This script works for any number of nodes, Ray will find and manage all resources
### Give all resources to a single Ray task, ray can manage the resources internally

# Load modules or your own conda environment here
# e.g. conda activate ocp-models
# module load intel/17.0.3
module purge

module load gcc-9.2.0/9.2.0
module load gpu/cuda/10.2

ulimit -s unlimited
ulimit -n 4096

eval "$(conda shell.bash hook)"
# source activate ocp-models
source activate ocp-gpu

ray stop

################# DON NOT CHANGE THINGS HERE UNLESS YOU KNOW WHAT YOU ARE DOING ###############
# This script is a modification to the implementation suggest by gregSchwartz18 here:
# https://github.com/ray-project/ray/issues/826#issuecomment-522116599
redis_password=$(uuidgen)
export redis_password

# nvidia-smi --query-gpu=timestamp,name,pci.bus_id,driver_version,pstate,pcie.link.gen.max,pcie.link.gen.current,temperature.gpu,utilization.gpu,utilization.memory,memory.total,memory.free,memory.used --format=csv -l 15 -f nvidialog.txt &
# bckpid=$!

nodes=$(scontrol show hostnames $SLURM_JOB_NODELIST) # Getting the node names
nodes_array=( $nodes )

node_1=${nodes_array[0]}
ip=$(srun --nodes=1 --ntasks=1 -w $node_1 hostname --ip-address) # making redis-address
port=6379
ip_head=$ip:$port
export ip_head
echo "IP Head: $ip_head"

echo "STARTING HEAD at $node_1"
srun --nodes=1 --ntasks=1 -w $node_1 start-head.sh $ip $redis_password &
sleep 45

worker_num=$(($SLURM_JOB_NUM_NODES - 1)) #number of nodes other than the head node

if [ $worker_num -gt 0 ]; then
  for ((  i=1; i<=$worker_num; i++ ))
    do
      node_i=${nodes_array[$i]}
      echo "STARTING WORKER $i at $node_i"
      srun --nodes=1 --ntasks=1 -w $node_i start-worker.sh $ip_head $redis_password &
      sleep 5
    done
fi
##############################################################################################

#### call your code below
# e.g. python path_to/run_tune.py --mode train --config-yml path_to/configs/s2ef/200k/forcenet/fn_forceonly.yml --run_dir path_to_run_dir

# bash ~/script-repo/memoryprofile.sh "$processname" &
# bckpid1=$!

python -u /mnt/io2/scratch_vshenoy1/chrispr/catalysis/ocp/configs/baseline_train/dimenet_plus_plus/run_tune.py --mode=train --config-yml=/mnt/io2/scratch_vshenoy1/chrispr/catalysis/ocp/configs/baseline_train/dimenet_plus_plus/dpp.yml --run_dir=/mnt/io2/scratch_vshenoy1/chrispr/catalysis/experiments/
python -u /mnt/io2/scratch_vshenoy1/chrispr/catalysis/ocp/configs/baseline_train/cgcnn/run_tune.py --mode=train --config-yml=/mnt/io2/scratch_vshenoy1/chrispr/catalysis/ocp/configs/baseline_train/cgcnn/cgcnn.yml --run_dir=/mnt/io2/scratch_vshenoy1/chrispr/catalysis/experiments/
python -u /mnt/io2/scratch_vshenoy1/chrispr/catalysis/ocp/configs/baseline_train/schnet/run_tune.py --mode=train --config-yml=/mnt/io2/scratch_vshenoy1/chrispr/catalysis/ocp/configs/baseline_train/schnet/schnet.yml --run_dir=/mnt/io2/scratch_vshenoy1/chrispr/catalysis/experiments/

# kill "$bckpid"
# kill "$bckpid1"
ray stop

exit
