#!/bin/bash

#SBATCH --job-name=h1c2
#SBATCH --time=06-02:00:00
#SBATCH --nodes=1
#SBATCH -e err.%j
#SBATCH --partition g_vsheno
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=4

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

python -u /mnt/io2/scratch_vshenoy1/chrispr/catalysis/ocp/main.py --mode=train --config-yml=/mnt/io2/scratch_vshenoy1/chrispr/catalysis/ocp/configs/baseline_train/dimenet_plus_plus/dpp_best.yml --run_dir=/mnt/io2/scratch_vshenoy1/chrispr/catalysis/train/


exit
