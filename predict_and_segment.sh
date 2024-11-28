#!/bin/bash
#SBATCH --time=02:00:00
#SBATCH --mem=40G
#SBATCH --cpus-per-task=6
#SBATCH --gres=gpu:1
#SBATCH --output=exec_logs.out

# Exit if a command fails, i.e., if it outputs a non-zero exit status.
set -e

# Get a newer GCC compiler and CUDA compilers
module load gcc/11.4.0 
module load cuda/12.2.1 

# Activate environment
module load mamba
source activate moviflow

# Run the script with the arguments: path to weights file, dataset
start=$(date +%s)
python move_predict.py --weights checkpoints/kitti.pt --ds 0010
end=$(date +%s)
echo " ---- Elapsed time to process dataset 0010: $(($end-$start)) seconds ----"

exit
