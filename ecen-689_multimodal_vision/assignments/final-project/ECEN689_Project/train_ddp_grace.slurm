#!/bin/bash

#SBATCH --job-name=ICL_DDP
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=96G
#SBATCH --output=TrajICL_DDP.%j
#SBATCH --gres=gpu:a100:2
#SBATCH --partition=gpu

source /scratch/user/blj/training/.venv/bin/activate

python -m torch.distributed.run --standalone --nproc_per_node=2 train.py \
  --config-name config_ft \
  training.batch_size=32 \
  training.num_workers=6
