#!/bin/bash
#SBATCH --cpus-per-task=28
#SBATCH --nodes=1
#SBATCH --reservation=chung
#SBATCH --gres=gpu:1
#SBATCH --gres=gpu:p40:4
#SBATCH --mem=102400
#SBATCH --job-name=training_logger
#SBATCH --time=04:00:00
#SBATCH --output=slurm_%j.out

#module load python3/intel/3.6.3 cuda/9.0.176 nccl/cuda9.0/2.4.2 

#source ~/pytorch_env/py3.6.3/bin/activate

python ~/project2/distributed-training/Data\ Collection/trainer_pytorch.py -c "35, 36, 37, 38" -d data/ -s training_data/ -w 4
