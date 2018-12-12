#!/usr/bin/env sh


#SBATCH --partition=Liveness
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH -n1
#SBATCH --job-name=a2c
#SBATCH -o log-cuda-a2c-simplestnet-%j
#SBATCH -w BJ-IDC1-10-10-16-82

srun python -u ../main.py --mode learning --cnn simplestnet --num-steps 16 --cuda True \
--net 6node.md --wave-num 10 --rou 8 --miu 300 --max-iter 300 \
--k 1 --weight None --workers 16 --steps 10e6 \
--img-height 224 --img-width 224


