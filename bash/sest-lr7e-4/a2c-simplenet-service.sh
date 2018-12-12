#!/usr/bin/env sh

#MV2_USE_CUDA=1 MV2_ENABLE_AFFINITY=0 MV2_SMP_USE_CMA=0
#SBATCH --partition=Liveness
#SBATCH -n1
#SBATCH --ntasks-per-node=1
#SBATCH --job-name=a2c
#SBATCH -o log-a2c-simplestnet-%j

srun python -u ../../main.py --mode learning --cnn simplestnet --num-steps 8 --step-over one_service \
	--net 6node.md --wave-num 5 --rou 8 --miu 120 --max-iter 300 --save-interval 500 --file-prefix "../../resources" \
	--k 1 --weight None --workers 16 --steps 10e6 --base-lr 7e-4 \
	--img-height 112 --img-width 112 \
	--reward 1 --punish -1

