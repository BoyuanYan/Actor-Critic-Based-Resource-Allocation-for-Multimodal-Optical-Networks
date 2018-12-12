#!/usr/bin/env sh

#MV2_USE_CUDA=1 MV2_ENABLE_AFFINITY=0 MV2_SMP_USE_CMA=0
#SBATCH --partition=Liveness
#SBATCH -w BJ-IDC1-10-10-16-70
#SBATCH --job-name=a2c
#SBATCH -o log-a2c-expandsimplenet-%j

srun python -u ../../main.py --mode learning --cnn expandsimplenet --num-steps 8 --step-over one_service \
		--net 6node.md --wave-num 40 --rou 8 --miu 1400 --max-iter 5000 --save-interval 500 --file-prefix "../../resources" \
			--k 1 --weight None --workers 16 --steps 10e8 --base-lr 7e-4 \
				--img-height 112 --img-width 112 \
					--reward 1 --punish -1 --expand-factor 3
