#!/usr/bin/env sh

python -u ../../main.py --mode learning --cnn simplestnet --num-steps 8 --step-over one_service \
	--net 6node.md --wave-num 5 --rou 8 --miu 120 --max-iter 300 --save-interval 500 --file-prefix "../../resources" \
	--k 1 --weight None --workers 16 --steps 10e6 --base-lr 7e-4 \
	--img-height 112 --img-width 112 \
	--reward 1 --punish -1

