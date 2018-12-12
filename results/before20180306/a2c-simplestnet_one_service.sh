#!/usr/bin/env sh
# 测试5个波长情况下的表现

python -u ../main.py --mode learning --cnn simplestnet --num-steps 16 --step-over one_service \
--net 6node.md --wave-num 5 --rou 8 --miu 120 --max-iter 1000 \
--k 1 --weight None --workers 16 --steps 10e6 \
--img-height 224 --img-width 224


