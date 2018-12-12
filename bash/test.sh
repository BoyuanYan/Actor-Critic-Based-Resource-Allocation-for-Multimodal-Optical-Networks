#!/usr/bin/env sh

python -u ../main.py --mode learning --cnn simplenet --num-steps 2 --step-over one_service \
--net NSFNET.md --wave-num 5 --rou 8 --miu 300 --max-iter 3000 --file-prefix "../resources" \
--k 1 --weight None --workers 1 --steps 3000 \
--img-height 112 --img-width 112 --node-size 0.1 \



