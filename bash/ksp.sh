#!/usr/bin/env bash

# python version should be 3.x
python -u ../main.py --mode alg --step-over one_service --file-prefix "../resources"                        \
--net NSFNET.md --wave-num 5 --rou 8 --miu 190 --max-iter 3000 \
--k 1 --weight None --workers 4 --steps 3000

