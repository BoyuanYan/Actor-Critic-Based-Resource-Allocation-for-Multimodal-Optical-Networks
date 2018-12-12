#!/bin/env python
from matplotlib import use
use('Agg')
import argparse
from matplotlib import pyplot as plt
import os
import subprocess as sp

parser = argparse.ArgumentParser()

parser.add_argument('--file', type=str)

args = parser.parse_args()


def parseUpdates(filename):
    out = sp.getoutput("cat {} | grep remain".format(filename))
    ids = []
    bps = []
    lines = out.split('\n')
    for line in lines:
        segs = line.split(',')
        index = int(segs[0].split(' ')[1])
        bp = float(segs[9].split('=')[1])
        ids.append(index)
        bps.append(bp)
    return ids, bps

ids, bps = parseUpdates(args.file)

plt.plot(ids, bps)
plt.savefig(args.file+".png")