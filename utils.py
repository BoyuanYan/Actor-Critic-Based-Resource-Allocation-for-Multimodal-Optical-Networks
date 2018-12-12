import torch
import torch.nn as nn
import subprocess as sp
import matplotlib.pyplot as plt
import os
from PIL import Image
import numpy as np
from args import args

file_prefix = args.file_prefix


def parse_log(file):
    """
    解析log文件，画出阻塞率的变化曲线
    :param file:
    :return:
    """
    prefix = 'bash'
    log_file = os.path.join(prefix, file)
    out = sp.getoutput("cat {}| grep remain".format(log_file))
    out = out.split('\n')
    y = []
    for i in out:
        tmp = i.split(' ')[26]
        tmp = tmp.split('=')[1]
        y.append(float(tmp))
    plt.plot(y)


def save_on_disk(stacked_data: np.ndarray, out_dir: str = 'test_visiable'):
    """

    :param stacked_data:归一化以后的叠加灰度图，格式仍然是CHW
    :param out_dir: 图像输出的目录
    :return:
    """
    print("shape of image is {}".format(stacked_data.shape))
    for channel in range(stacked_data.shape[0]):
        data = stacked_data[channel, :, :].squeeze()
        data = data * 255
        data = data.astype(np.uint8)
        img = Image.fromarray(data, mode='L')
        saved = os.path.join(out_dir, str(channel)+".png")
        img.save(saved)