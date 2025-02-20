from __future__ import print_function

from miscc.config import cfg
from datasets_Reg import Response_Dataset
from Reg_trainer import Regression_Only_Trainer as trainer

import os
import sys
import time
import yaml
import torch
import random
import pprint
import datetime
import dateutil.tz
import argparse
import numpy as np

import torch
from miscc.utils import mkdir_p
import torchvision.transforms as transforms

dir_path = (os.path.abspath(os.path.join(os.path.realpath(__file__), './.')))
sys.path.append(dir_path)


def creat_dataset(cfg):

    dataset_train = Response_Dataset(cfg.DATA_DIR, cfg.DATA_FILE, cfg.key_X, cfg.key_Y, cfg.train_series)
    dataset_valid = Response_Dataset(cfg.DATA_DIR, cfg.DATA_FILE, cfg.key_X, cfg.key_Y, cfg.valid_series)
    dataset_test = Response_Dataset(cfg.DATA_DIR, cfg.DATA_FILE, cfg.key_X, cfg.key_Y, cfg.test_series)
    assert dataset_train
    assert dataset_valid
    assert dataset_test

    print("Train Input Size: ", dataset_train.input_shape)
    print("Train Output Size: ", dataset_train.output_shape)
    print("Valid Input Size: ", dataset_valid.input_shape)
    print("Valid Output Size: ", dataset_valid.output_shape)
    print("Test Input Size: ", dataset_test.input_shape)
    print("Test Output Size: ", dataset_test.output_shape)

    dataloader_train = torch.utils.data.DataLoader(
        dataset_train, batch_size=cfg.TRAIN.BATCH_SIZE,
        drop_last=False, shuffle=True, num_workers=int(cfg.WORKERS))

    dataloader_valid = torch.utils.data.DataLoader(
        dataset_valid, batch_size=cfg.TRAIN.BATCH_SIZE,
        drop_last=False, shuffle=False, num_workers=int(cfg.WORKERS))

    dataloader_test = torch.utils.data.DataLoader(
        dataset_test, batch_size=cfg.TRAIN.BATCH_SIZE,
        drop_last=False, shuffle=False, num_workers=int(cfg.WORKERS))

    return dataloader_train, dataloader_valid, dataloader_test


if __name__ == "__main__":

    # print('Using config:')
    # pprint.pprint(cfg)

    # 设置随机数种子
    if not cfg.TRAIN.FLAG:
        cfg.manualSeed = 100
    elif cfg.manualSeed is None:
        cfg.manualSeed = random.randint(1, 10000)

    random.seed(cfg.manualSeed)
    np.random.seed(cfg.manualSeed)
    torch.manual_seed(cfg.manualSeed)
    if cfg.CUDA:
        torch.cuda.manual_seed_all(cfg.manualSeed)

    # 计时与存储
    now = datetime.datetime.now(dateutil.tz.tzlocal())
    timestamp = now.strftime('%Y_%m_%d_%H_%M_%S')
    output_dir = './output_train/%s' % timestamp

    # Get data loader
    # 每多一个分支，就会扩大一层维度
    imsize = cfg.TREE.BASE_SIZE * (2 ** (cfg.TREE.BRANCH_NUM - 1))
    # 做一些随机噪声类采样，或许有用或许没用
    image_transform = transforms.Compose([
        transforms.Resize(int(imsize * 76 / 64)),
        transforms.RandomCrop(imsize),
        transforms.RandomHorizontalFlip()])

    # 读取文本数据集（这里改成了序列数据集）
    dataloader = creat_dataset(cfg)

    # 将配置转换为YAML格式的字符串
    yaml_str = yaml.dump(cfg, default_flow_style=False)
    # 将YAML字符串写入文件
    mkdir_p(output_dir)
    with open('%s/config_current_case.yml' % output_dir, 'w') as file:
        file.write(yaml_str)
    del yaml_str

    algo = trainer(output_dir, dataloader)

    start_t = time.time()
    if cfg.TRAIN.FLAG:
        algo.train()
    else:
        algo.valid()

    end_t = time.time()
    print('Total time for training:', end_t - start_t)