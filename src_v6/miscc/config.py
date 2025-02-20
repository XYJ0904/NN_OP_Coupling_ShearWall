from __future__ import division
from __future__ import print_function

import os.path as osp
import numpy as np
from easydict import EasyDict as edict

from matplotlib.colors import LinearSegmentedColormap

__C = edict()
cfg = __C

__C.manualSeed = 1

# Dataset name: flowers, birds
__C.DATA_DIR = r'/root/autodl-tmp/xyj/Dataset'
__C.DATA_FILE = 'Dataset_5_20240327_All.h5'

if "." in __C.DATA_FILE:
    __C.DATA_FILE_PURE = __C.DATA_FILE.rstrip("mat").rstrip("h5").rstrip(".")
else:
    __C.DATA_FILE_PURE = __C.DATA_FILE

__C.key_X = "input"
__C.key_Y = "Strain"
__C.train_series = np.loadtxt("%s/Norm_and_Split/train_series_%s.txt" %
                              (__C.DATA_DIR, __C.DATA_FILE_PURE), dtype="int").tolist()
__C.valid_series = np.loadtxt("%s/Norm_and_Split/valid_series_%s.txt" %
                              (__C.DATA_DIR, __C.DATA_FILE_PURE), dtype="int").tolist()
__C.test_series = np.loadtxt("%s/Norm_and_Split/test_series_%s.txt" %
                             (__C.DATA_DIR, __C.DATA_FILE_PURE), dtype="int").tolist()

__C.sequence_length = 1000
__C.input_dim = 2
__C.output_dim = 6
cfg.input_sampling_ratio = 1

__C.GPU_ID = 0
__C.CUDA = True
__C.WORKERS = 10

__C.RNN_TYPE = 'GRU'   # 'GRU'
__C.B_VALIDATION = False

__C.TREE = edict()
__C.TREE.BRANCH_NUM = 4
__C.TREE.BASE_SIZE = 16
__C.TREE.initial_enlarge_layer = 4

# Training options
__C.TRAIN = edict()
__C.TRAIN.BATCH_SIZE = 128
__C.TRAIN.MAX_EPOCH = 10000
# __C.TRAIN.SNAPSHOT_INTERVAL = 10

# __C.TRAIN.DISCRIMINATOR_LR = 2e-3
__C.TRAIN.GENERATOR_LR = 5e-4
__C.TRAIN.ENCODER_LR = __C.TRAIN.GENERATOR_LR
__C.TRAIN.warmup_steps = 20
__C.TRAIN.LR_decay_ratio = 0.999

__C.TRAIN.RNN_GRAD_CLIP = 0.25
__C.TRAIN.FLAG = True
__C.TRAIN.NET_E = ''
__C.TRAIN.NET_G = ''
# __C.TRAIN.B_NET_D = True

# __C.TRAIN.SMOOTH = edict()
# __C.TRAIN.SMOOTH.GAMMA1 = 5.0
# __C.TRAIN.SMOOTH.GAMMA3 = 10.0
# __C.TRAIN.SMOOTH.GAMMA2 = 5.0
# __C.TRAIN.SMOOTH.LAMBDA = 1.0

__C.TRAIN.REG_LOSS_WEIGHT = 1.0
# __C.TRAIN.GEN_LOSS_WEIGHT = 0.0
__C.TRAIN.KL_LOSS_WEIGHT = 0.01
__C.TRAIN.REG_LOSS_TYPE = "MSE"


__C.TEXT = edict()
__C.TEXT.EMBEDDING_DIM = 512

# Modal options
__C.GAN = edict()
__C.GAN.DF_DIM = int(__C.TEXT.EMBEDDING_DIM / 8)
__C.GAN.GF_DIM = int(__C.TEXT.EMBEDDING_DIM / 6)
__C.GAN.Z_DIM = int(__C.TEXT.EMBEDDING_DIM / 2)
__C.GAN.CONDITION_DIM = int(__C.TEXT.EMBEDDING_DIM / 2)
__C.GAN.R_NUM = 2
__C.GAN.B_ATTENTION = True
__C.GAN.B_DCGAN = False


__C.PLOT = edict()

colors = [(1, 1, 1), (1, 0, 0)]  # 白色到红色，但顺序是反过来的；差值是从上往下的
n_bins = 100
cmap_name = 'white_red'
__C.PLOT.COLORMAP = LinearSegmentedColormap.from_list(cmap_name, colors, N=100)


# def _merge_a_into_b(a, b):
#     """Merge config dictionary a into config dictionary b, clobbering the
#     options in b whenever they are also specified in a.
#     """
#     if type(a) is not edict:
#         return
#
#     for k, v in a.items():
#         # a must specify keys that are in b
#         if not k in b:
#             raise KeyError('{} is not a valid config key'.format(k))
#
#         # the types must match, too
#         old_type = type(b[k])
#         if old_type is not type(v):
#             if isinstance(b[k], np.ndarray):
#                 v = np.array(v, dtype=b[k].dtype)
#             else:
#                 raise ValueError(('Type mismatch ({} vs. {}) '
#                                   'for config key: {}').format(type(b[k]),
#                                                                type(v), k))
#
#         # recursively merge dicts
#         if type(v) is edict:
#             try:
#                 _merge_a_into_b(a[k], b[k])
#             except:
#                 print('Error under config key: {}'.format(k))
#                 raise
#         else:
#             b[k] = v
#
#
# def cfg_from_file(filename):
#     """Load a config file and merge it into the default options."""
#     import yaml
#     with open(filename, 'r') as f:
#         yaml_cfg = edict(yaml.load(f, Loader=yaml.SafeLoader))
#
#     _merge_a_into_b(yaml_cfg, __C)
#
#
# # cfg_pre_load_file = "./cfg/DAMSM/SW_Pretrain.yml"
# cfg_pre_load_file = "./cfg/SW_Train.yml"
# if cfg_pre_load_file is not None:
#     cfg_from_file(cfg_pre_load_file)
#
# __C.PRELOAD_CFG = cfg_pre_load_file