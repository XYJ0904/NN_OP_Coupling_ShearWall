import pandas as pd
import scipy.io as io
import random
import numpy as np
import torch
import os
import torch.utils.data as data
from miscc.config import cfg
import torchvision.transforms as transforms
from torch.autograd import Variable
import glob
import h5py


class Response_Dataset(data.Dataset):

    def __init__(self, root_path, r_mat_file, key_X, key_Y, series):

        final_path = os.path.join(root_path, r_mat_file)

        if os.path.exists(final_path) and final_path.endswith(".mat"):
            combined_data = io.loadmat(final_path)
            all_input = combined_data[key_X][series]
            all_output = combined_data[key_Y][series]

            r_mat_file_pure = r_mat_file.rstrip("mat").rstrip(".")

        elif os.path.exists(final_path) and final_path.endswith(".h5"):
            with h5py.File(final_path, 'r') as hf:
                # 读取数据集
                all_input = hf[key_X][series][:]
                all_output = hf[key_Y][series][:]

                r_mat_file_pure = r_mat_file.rstrip("h5").rstrip(".")

        elif os.path.isdir(final_path):
            mat_files = glob.glob('%s/*.mat' % final_path)
            r_mat_file_pure = r_mat_file

            # 用于存储合并后的数据
            combined_data = {}

            if mat_files:
                # 从第一个文件中加载所有的key
                initial_data = io.loadmat(mat_files[0])
                for key in initial_data:
                    if not key.startswith('__'):  # 忽略掉内置的属性
                        combined_data[key] = initial_data[key]

                # 遍历剩余的文件并合并数据
                for mat_file in mat_files[1:]:
                    data = io.loadmat(mat_file)
                    for key in combined_data:
                        # 假设所有的key都相同，直接拼接对应的值
                        combined_data[key] = np.concatenate((combined_data[key], data[key]), axis=0)

        else:
            raise FileExistsError

        normalize_file = "%s/Norm_and_Split/record_norm_%s_overall.txt" % (root_path, r_mat_file_pure)
        normalize_data = np.loadtxt(normalize_file, delimiter=",", dtype="str")
        normalize_data = pd.DataFrame(normalize_data, columns=None)
        normalize_data.set_index(0, inplace=True)
        # print(normalize_data)

        min_disp, max_disp = float(normalize_data.loc["Min Value Disp"].values[0]), \
                             float(normalize_data.loc["Max Value Disp"].values[0])
        min_time, max_time = float(normalize_data.loc["Min Value Time"].values[0]), \
                             float(normalize_data.loc["Max Value Time"].values[0])
        min_Y, max_Y = float(normalize_data.loc["Min Value %s" % key_Y].values[0]), \
                       float(normalize_data.loc["Max Value %s" % key_Y].values[0])

        self.len = all_input.shape[0]

        tensor_input = torch.from_numpy(all_input)
        tensor_output = torch.from_numpy(all_output)

        # 注意mask如果要用作屏蔽，就是同时对输入序列/文本某个时间步之后的所有维度进行屏蔽，不会做维度区分
        # 所以time和disp的影响会同时消除；这要求mask的维度就是二维（B * seq_length，没有第三个维度）
        self.input = tensor_input
        self.masks = (tensor_input[:, :, 1] == 0.0)
        self.masks[:, 0] = 0.0

        self.input[:, :, 0] = (tensor_input[:, :, 0] - min_time) / (max_time - min_time)
        self.input[:, :, 1] = (tensor_input[:, :, 1] - min_disp) / (max_disp - min_disp)

        self.input = self.input[:, 0::cfg.input_sampling_ratio, :]
        self.output = (tensor_output - min_Y) / (max_Y - min_Y)
        self.output = self.output.permute(0, 3, 1, 2)

        self.input_shape = list(self.input.size())
        self.output_shape = list(self.output.size())
        self.mask_shape = list(self.masks.size())

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        return self.input[index], self.output[index], self.masks[index]


def prepare_data(output_data):
    imgs = output_data

    real_imgs = []
    imsize = []
    base_size = cfg.TREE.BASE_SIZE

    for i in range(cfg.TREE.BRANCH_NUM):
        imsize.append(base_size)
        base_size = base_size * 2

    # print("imsize", imsize)
    for i in range(cfg.TREE.BRANCH_NUM):
        imgs_resize = resize_imgs(imgs, imsize[i])
        if cfg.CUDA:
            real_imgs.append(Variable(imgs_resize).cuda())
        else:
            real_imgs.append(Variable(imgs_resize))

    return real_imgs


def resize_imgs(img, imsize):

    re_img = transforms.Resize((imsize, imsize))(img)

    return re_img


if __name__ == "__main__":
    root_path = r"D:\XYJ-科研工作\20230715 单向耦合剪力墙模拟\正式神经网络训练\Dataset"
    file_name = "Dataset_1_20240316.mat"
    key_X = "input"
    key_Y = "Strain"
    series = np.arange(0, 100, 1).tolist()

    Response_Dataset(root_path, file_name, key_X, key_Y, series)
