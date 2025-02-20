from __future__ import print_function
from six.moves import range

import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
from torch.optim.lr_scheduler import LambdaLR, ExponentialLR

from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
import matplotlib

from miscc.config import cfg
from miscc.utils import mkdir_p
# from miscc.utils import build_super_images, build_super_images2
from miscc.utils import weights_init, load_params, copy_G_params
from model import G_NET
from datasets_Reg import prepare_data
from model import RNN_ENCODER

from miscc.losses import words_loss
from miscc.losses import discriminator_loss, generator_loss, regression_loss

from scipy.io import savemat, loadmat

import os
import time
import numpy as np
import sys
import skimage

# import scienceplots

# matplotlib.use("qtagg")
# plt.style.use(["science", "ieee"])

# TODO: 是否考虑noise也会对结果产生影响，或许后面可以不考虑
# TODO: 暂且不添加mask，后续再进一步分析其影响 (已解决）
# TODO：二阶段网络，第一阶段执行序列到图像，第二阶段把图像融合进去再拟合一遍残差看看效果如何
# TODO：物理引导，参考维基百科的Linear Elasticity
# TODO：条件增强网络CA_NET是否可以去掉


class WarmupThenExpDecayLR(torch.optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, last_epoch=-1):
        self.warmup_steps = cfg.TRAIN.warmup_steps
        self.gamma = cfg.TRAIN.LR_decay_ratio
        self.step_after_warmup = 0
        super(WarmupThenExpDecayLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch < self.warmup_steps:
            # Warmup阶段：线性增加
            return [base_lr * float(self.last_epoch + 1) / self.warmup_steps for base_lr in self.base_lrs]
        else:
            # Warmup之后：指数衰减
            self.step_after_warmup += 1
            return [base_lr * (self.gamma ** self.step_after_warmup) for base_lr in self.base_lrs]


# ################# Sequence to image task############################ #

class Regression_Only_Trainer(object):
    def __init__(self, output_dir, data_loader):
        if cfg.TRAIN.FLAG:

            self.split = "Train"

            # 保存模型和结果到指定文件夹
            self.model_dir = os.path.join(output_dir, 'Model')
            self.image_dir = os.path.join(output_dir, 'Image')
            self.log_dir = os.path.join(output_dir, 'Log')
            self.data_dir = os.path.join(output_dir, 'Data')
            mkdir_p(self.model_dir)
            mkdir_p(self.image_dir)
            mkdir_p(self.log_dir)
            mkdir_p(self.data_dir)

        else:
            self.split = "Valid"

        torch.cuda.set_device(cfg.GPU_ID)
        self.device = torch.device('cuda:%s' % cfg.GPU_ID)
        cudnn.benchmark = True

        # 指定关键参数，包括batch size和最大epoch，以及保存模型的间隔
        self.batch_size = cfg.TRAIN.BATCH_SIZE
        self.max_epoch = cfg.TRAIN.MAX_EPOCH
        # self.snapshot_interval = cfg.TRAIN.SNAPSHOT_INTERVAL

        # 编码和解码信息，以及数据集信息
        self.n_words = None

        self.data_loader_train, self.data_loader_valid, self.data_loader_test = data_loader
        self.num_batches_train, self.num_batches_valid, self.num_batches_test = \
            len(self.data_loader_train), len(self.data_loader_valid), len(self.data_loader_test)


    def build_models(self):
        # ###################encoders######################################## #

        # 文本编码器直接采用LSTM或者GRU，并加载预训练参数
        text_encoder = RNN_ENCODER(ninput=int(cfg.TEXT.EMBEDDING_DIM / 2), nhidden=cfg.TEXT.EMBEDDING_DIM)

        # #######################generator and discriminators############## #
        # 这里只需要生成器网络，不需要判别器网络
        netG = G_NET()
        # 执行特定层次的初始化
        netG.apply(weights_init)

        print("\n", "*" * 100, "\n")

        # 如果生成器有预训练，则加载预训练的生成器参数，否则从头开始训练

        if cfg.TRAIN.NET_G != '':
            state_dict = \
                torch.load(cfg.TRAIN.NET_G, map_location=lambda storage, loc: storage)
            netG.load_state_dict(state_dict)
            print('Load G from: ', cfg.TRAIN.NET_G)

        # ########################################################### #
        if cfg.CUDA:
            text_encoder = text_encoder.to(self.device)
            netG = netG.to(self.device)

        return [text_encoder, netG]


    def set_requires_grad_value(self, models_list, brequires):
        for i in range(len(models_list)):
            for p in models_list[i].parameters():
                p.requires_grad = brequires


    def train(self):
        # 创建基础模型，读取预训练参数
        text_encoder, netG = self.build_models()
        avg_param_G = copy_G_params(netG)

        # 生成基础标签以及优化器
        optimizerG = optim.Adam(netG.parameters(), lr=cfg.TRAIN.GENERATOR_LR, betas=(0.5, 0.999))
        optimizerE = optim.Adam(text_encoder.parameters(), lr=cfg.TRAIN.ENCODER_LR, betas=(0.5, 0.999))

        schedulerG = WarmupThenExpDecayLR(optimizerG)
        schedulerE = WarmupThenExpDecayLR(optimizerE)

        scaler = GradScaler()

        batch_size = self.batch_size

        nz = cfg.GAN.Z_DIM
        noise = torch.FloatTensor(batch_size, nz)

        if cfg.CUDA:
            noise = noise.to(self.device)

        # fixed_noise = Variable(torch.FloatTensor(batch_size, nz).normal_(0, 1))
        # if cfg.CUDA:
        #     noise, fixed_noise = noise.to(self.device), fixed_noise.to(self.device)

        f = open(f"{self.log_dir}/Log_Train.csv", "w")
        f.write("epoch,max_epoch,G Loss,Train Time\n")
        f.close()

        f = open(f"{self.log_dir}/Log_Valid.csv", "w")
        f.write("epoch,max_epoch,G Loss,Minimum G Loss,Valid Time\n")
        f.close()

        break_flag = 0
        min_loss_G_reg_v, min_loss_test = 1e20, 1e20

        for epoch in range(self.max_epoch):
            start_t = time.time()

            netG.train()
            text_encoder.train()

            # data_iter_train = iter(self.data_loader_train)
            step = 0
            gen_iterations = 0

            error_G_total_reg, error_G_total = 0.0, 0.0

            train_fake_list, train_real_list, train_input_list, train_mask_list = [], [], [], []

            for input_data, output_data, mask_input in self.data_loader_train:

                netG.zero_grad()
                text_encoder.zero_grad()

                noise.data.normal_(0, 1)

                if cfg.CUDA:
                    input_data = input_data.to(self.device).float()
                    output_data = output_data.to(self.device).float()
                    mask_input = mask_input.to(self.device)
                else:
                    input_data = input_data.float()
                    output_data = output_data.float()
                    mask_input = mask_input

                ######################################################
                # (1) Prepare training data and Compute text embeddings
                ######################################################

                # 对文本进行编码，但编码结果不参与反向传播所以要取消梯度关联;前者是编码层的输出，后者是其隐藏层状态
                # words_embs: batch x num_directions * hidden_size (两者乘起来即为nef) x seq_len
                # sent_emb: batch x num_directions * hidden_size (两者乘起来即为nef)
                # 这里的words_embs是单词尺度的编码结果，而sent_emb则是句子尺度的编码结果
                with autocast():

                    real_batch_size = input_data.shape[0]
                    noise_cut = noise[0:real_batch_size]
                    hidden = text_encoder.init_hidden(real_batch_size)
                    hidden = hidden.to(self.device)

                    words_embs, sent_emb = text_encoder(input_data, hidden)

                    num_words = words_embs.size(2)

                    # 注意mask只是一个二维张量，没有第三个维度
                    if mask_input.size(1) > num_words:
                        mask_input = mask_input[:, :num_words]

                    #######################################################
                    # (2) Generate images
                    #######################################################
                    # 输入随机噪声、send_emb和words_embs来生成虚假的图像

                    fake_imgs, _ = netG(noise_cut, sent_emb, words_embs, mask_input)
                    real_imgs = prepare_data(output_data)

                    reg_loss = 0.0
                    for i in range(cfg.TREE.BRANCH_NUM):
                        # print(fake_imgs[i].size(), real_imgs[i].size())
                        reg_loss_i = regression_loss(fake_imgs[i], real_imgs[i])
                        reg_loss += reg_loss_i

                    errG_total = reg_loss

                scaler.scale(errG_total).backward()
                scaler.step(optimizerG)
                scaler.step(optimizerE)

                scaler.update()

                error_G_total_reg += reg_loss.item() * input_data.size(0)
                error_G_total += errG_total.item() * input_data.size(0)
                gen_iterations += input_data.size(0)

                # 更新完之后计算参数的滑动平均值
                for p, avg_p in zip(netG.parameters(), avg_param_G):
                    avg_p.mul_(0.999).add_(p.data, alpha=0.001)

                step += 1

                train_fake_list.append(fake_imgs)
                train_real_list.append(real_imgs)
                train_input_list.append(input_data)
                train_mask_list.append(mask_input)

            ######################################################
            # 训练完成后，更新学习率并计算平均损失函数值
            ######################################################

            schedulerG.step()
            schedulerE.step()

            error_G_total_reg /= gen_iterations
            error_G_total /= gen_iterations
            end_t = time.time()

            print('''[%d/%d Train] Loss G Reg: %.6f   Loss G Overall: %.6f   Time: %s s'''
                  % (epoch, self.max_epoch, error_G_total_reg * 1000,
                     error_G_total * 1000, int(end_t - start_t)))

            f = open(f"{self.log_dir}/Log_Train.csv", "a")
            f.write(f"{epoch},{self.max_epoch},{error_G_total_reg},{error_G_total},{int(end_t - start_t)}\n")
            f.close()

            ######################################################
            # 开展模型验证，注意只需要验证生成器，而且只需要回归损失
            ######################################################

            error_valid, start_t, _, fake_imgs_list_save, real_imgs_list_save, input_seq_list, input_mask_list = \
                self.valid(netG, text_encoder, self.data_loader_valid, noise)

            Test_Flag = False
            if error_valid < min_loss_G_reg_v:
                min_loss_G_reg_v = error_valid
                Test_Flag = True

            # if Test_Flag:
                # self.save_result(fake_imgs_list_save, real_imgs_list_save, input_seq_list, input_mask_list, "Valid")
                # self.save_result(train_fake_list, train_real_list, train_input_list, train_mask_list, "Train")

            end_t = time.time()
            print('''[%d/%d Valid] Loss G Reg: %.6f    Min G Reg: %.6f   Time: %s s'''
                  % (epoch, self.max_epoch, error_valid * 1000, min_loss_G_reg_v * 1000, int(end_t - start_t)))

            f = open(f"{self.log_dir}/Log_Valid.csv", "a")
            f.write(f"{epoch},{self.max_epoch},{error_valid},{min_loss_G_reg_v},{int(end_t - start_t)}\n")
            f.close()

            if Test_Flag:

                error_test, start_t, _, fake_imgs_list_save, real_imgs_list_save, input_seq_list, input_mask_list = \
                    self.valid(netG, text_encoder, self.data_loader_test, noise)
                if error_test < min_loss_test:
                    min_loss_test = error_test

                self.save_model(netG, avg_param_G, text_encoder)

                self.save_result(fake_imgs_list_save, real_imgs_list_save, input_seq_list, input_mask_list, "Test")
                break_flag = 0

                end_t = time.time()
                print('''[%d/%d  Test] Loss G Reg: %.6f    Min G Reg: %.6f   Time: %s s'''
                      % (epoch, self.max_epoch, error_test * 1000, min_loss_test * 1000, int(end_t - start_t)))
                f = open(f"{self.log_dir}/Log_Test.csv", "a")
                f.write(f"{epoch},{self.max_epoch},{error_test},{min_loss_test},{int(end_t - start_t)}\n")
                f.close()

                print("Checkpoint and Figures Updated in Current Iteration")

            print("\n", "-" * 100, "\n")


    def valid(self, netG, text_encoder, data_loader, noise):

        start_t = time.time()

        with torch.no_grad():
            netG.eval()
            text_encoder.eval()
            # data_iter_valid = iter(self.data_loader_valid)
            generation, error_G_total_reg_v = 0, 0

            step = 0

            input_seq_list, input_mask_list = [], []
            fake_imgs_list_save, real_imgs_list_save = [], []

            for input_data_v, output_data_v, mask_v in data_loader:

                noise.data.normal_(0, 1)

                if cfg.CUDA:
                    input_data_v = input_data_v.to(self.device).float()
                    output_data_v = output_data_v.to(self.device).float()
                    mask_v = mask_v.to(self.device)
                else:
                    input_data_v = input_data_v.float()
                    output_data_v = output_data_v.float()
                    mask_v = mask_v

                real_batch_size = input_data_v.shape[0]
                noise_cut = noise[0:real_batch_size]
                hidden_v = text_encoder.init_hidden(real_batch_size)
                words_embs_v, sent_emb_v = text_encoder(input_data_v, hidden_v)

                num_words_v = words_embs_v.size(2)
                # 注意mask_v也是二维
                if mask_v.size(1) > num_words_v:
                    mask_v = mask_v[:, :num_words_v]

                fake_imgs_v, _ = netG(noise_cut, sent_emb_v, words_embs_v, mask_v)
                real_imgs_v = prepare_data(output_data_v)

                reg_loss_v = 0.0
                for i in range(cfg.TREE.BRANCH_NUM):
                    reg_loss_i_v = regression_loss(fake_imgs_v[i], real_imgs_v[i])
                    reg_loss_v += reg_loss_i_v

                error_G_total_reg_v += reg_loss_i_v.item() * input_data_v.size(0)
                generation += input_data_v.size(0)

                fake_imgs_list_save.append(fake_imgs_v)
                real_imgs_list_save.append(real_imgs_v)
                input_seq_list.append(input_data_v)
                input_mask_list.append(mask_v)

                step += 1

        error_G_total_reg_v /= generation
        end_t = time.time()

        return error_G_total_reg_v, start_t, end_t, fake_imgs_list_save, real_imgs_list_save, input_seq_list, input_mask_list


    def get_text_array(self, fake_img_sel, vis_size, text_img_height, prefix):
        text_image = Image.new('L', (vis_size, text_img_height), 255)  # 灰度模式
        draw = ImageDraw.Draw(text_image)
        # 设置字体（需要指定字体文件路径）
        fontsize = 50
        font = ImageFont.truetype('C:\\Windows\\Fonts\\times.ttf', size=fontsize)  # 替换为您的字体路径和大小
        # 在图像上添加文本
        text = f"{prefix} Img Min {np.min(fake_img_sel):.6f} Max {np.max(fake_img_sel):.6f}"  # 替换为您要添加的文本
        text_width = draw.textlength(text, font=font)
        text_height = fontsize
        text_position = (int(vis_size / 2 - text_width / 2), int(text_img_height / 2 - text_height / 2))  # 文本位置
        draw.text(text_position, text, fill=0, font=font)  # 黑色文本
        text_image_array = np.array(text_image) / 255
        text_image_array = np.expand_dims(text_image_array, axis=2)

        return text_image_array


    def save_model(self, netG, avg_param_G, text_encoder):
        backup_para = copy_G_params(netG)
        load_params(netG, avg_param_G)
        torch.save(netG.state_dict(), '%s/netG_ave.pth' % self.model_dir)
        load_params(netG, backup_para)
        torch.save(netG.state_dict(), '%s/netG_best.pth' % self.model_dir)

        torch.save(text_encoder.state_dict(), '%s/text_encoder_best.pth' % self.model_dir)


    def save_result(self, fake_imgs_list_save, real_imgs_list_save, input_data_list, input_mask_list, prefix):

        result_dict = {}

        input_data_save = []
        input_mask_save = []
        for input_data in input_data_list:
            input_data_save.append(input_data.detach().cpu().numpy())

        for input_mask in input_mask_list:
            input_mask_save.append(input_mask.detach().cpu().numpy())

        input_data_save = np.concatenate(input_data_save, axis=0)
        input_mask_save = np.concatenate(input_mask_save, axis=0)
        result_dict["Input_Data"] = input_data_save
        result_dict["Mask_Data"] = input_mask_save

        for num_level in range(cfg.TREE.BRANCH_NUM):
            total_batch_number = len(fake_imgs_list_save)
            select_level_fake, select_level_real = [], []
            # 将每个尺寸（level）的结果分别提取出，拼接到一起形成一个整体文件
            for i in range(total_batch_number):
                select_level_fake.append(fake_imgs_list_save[i][num_level].detach().cpu().numpy())
                select_level_real.append(real_imgs_list_save[i][num_level].detach().cpu().numpy())
            select_level_fake = np.concatenate(select_level_fake, axis=0)
            select_level_real = np.concatenate(select_level_real, axis=0)

            # if num_level == cfg.TREE.BRANCH_NUM - 1 and prefix == "Test":
            #     self.update_figure(select_level_real, select_level_fake, input_data_save, input_mask_save, num_level, prefix)

            result_dict["Real_Level_%s" % num_level] = select_level_real
            result_dict["Fake_Level_%s" % num_level] = select_level_fake

        savemat(f"{self.data_dir}/{prefix}_Data.mat", result_dict)


    def update_figure(self, real_img, fake_img, input_seq, input_mask, num_level, suffix):

        max_plot_number = 3
        vis_size = 1024
        # text_height = 100

        fake_img = fake_img[0:max_plot_number, :, :, :]
        real_img = real_img[0:max_plot_number, :, :, :]
        input_seq = input_seq[0:max_plot_number, :, :]
        input_mask = input_mask[0:max_plot_number, :]

        fake_img = np.transpose(fake_img, (0, 2, 3, 1))
        real_img = np.transpose(real_img, (0, 2, 3, 1))

        # middle_pad = np.ones((vis_size, 20, 1))
        # middle_pad_text = np.ones((text_height, 20, 1))

        for i in range(max_plot_number):
            fake_img_sel = fake_img[i]
            real_img_sel = real_img[i]
            input_seq_sel = input_seq[i]
            input_mask_sel = input_mask[i]

            fake_img_sel = skimage.transform.resize(fake_img_sel, (vis_size, vis_size))
            fake_img_sel = np.clip(fake_img_sel, 0, 1)
            real_img_sel = skimage.transform.resize(real_img_sel, (vis_size, vis_size))
            dev_abs = np.abs((fake_img_sel - real_img_sel))

            save_path = os.path.join(self.image_dir, f'{suffix}_Input_{i}.png')

            if not os.path.exists(save_path):
                # 创建一个图形和两个子图
                fig, axs = plt.subplots(1, 2, figsize=(12, 6))  # figsize可以根据需要调整

                # 第一个子图：Input Sequence
                axs[0].plot(input_seq_sel[:, 0], input_seq_sel[:, 1])
                axs[0].set_title("Input Sequence")

                # 第二个子图：Input Mask
                axs[1].plot(input_seq_sel[:, 0], input_mask_sel)
                axs[1].set_title("Input Mask")

                # 调整子图布局
                plt.tight_layout()

                # 保存图形
                plt.savefig(save_path)

                # 关闭图形，释放资源
                plt.close(fig)

            for output_dim in range(cfg.output_dim):

                fake_img_sel_dim = fake_img_sel[:, :, output_dim]
                real_img_sel_dim = real_img_sel[:, :, output_dim]
                dev_abs_sel_dim = dev_abs[:, :, output_dim]

                fig, axs = plt.subplots(1, 3, figsize=(18, 6))  # 创建一个1行3列的子图布局，总宽度为18英寸，高度为6英寸

                # 第一个子图
                axs[0].imshow(dev_abs_sel_dim)
                axs[0].set_title("Dev_Img_%.4f_%.4f" % (np.max(dev_abs_sel_dim), np.min(dev_abs_sel_dim)))
                axs[0].axis("off")

                # 第二个子图
                axs[1].imshow(fake_img_sel_dim)
                axs[1].set_title("Fake_Img_%.4f_%.4f" % (np.max(fake_img_sel_dim), np.min(fake_img_sel_dim)))
                axs[1].axis("off")

                # 第三个子图
                axs[2].imshow(real_img_sel_dim)
                axs[2].set_title("Real_Img_%.4f_%.4f" % (np.max(real_img_sel_dim), np.min(real_img_sel_dim)))
                axs[2].axis("off")

                # 保存整个图形
                save_path = os.path.join(self.image_dir,
                                         f'{suffix}_img_{i}_Dim_{output_dim}_Level_{num_level}_Output.png')
                plt.savefig(save_path)

                plt.close(fig)  # 关闭图形