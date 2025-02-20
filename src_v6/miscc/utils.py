import os
import errno
import numpy as np
from torch.nn import init

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont
from copy import deepcopy
import skimage.transform

from miscc.config import cfg
from io import BytesIO

# For visualization ################################################
COLOR_DIC = {0: [128, 64, 128],  1: [244, 35, 232],
             2: [70, 70, 70],  3: [102, 102, 156],
             4: [190, 153, 153], 5: [153, 153, 153],
             6: [250, 170, 30], 7: [220, 220, 0],
             8: [107, 142, 35], 9: [152, 251, 152],
             10: [70, 130, 180], 11: [220, 20, 60],
             12: [255, 0, 0],  13: [0, 0, 142],
             14: [119, 11, 32], 15: [0, 60, 100],
             16: [0, 80, 100], 17: [0, 0, 230],
             18: [0,  0, 70],  19: [0, 0,  0]}
FONT_MAX = 50


def plot_to_image(captions, size):
    plt.figure(figsize=(8, 8))  # 设置图表大小
    plt.plot(captions, marker='o')
    # plt.xticks([])
    # plt.yticks([])

    buf = BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)

    # 使用PIL读取图像
    image = Image.open(buf)
    image = image.resize(size)
    # 将图像转换为NumPy数组
    image_array = np.array(image)[:, :, 0:3]
    buf.close()
    plt.close()

    # 将保存的图像转换为 PIL 图像
    return image_array


# 调用：text_map, sentences = drawCaption(text_convas, captions, ixtoword, vis_size)
# def drawCaption(convas, captions, ixtoword, vis_size, off1=2, off2=2):
#     # 读取数目信息，并将代表空白画布的数组转换为图像
#     num = captions.size(0)
#     img_txt = Image.fromarray(convas)
#
#     captions = captions * (cfg.MAX_VALUE_X - cfg.MIN_VALUE_X) + cfg.MIN_VALUE_X
#
#     # sentence_list = []
#     for i in range(num):
#         cap = captions[i].squeeze().detach().cpu().numpy()  # 假设每个序列是 14 x 1
#         line_plot = plot_to_image(cap, size=(vis_size, vis_size))
#         # off1：水平偏移，恒定为2像素，主要用于给分割线留位置（分割线宽度2像素）
#         # off2：竖直偏移，对于每个图而言，vis_size + off2（2像素）是其尺寸，第i个图需要向下偏移i次（估计是左下角定位方式）
#         position = (off1, i * (vis_size + off2))
#         img_txt.paste(line_plot, position)
#
#     return img_txt, captions

'''
# 调用：build_super_images(imgs[-1].cpu(), captions, ixtoword, attn_maps, att_sze)
def build_super_images(real_imgs, captions, ixtoword,
                       attn_maps, att_sze, lr_imgs=None,
                       batch_size=cfg.TRAIN.BATCH_SIZE,
                       max_word_num=cfg.TEXT.SEQ_LENGTH):

    # 选择需要绘制的对象（最多8个）
    nvis = 8
    real_imgs = real_imgs[:nvis]

    if lr_imgs is not None:
        lr_imgs = lr_imgs[:nvis]
    if att_sze == 17:
        vis_size = att_sze * 16
    else:
        vis_size = 256

    # 创建一个用于显示文本描述的大型白色画布（text_convas）
    text_convas = np.ones([batch_size * vis_size, (max_word_num + 2) * (vis_size + 2), 3], dtype=np.uint8)

    for i in range(max_word_num):
        istart = (i + 2) * (vis_size + 2)
        iend = (i + 3) * (vis_size + 2)
        text_convas[:, istart:iend, :] = COLOR_DIC[i]

    # 将real_imgs统一采样到(vis_size, vis_size)，方便后续观察
    real_imgs = \
        nn.Upsample(size=(vis_size, vis_size), mode='bilinear')(real_imgs)
    # real_imgs.mul_((cfg.MAX_VALUE_Y - cfg.MIN_VALUE_Y)).add_(cfg.MIN_VALUE_Y)
    real_imgs = real_imgs.data.numpy()
    # b x c x h x w --> b x h x w x c
    # 交换维度方便绘图，并获取真实图像尺寸
    real_imgs = np.transpose(real_imgs, (0, 2, 3, 1))
    pad_sze = real_imgs.shape
    # print(pad_sze)
    # 创建黑色（纯0）视觉填充条用于图像分割
    middle_pad = np.zeros([pad_sze[2], 2, 3])  # W * 2 * 3，宽2像素高度和图像一致的纯黑（3通道为0）分割条
    post_pad = np.zeros([pad_sze[1], pad_sze[2], 3])  # W * B * 3，宽高和图像一致的纯黑（3通道为0）分割条，用于不同样本之间的大分割线

    # 这里的lr是low resolution的意思，就是如果有低分辨率图像，先对其进行缩放等处理并进行格式和通道转换
    # （上面real_imgs只传进来了最后一个最高分辨率的）
    if lr_imgs is not None:
        lr_imgs = nn.Upsample(size=(vis_size, vis_size), mode='bilinear')(lr_imgs)
        # [-1, 1] --> [0, 1]
        # lr_imgs.mul_(255)
        lr_imgs = lr_imgs.data.numpy()
        # b x c x h x w --> b x h x w x c
        lr_imgs = np.transpose(lr_imgs, (0, 2, 3, 1))

    text_map, sentences = drawCaption(text_convas, captions, ixtoword, vis_size)
    text_map = np.asarray(text_map).astype(np.uint8)

    # batch x seq_len x 17 x 17 --> batch x 1 x 17 x 17
    seq_len = max_word_num
    img_set = []
    num = nvis  # len(attn_maps)

    bUpdate = 1
    # 对所有注意力图进行循环，并叠加到原始图像上
    for i in range(num):
        # 将第 i 个注意力图转移到 CPU 并调整其形状，使其成为一个四维张量
        attn = attn_maps[i].cpu().view(1, -1, att_sze, att_sze)
        # 找到注意力图中的最大值，用于突出显示最强的注意力区域；--> 1 x 1 x 17 x 17
        attn_max = attn.max(dim=1, keepdim=True)
        # 将原始的注意力图和最大值图合并
        attn = torch.cat([attn_max[0], attn], 1)
        # 调整张量的形状以适应后续处理
        # B * 1 * 17 * 17
        attn = attn.view(-1, 1, att_sze, att_sze).detach().cpu().numpy()

        # print("attn map shape", attn.shape)
        # 将注意力图重复三次以匹配 RGB 通道，然后转换为 NumPy 数组，最后转变为绘图需要的格式（通道在最后）
        # attn = attn.repeat(1, 3, 1, 1).data.numpy()
        # n x c x h x w --> n x h x w x c

        # B * 17 * 17 * 1
        attn = np.transpose(attn, (0, 2, 3, 1))
        num_attn = attn.shape[0]

        img = real_imgs[i]

        if lr_imgs is None:
            lrI = img
        else:
            lrI = lr_imgs[i]

        # 这里是把原始图像和分隔条合并，后面一起绘制
        row = [lrI, middle_pad]
        row_merge = [img, middle_pad]
        row_beforeNorm = []
        minVglobal, maxVglobal = 1, 0

        for j in range(num_attn):
            # 循环遍历每个注意力图，对其进行标准化和缩放处理，使其尺寸与原始图像匹配
            one_map = attn[j].squeeze()

            # 272 * 272
            if (vis_size // att_sze) > 1:
                one_map = \
                    skimage.transform.pyramid_expand(one_map, sigma=20,
                                                     upscale=vis_size // att_sze)
            # 272 * 272 * 1
            one_map = np.expand_dims(one_map, axis=-1)
            # 272 * 272 * 3
            one_map = np.concatenate((one_map, one_map, one_map), axis=2)
            row_beforeNorm.append(one_map)

            # 调取注意力图的最大值和最小值，并替换预先设定的全局范围方便展示注意力值的相对大小
            minV = one_map.min()
            maxV = one_map.max()
            if minVglobal > minV:
                minVglobal = minV
            if maxVglobal < maxV:
                maxVglobal = maxV

        # print("0", one_map.shape)

        for j in range(seq_len + 1):
            if j < num_attn:
                # 对图像进行一定程度的标准化（这里还需要进一步研究怎么标准化比较好看！）
                one_map = row_beforeNorm[j]
                one_map = (one_map - minVglobal) / (maxVglobal - minVglobal)
                # one_map *= 255
                # 通过创建一个新的图像 merged 并将注意力图作为蒙版贴在原始图像上实现融合绘图
                # 将原始图像和注意力机制图像转化为PIL图像
                red, green, blue = np.zeros_like(img[:, :, 0:1]), np.zeros_like(img[:, :, 0:1]), img[:, :, 0:1]
                img = np.concatenate((red, green, blue), axis=2)
                PIL_im = Image.fromarray(np.uint8(img))
                # PIL_im = Image.merge("RGB", (Image.fromarray(red), Image.fromarray(green), Image.fromarray(blue)))

                red, green, blue = np.uint8(one_map[:, :, 0:1]), np.uint8(np.zeros_like(one_map[:, :, 0:1])), \
                                   np.uint8(np.zeros_like(one_map[:, :, 0:1]))
                att_img = np.concatenate((red, green, blue), axis=2)
                # PIL_att = Image.merge("RGB", (Image.fromarray(red), Image.fromarray(green), Image.fromarray(blue)))
                PIL_att = Image.fromarray(np.uint8(att_img))
                merged = Image.new('RGBA', (vis_size, vis_size), (0, 0, 0, 0))
                # 创建一个灰度蒙版 mask，用于控制 PIL_att 图像的透明度
                mask = Image.new('L', (vis_size, vis_size), (155))
                # 将两个图像粘贴到新的图像上
                merged.paste(PIL_im, (0, 0))
                merged.paste(PIL_att, (0, 0), mask)
                merged = np.array(merged)[:, :, :3]
            else:
                one_map = post_pad
                merged = post_pad

            row.append(one_map)
            row.append(middle_pad)
            #
            row_merge.append(merged)
            row_merge.append(middle_pad)

        row = np.concatenate(row, 1)
        row_merge = np.concatenate(row_merge, 1)

        # 这里的txt原来是文本中的单词，后来用了序列中的数字替代；同时，FONT_MAX这个原来针对文本设置的格式也自然的不合适了
        # 需要用vis_size替代；原来的设计中，每个单词用不同的颜色
        txt = text_map[i * vis_size: (i + 1) * vis_size]
        if txt.shape[1] != row.shape[1]:
            print('txt', txt.shape, 'row', row.shape)
            bUpdate = 0
            break
        # 这里是对每个样本进行了图像整合，第一列从上到下分别是文本（序列）信息分解、对应的低分辨率图像、对应的高分辨率图像
        # 第二列开始分别是图像和对应的attention map的整合结果（总的组数是batch size）
        row = np.concatenate([txt, row, row_merge], 0)
        img_set.append(row)

    if bUpdate:
        img_set = np.concatenate(img_set, 0)
        img_set = img_set.astype(np.uint8)
        return img_set, sentences
    else:
        return None


def build_super_images2(real_imgs, captions, cap_lens, ixtoword,
                        attn_maps, att_sze, vis_size=256, topK=5):

    batch_size = real_imgs.size(0)
    max_word_num = np.max(cap_lens)
    # 创建一个用于显示文本描述的大型白色画布（text_convas）
    text_convas = np.ones([batch_size * FONT_MAX,
                           max_word_num * (vis_size + 2), 3],
                           dtype=np.uint8)

    # 将real_imgs统一采样到(vis_size, vis_size)，方便后续观察
    real_imgs = \
        nn.Upsample(size=(vis_size, vis_size), mode='bilinear')(real_imgs)
    # [-1, 1] --> [0, 1]
    # real_imgs.add_(1).div_(2).mul_(255)
    # 对数据集进行还原
    real_imgs.mul_((cfg.MAX_VALUE_Y - cfg.MIN_VALUE_Y)).add_(cfg.MIN_VALUE_Y)
    real_imgs = real_imgs.data.numpy()
    # b x c x h x w --> b x h x w x c
    real_imgs = np.transpose(real_imgs, (0, 2, 3, 1))
    pad_sze = real_imgs.shape
    middle_pad = np.zeros([pad_sze[2], 2, 3])

    text_map, sentences = \
        drawCaption(text_convas, captions, ixtoword, vis_size, off1=0)
    text_map = np.asarray(text_map).astype(np.uint8)

    # batch x seq_len x 17 x 17 --> batch x 1 x 17 x 17
    img_set = []
    num = len(attn_maps)

    bUpdate = 1
    for i in range(num):
        attn = attn_maps[i].cpu().view(1, -1, att_sze, att_sze)
        #
        attn = attn.view(-1, 1, att_sze, att_sze)
        attn = attn.repeat(1, 3, 1, 1).data.numpy()
        # n x c x h x w --> n x h x w x c
        attn = np.transpose(attn, (0, 2, 3, 1))
        num_attn = cap_lens[i]
        thresh = 2./float(num_attn)
        #
        img = real_imgs[i]
        row = []
        row_merge = []
        row_txt = []
        row_beforeNorm = []
        conf_score = []
        for j in range(num_attn):
            one_map = attn[j]
            mask0 = one_map > (2. * thresh)
            conf_score.append(np.sum(one_map * mask0))
            mask = one_map > thresh
            one_map = one_map * mask
            if (vis_size // att_sze) > 1:
                one_map = \
                    skimage.transform.pyramid_expand(one_map, sigma=20,
                                                     upscale=vis_size // att_sze)
            minV = one_map.min()
            maxV = one_map.max()
            one_map = (one_map - minV) / (maxV - minV)
            row_beforeNorm.append(one_map)
        sorted_indices = np.argsort(conf_score)[::-1]

        for j in range(num_attn):
            one_map = row_beforeNorm[j]
            one_map *= 255
            #
            PIL_im = Image.fromarray(np.uint8(img))
            PIL_att = Image.fromarray(np.uint8(one_map))
            merged = \
                Image.new('RGBA', (vis_size, vis_size), (0, 0, 0, 0))
            mask = Image.new('L', (vis_size, vis_size), (180))  # (210)
            merged.paste(PIL_im, (0, 0))
            merged.paste(PIL_att, (0, 0), mask)
            merged = np.array(merged)[:, :, :3]

            row.append(np.concatenate([one_map, middle_pad], 1))
            #
            row_merge.append(np.concatenate([merged, middle_pad], 1))
            #
            txt = text_map[i * FONT_MAX:(i + 1) * FONT_MAX,
                           j * (vis_size + 2):(j + 1) * (vis_size + 2), :]
            row_txt.append(txt)
        # reorder
        row_new = []
        row_merge_new = []
        txt_new = []
        for j in range(num_attn):
            idx = sorted_indices[j]
            row_new.append(row[idx])
            row_merge_new.append(row_merge[idx])
            txt_new.append(row_txt[idx])
        row = np.concatenate(row_new[:topK], 1)
        row_merge = np.concatenate(row_merge_new[:topK], 1)
        txt = np.concatenate(txt_new[:topK], 1)
        if txt.shape[1] != row.shape[1]:
            print('Warnings: txt', txt.shape, 'row', row.shape,
                  'row_merge_new', row_merge_new.shape)
            bUpdate = 0
            break
        row = np.concatenate([txt, row_merge], 0)
        img_set.append(row)
    if bUpdate:
        img_set = np.concatenate(img_set, 0)
        img_set = img_set.astype(np.uint8)
        return img_set, sentences
    else:
        return None
'''


def build_not_super_images(real_imgs, captions, attn_maps, att_sze, lr_imgs, image_dir, suffix):

    # 选择需要绘制的对象（最多8个）
    nvis = 8
    vis_size = 1024
    real_imgs = real_imgs[:nvis]
    captions = captions[:nvis]
    lr_imgs = lr_imgs[:nvis]
    attn_maps = attn_maps[:nvis]

    # captions = captions * (cfg.MAX_VALUE_X - cfg.MIN_VALUE_X) + cfg.MIN_VALUE_X

    # 将real_imgs统一采样到(vis_size, vis_size)，方便后续观察
    real_imgs = nn.Upsample(size=(vis_size, vis_size), mode='bilinear')(real_imgs)
    # real_imgs.mul_((cfg.MAX_VALUE_Y - cfg.MIN_VALUE_Y)).add_(cfg.MIN_VALUE_Y)
    real_imgs = real_imgs.data.numpy()
    # b x c x h x w --> b x h x w x c
    # 交换维度方便绘图，并获取真实图像尺寸
    real_imgs = np.transpose(real_imgs, (0, 2, 3, 1))
    pad_sze = real_imgs.shape
    # 创建黑色（纯0）视觉填充条用于图像分割
    middle_pad = np.zeros([pad_sze[2], 10, 3])  # W * 10 * 3，宽10像素高度和图像一致的纯黑（3通道为0）分割条
    middle_pad_white = np.ones([pad_sze[2], 10, 3])  # W * 10 * 3，宽10像素高度和图像一致的白色（3通道为1）分割条

    # 这里的lr是low resolution的意思，就是如果有低分辨率图像，先对其进行缩放等处理并进行格式和通道转换
    # （上面real_imgs只传进来了最后一个最高分辨率的）
    lr_imgs = nn.Upsample(size=(vis_size, vis_size), mode='bilinear')(lr_imgs)
    # lr_imgs.mul_((cfg.MAX_VALUE_Y - cfg.MIN_VALUE_Y)).add_(cfg.MIN_VALUE_Y)
    lr_imgs = lr_imgs.data.numpy()
    # b x c x h x w --> b x h x w x c
    lr_imgs = np.transpose(lr_imgs, (0, 2, 3, 1))

    # print()
    # target_array = np.concatenate((real_imgs, middle_pad, lr_imgs), axis=2)

    for i in range(real_imgs.shape[0]):
        # plot_array = real_imgs[i]
        sel_imgs_real = real_imgs[i]
        sel_imgs_lr = lr_imgs[i]
        cap = captions[i].squeeze().detach().cpu().numpy()  # 假设每个序列是 14 x 1
        line_plot = plot_to_image(cap, size=(vis_size, vis_size))
        line_plot = line_plot / 255

        # print(type(line_plot), type(sel_imgs_lr), type(pad_sze), type(sel_imgs_lr))
        # print(line_plot.shape, sel_imgs_lr.shape, middle_pad.shape, sel_imgs_lr.shape)
        plot_array = np.concatenate((line_plot, middle_pad, sel_imgs_real, middle_pad, sel_imgs_lr), axis=1)

        plt.figure(figsize=(21, 7))
        # target_array = resize_image(target_array, target_shape)
        # plt.imshow(plot_array, cmap=cfg.PLOT.COLORMAP)  # 使用自定义的 colormap
        plt.imshow(plot_array, cmap="gray")  # 使用自定义的 colormap
        plt.axis('off')  # 关闭坐标轴
        plt.colorbar()
        filename = os.path.join(image_dir, f'img_{i}_{cfg.KEY_Y}_{suffix}.png')
        plt.savefig(filename)
        plt.close()

    # 对所有注意力图进行循环，并叠加到原始图像上
    # for i in range(len(attn_maps)):
        # 将第 i 个注意力图转移到 CPU 并调整其形状，使其成为一个四维张量
        attn = attn_maps[i].cpu().view(1, -1, att_sze, att_sze)
        # 找到注意力图中的最大值，用于突出显示最强的注意力区域；--> 1 x 1 x 17 x 17
        attn_max = attn.max(dim=1, keepdim=True)
        # 将原始的注意力图和最大值图合并
        attn = torch.cat([attn_max[0], attn], 1)
        # 调整张量的形状以适应后续处理
        # B * 1 * 17 * 17
        attn = attn.view(-1, 1, att_sze, att_sze).detach().cpu().numpy()
        # B * 17 * 17 * 1
        attn = np.transpose(attn, (0, 2, 3, 1))
        num_attn = attn.shape[0]
        atten_array = sel_imgs_real

        for j in range(num_attn):
            # 循环遍历每个注意力图，对其进行标准化和缩放处理，使其尺寸与原始图像匹配
            one_map = attn[j].squeeze()

            # vis_size * vis_size
            if (vis_size // att_sze) > 1:
                one_map = skimage.transform.pyramid_expand(one_map, sigma=20,
                                                     upscale=vis_size // att_sze)
                one_map = skimage.transform.resize(one_map, (vis_size, vis_size))
            # vis_size * vis_size * 1
            one_map = np.expand_dims(one_map, axis=-1)
            # vis_size * vis_size * 3
            one_map = np.concatenate((one_map, one_map, one_map), axis=2)
            # row_beforeNorm.append(one_map)

            # 调取注意力图的最大值和最小值，并替换预先设定的全局范围方便展示注意力值的相对大小
            minV = one_map.min()
            maxV = one_map.max()
            minVglobal = max(0.0, minV)
            maxVglobal = min(1.0, maxV)

            # 对图像进行一定程度的标准化（这里还需要进一步研究怎么标准化比较好看！）
            # one_map = row_beforeNorm[j]
            one_map = (one_map - minVglobal) / (maxVglobal - minVglobal + 1e-8)
            atten_array = np.concatenate((atten_array, middle_pad_white, one_map), axis=1)

        atten_array = np.concatenate((line_plot, atten_array), axis=1)
        plt.figure(figsize=(165.34, 10.24), dpi=100)
        # plt.figure(10.24)
        # target_array = resize_image(target_array, target_shape)
        # plt.imshow(atten_array, cmap=cfg.PLOT.COLORMAP)  # 使用自定义的 colormap
        # print(atten_array.shape)
        plt.imshow(atten_array, cmap="gray")  # 使用自定义的 colormap
        plt.axis('off')  # 关闭坐标轴
        plt.tight_layout()  # 调整布局以填满画布
        plt.subplots_adjust(left=0, right=1, top=1, bottom=0)  # 移除周围空白
        # plt.colorbar()
        filename = os.path.join(image_dir, f'img_{i}_Attention_{suffix}.png')
        plt.savefig(filename)
        plt.close()

    # 这里的txt原来是文本中的单词，后来用了序列中的数字替代；同时，FONT_MAX这个原来针对文本设置的格式也自然的不合适了
    # 需要用vis_size替代；原来的设计中，每个单词用不同的颜色
    #     txt = text_map[i * vis_size: (i + 1) * vis_size]
    #     if txt.shape[1] != row.shape[1]:
    #         print('txt', txt.shape, 'row', row.shape)
    #         bUpdate = 0
    #         break
    #     # 这里是对每个样本进行了图像整合，第一列从上到下分别是文本（序列）信息分解、对应的低分辨率图像、对应的高分辨率图像
    #     # 第二列开始分别是图像和对应的attention map的整合结果（总的组数是batch size）
    #     row = np.concatenate([txt, row, row_merge], 0)
    #     img_set.append(row)
    #
    # if bUpdate:
    #     img_set = np.concatenate(img_set, 0)
    #     img_set = img_set.astype(np.uint8)
    #     return img_set, sentences
    # else:
    #     return None


####################################################################
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.orthogonal_(m.weight.data, 1.0)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)
    elif classname.find('Linear') != -1:
        nn.init.orthogonal_(m.weight.data, 1.0)
        if m.bias is not None:
            m.bias.data.fill_(0.0)


def load_params(model, new_param):
    for p, new_p in zip(model.parameters(), new_param):
        p.data.copy_(new_p)


def copy_G_params(model):
    flatten = deepcopy(list(p.data for p in model.parameters()))
    return flatten


def mkdir_p(path):
    try:
        os.makedirs(path)
    except OSError as exc:  # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise
