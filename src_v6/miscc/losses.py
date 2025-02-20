import torch
import torch.nn as nn

import numpy as np
from miscc.config import cfg

from GlobalAttention import func_attention


# ##################Loss for matching text-image###################
def cosine_similarity(x1, x2, dim=1, eps=1e-8):
    """Returns cosine similarity between x1 and x2, computed along dim.
    """
    w12 = torch.sum(x1 * x2, dim)
    w1 = torch.norm(x1, 2, dim)
    w2 = torch.norm(x2, 2, dim)
    return (w12 / (w1 * w2).clamp(min=eps)).squeeze()


def sent_loss(cnn_code, rnn_code, labels, class_ids,
              batch_size, eps=1e-8):
    # ### Mask mis-match samples  ###
    # that come from the same class as the real sample ###
    masks = []
    if class_ids is not None:
        for i in range(batch_size):
            mask = (class_ids == class_ids[i]).astype(np.uint8)
            mask[i] = 0
            masks.append(mask.reshape((1, -1)))
        masks = np.concatenate(masks, 0)
        # masks: batch_size x batch_size
        masks = torch.ByteTensor(masks).bool()
        if cfg.CUDA:
            masks = masks.cuda()

    # --> seq_len x batch_size x nef
    if cnn_code.dim() == 2:
        cnn_code = cnn_code.unsqueeze(0)
        rnn_code = rnn_code.unsqueeze(0)

    # cnn_code_norm / rnn_code_norm: seq_len x batch_size x 1
    cnn_code_norm = torch.norm(cnn_code, 2, dim=2, keepdim=True)
    rnn_code_norm = torch.norm(rnn_code, 2, dim=2, keepdim=True)
    # scores* / norm*: seq_len x batch_size x batch_size
    scores0 = torch.bmm(cnn_code, rnn_code.transpose(1, 2))
    norm0 = torch.bmm(cnn_code_norm, rnn_code_norm.transpose(1, 2))
    scores0 = scores0 / norm0.clamp(min=eps) * cfg.TRAIN.SMOOTH.GAMMA3

    # --> batch_size x batch_size
    scores0 = scores0.squeeze()
    if class_ids is not None:
        scores0.data.masked_fill_(masks, -float('inf'))
    scores1 = scores0.transpose(0, 1)
    if labels is not None:
        loss0 = nn.CrossEntropyLoss()(scores0, labels)
        loss1 = nn.CrossEntropyLoss()(scores1, labels)
    else:
        loss0, loss1 = None, None
    return loss0, loss1


def words_loss(img_features, words_emb, labels,
               cap_lens, class_ids, batch_size):
    """
        words_emb(query): batch x nef x seq_len
        img_features(context): batch x nef x 17 x 17
    """
    masks = []
    att_maps = []
    similarities = []
    cap_lens = cap_lens.data.tolist()
    for i in range(batch_size):
        # 这里只计算同一个类别样本的损失；如果类别没有意义，那生成的本来就是一个等差数列，那所有的mask都会是0，也就是都参与计算
        # 否则有些是0，但同一类的都是1（除了这个mask[i]）
        if class_ids is not None:
            mask = (class_ids == class_ids[i]).astype(np.uint8)
            mask[i] = 0
            masks.append(mask.reshape((1, -1)))
        # Get the i-th text description（这里是第i个句子的长度）
        words_num = cap_lens[i]
        # -> 1 x nef x words_num
        # 取出第i个样本（句子）中所有有意义的编码结果（不超过总单词数目，也即不是padding的）
        word = words_emb[i, :, :words_num].unsqueeze(0).contiguous()
        # -> batch_size x nef x words_num
        # 将其扩展为一个batch，方便处理
        word = word.repeat(batch_size, 1, 1)
        # batch x nef x 17*17
        context = img_features
        """
            word(query): batch x nef x words_num
            context: batch x nef x 17 x 17
            weiContext: batch x nef x words_num
            attn: batch x words_num x 17 x 17
        """
        # 计算相似矩阵，获得加权处理后的weiContext以及权重信息；GAMMA1不知道咋取的，可能是试出来的
        weiContext, attn = func_attention(word, context, cfg.TRAIN.SMOOTH.GAMMA1)
        att_maps.append(attn[i].unsqueeze(0).contiguous())
        # --> batch_size x words_num x nef
        word = word.transpose(1, 2).contiguous()
        weiContext = weiContext.transpose(1, 2).contiguous()
        # --> batch_size*words_num x nef
        word = word.view(batch_size * words_num, -1)
        weiContext = weiContext.view(batch_size * words_num, -1)

        # 计算余弦相似度word（单词编码本身）和weiContext（加权后单词编码）的相似度
        # -->batch_size*words_num
        row_sim = cosine_similarity(word, weiContext)
        # --> batch_size x words_num
        row_sim = row_sim.view(batch_size, words_num)

        # Eq. (10)；GAMMA2也是个参数，可能是试出来的
        row_sim.mul_(cfg.TRAIN.SMOOTH.GAMMA2).exp_()
        row_sim = row_sim.sum(dim=1, keepdim=True)
        row_sim = torch.log(row_sim)

        # --> 1 x batch_size
        # similarities(i, j): the similarity between the i-th image and the j-th text description
        similarities.append(row_sim)

    # batch_size x batch_size
    similarities = torch.cat(similarities, 1)
    if class_ids is not None:
        masks = np.concatenate(masks, 0)
        # masks: batch_size x batch_size
        masks = torch.ByteTensor(masks).bool()
        if cfg.CUDA:
            masks = masks.cuda()

    similarities = similarities * cfg.TRAIN.SMOOTH.GAMMA3
    # 执行掩码处理，避免不必要的类别的影响；本案例中，一张图像就是一个类别
    # 注意！如果掩码是0反而不会处理，如果掩码是1反而会填充为-inf，也即忽略其影响，所以上面创建掩码的时候同一类别及自身是0
    if class_ids is not None:
        similarities.data.masked_fill_(masks, -float('inf'))
    similarities1 = similarities.transpose(0, 1)
    # 这里计算的是交叉熵损失，也就是看看所有的similarities是否和对应的类别（本案例就是一个编号）相近
    # 理论上，最佳情况就是一个单词向量和对应位置的图像最为接近
    # 注意CrossEntropyLoss的label输入不是one hot！而是一个整数！
    if labels is not None:
        loss0 = nn.CrossEntropyLoss()(similarities, labels)
        loss1 = nn.CrossEntropyLoss()(similarities1, labels)
    else:
        loss0, loss1 = None, None
    return loss0, loss1, att_maps


# ##################Loss for G and Ds##############################
def discriminator_loss(netD, real_imgs, fake_imgs, conditions,
                       real_labels, fake_labels):

    # Forward，这两个是判别器的判别结果，分别对应真实图像和生成的虚假图像
    # print("real_imgs", real_imgs.size())
    # print("fake_imgs", fake_imgs.size())

    real_features = netD(real_imgs)
    fake_features = netD(fake_imgs.detach())

    # 将真实的图像和conditions（句子尺度的编码结果）一同输入netD的最后一组判别层，执行融合、判别
    cond_real_logits = netD.COND_DNET(real_features, conditions)
    # 注意这里必须使用BCELoss，不能改成Cross-Entropy，这两者在二分类情况下并不等效！
    cond_real_errD = nn.BCELoss()(cond_real_logits, real_labels)
    # 同上，但是对生成图像进行处理
    cond_fake_logits = netD.COND_DNET(fake_features, conditions)
    cond_fake_errD = nn.BCELoss()(cond_fake_logits, fake_labels)

    # 这部分代码生成了错误配对的logits
    # 在条件GAN中，除了区分真实图像和生成图像，还需要确保图像与其对应的条件（这里是文本描述的编码结果）是匹配的
    # 因此，这里不匹配的情况下，理论上应该都会判别为“假”（这就是为什么生成数据集的时候每个数据集只能有一个句子被选中）
    batch_size = real_features.size(0)
    cond_wrong_logits = netD.COND_DNET(real_features[:(batch_size - 1)], conditions[1:batch_size])
    cond_wrong_errD = nn.BCELoss()(cond_wrong_logits, fake_labels[1:batch_size])

    # 这里额外计算了一组不包含conditions（也就是不包含句子编码结果，只看图像本身）的判别
    # 目的应该是判别图像本身是否合理；同时把计算得到的loss和前面的考虑条件的取一个平均值
    if netD.UNCOND_DNET is not None:
        real_logits = netD.UNCOND_DNET(real_features)
        fake_logits = netD.UNCOND_DNET(fake_features)
        real_errD = nn.BCELoss()(real_logits, real_labels)
        fake_errD = nn.BCELoss()(fake_logits, fake_labels)
        errD = ((real_errD + cond_real_errD) / 2. +
                (fake_errD + cond_fake_errD + cond_wrong_errD) / 3.)
    else:
        errD = cond_real_errD + (cond_fake_errD + cond_wrong_errD) / 2.
    return errD


def generator_loss(netsD, image_encoder, fake_imgs, real_labels,
                   words_embs, sent_emb, match_labels,
                   cap_lens, class_ids):
    numDs = len(netsD)
    batch_size = real_labels.size(0)
    # logs = ''
    # Forward
    errG_total = 0
    # 因为判别器有多个，所以生成器的结果同样要经过多次判别
    for i in range(numDs):
        # 原理和前面相近，计算一下网络生成的虚假图像的特征并融合句子编码（sent_emb），然后理论上应该判别为真（real_labels）
        features = netsD[i](fake_imgs[i])
        cond_logits = netsD[i].COND_DNET(features, sent_emb)
        cond_errG = nn.BCELoss()(cond_logits, real_labels)

        # 如果有UNCOND_DNET，则计算不考虑句子编码的损失并累加
        if netsD[i].UNCOND_DNET is not None:
            logits = netsD[i].UNCOND_DNET(features)
            errG = nn.BCELoss()(logits, real_labels)
            g_loss = errG + cond_errG
        else:
            g_loss = cond_errG

        errG_total += g_loss

        # Ranking loss
        if i == (numDs - 1):
            # words_features: batch_size x nef x 17 x 17
            # sent_code: batch_size x nef
            # 这里的image_encoder就是预训练的图像编码器，这里要生成的图像进行编码，用于计算words_loss
            region_features, cnn_code = image_encoder(fake_imgs[i])
            # 这里的用于计算words_loss描述了单词编码和图像之间的相似程度
            w_loss0, w_loss1, _ = words_loss(region_features, words_embs,
                                             match_labels, cap_lens,
                                             class_ids, batch_size)
            w_loss = (w_loss0 + w_loss1) * cfg.TRAIN.SMOOTH.LAMBDA
            # err_words = err_words + w_loss.item()
            # 这里采取了类似的措施，计算句子编码和图像之间的相似度
            s_loss0, s_loss1 = sent_loss(cnn_code, sent_emb,
                                         match_labels, class_ids, batch_size)
            s_loss = (s_loss0 + s_loss1) * cfg.TRAIN.SMOOTH.LAMBDA
            # err_sent = err_sent + s_loss.item()
            # 单词编码和句子编码的相似度融合，作为最终的相似度计算结果
            # 可以看出，这里并没有引入生成结果和图像之间的直接相似度（例如MSE或者MAE）的计算；这也是生成类任务常见的情景
            errG_total += (w_loss + s_loss)

    return errG_total


##################################################################
def KL_loss(mu, logvar):
    # -0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD_element = mu.pow(2).add_(logvar.exp()).mul_(-1).add_(1).add_(logvar)
    KLD = torch.mean(KLD_element).mul_(-0.5)
    return KLD


def regression_loss(input, target):
    if cfg.TRAIN.REG_LOSS_TYPE == "MSE":
        loss_func = nn.MSELoss()
    elif cfg.TRAIN.REG_LOSS_TYPE == "MAE":
        loss_func = nn.L1Loss()

    loss_value = loss_func(input, target)
    return loss_value
