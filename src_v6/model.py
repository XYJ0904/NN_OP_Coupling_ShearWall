import torch
import torch.nn as nn
import torch.nn.parallel
from torch.autograd import Variable
from torchvision import models
import torch.utils.model_zoo as model_zoo
import torch.nn.functional as F

from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from miscc.config import cfg
from GlobalAttention import GlobalAttentionGeneral as ATT_NET


class GLU(nn.Module):
    def __init__(self):
        super(GLU, self).__init__()

    # 部分用于计算线性变换，另一部分用于计算门控信号
    # 可以看作是一个控制信息流动的“门”（会导致维度减半）
    def forward(self, x):
        nc = x.size(1)
        assert nc % 2 == 0, 'channels dont divide 2!'
        nc = int(nc/2)
        return x[:, :nc] * torch.sigmoid(x[:, nc:])


def conv1x1(in_planes, out_planes, bias=False):
    "1x1 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=1,
                     padding=0, bias=bias)


def conv3x3(in_planes, out_planes):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=1,
                     padding=1, bias=False)


# Upsale the spatial size by a factor of 2
def upBlock(in_planes, out_planes):
    block = nn.Sequential(
        nn.Upsample(scale_factor=2, mode='nearest'),
        conv3x3(in_planes, out_planes * 2),
        nn.BatchNorm2d(out_planes * 2),
        GLU())
    return block


# Keep the spatial size
def Block3x3_relu(in_planes, out_planes):
    block = nn.Sequential(
        conv3x3(in_planes, out_planes * 2),
        nn.BatchNorm2d(out_planes * 2),
        GLU())
    return block


class ResBlock(nn.Module):
    def __init__(self, channel_num):
        super(ResBlock, self).__init__()
        self.block = nn.Sequential(
            conv3x3(channel_num, channel_num * 2),
            nn.BatchNorm2d(channel_num * 2),
            GLU(),
            conv3x3(channel_num, channel_num),
            nn.BatchNorm2d(channel_num))

    def forward(self, x):
        residual = x
        out = self.block(x)
        out += residual
        return out


# ############## Text2Image Encoder-Decoder #######
class RNN_ENCODER(nn.Module):
    def __init__(self, ninput, drop_prob=0.0,
                 nhidden=128, nlayers=1, bidirectional=False):
        super(RNN_ENCODER, self).__init__()
        # self.n_steps = cfg.TEXT.SEQ_LENGTH
        # self.ntoken = ntoken  # size of the dictionary
        self.ninput = ninput  # size of each embedding vector
        self.drop_prob = drop_prob  # probability of an element to be zeroed
        self.nlayers = nlayers  # Number of recurrent layers
        self.bidirectional = bidirectional
        self.rnn_type = cfg.RNN_TYPE
        if bidirectional:
            self.num_directions = 2
        else:
            self.num_directions = 1
        # number of features in the hidden state
        self.nhidden = nhidden // self.num_directions

        self.define_module()
        self.init_weights()

    def define_module(self):
        # self.encoder = nn.Embedding(self.ntoken, self.ninput)
        self.encoder = nn.Linear(cfg.input_dim, self.ninput)

        if self.nlayers == 1:
            self.drop_prob = 0.0
        elif self.nlayers >= 2:
            pass

        self.drop = nn.Dropout(self.drop_prob)

        if self.rnn_type == 'LSTM':
            # dropout: If non-zero, introduces a dropout layer on
            # the outputs of each RNN layer except the last layer
            self.rnn = nn.LSTM(self.ninput, self.nhidden,
                               self.nlayers, batch_first=True,
                               dropout=self.drop_prob,
                               bidirectional=self.bidirectional)
        elif self.rnn_type == 'GRU':
            self.rnn = nn.GRU(self.ninput, self.nhidden,
                              self.nlayers, batch_first=True,
                              dropout=self.drop_prob,
                              bidirectional=self.bidirectional)
        else:
            raise NotImplementedError

    def init_weights(self):
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        # Do not need to initialize RNN parameters, which have been initialized
        # http://pytorch.org/docs/master/_modules/torch/nn/modules/rnn.html#LSTM
        # self.decoder.weight.data.uniform_(-initrange, initrange)
        # self.decoder.bias.data.fill_(0)

    def init_hidden(self, bsz):
        weight = next(self.parameters()).data
        if self.rnn_type == 'LSTM':
            return (Variable(weight.new(self.nlayers * self.num_directions,
                                        bsz, self.nhidden).zero_()),
                    Variable(weight.new(self.nlayers * self.num_directions,
                                        bsz, self.nhidden).zero_()))
        else:
            return Variable(weight.new(self.nlayers * self.num_directions,
                                       bsz, self.nhidden).zero_())

    def forward(self, captions, hidden, mask=None):
        # 输入是一个batch x n_steps的时间序列，并将其映射到指定维度 batch x n_steps x ninput
        emb = self.drop(self.encoder(captions))

        output, hidden = self.rnn(emb, hidden)
        # --> batch x hidden_size*num_directions x seq_len
        words_emb = output.transpose(1, 2)
        # --> batch x num_directions*hidden_size
        if self.rnn_type == 'LSTM':
            sent_emb = hidden[0].transpose(0, 1).contiguous()
        else:
            sent_emb = hidden.transpose(0, 1).contiguous()
        sent_emb = sent_emb.view(-1, self.nhidden * self.num_directions)
        return words_emb, sent_emb


class CNN_ENCODER(nn.Module):
    def __init__(self, nef):
        super(CNN_ENCODER, self).__init__()
        if cfg.TRAIN.FLAG:
            self.nef = nef
        else:
            self.nef = 256  # define a uniform ranker

        # weights = Inception_V3_Weights.IMAGENET1K_V1  # 或者 Inception_V3_Weights.DEFAULT
        # model = models.inception_v3(weights=weights)
        # model = models.inception_v3(pretrained=True)
        model = models.inception_v3(init_weights=False)

        url = 'https://download.pytorch.org/models/inception_v3_google-1a9a5a14.pth'
        model.load_state_dict(model_zoo.load_url(url))
        for param in model.parameters():
            param.requires_grad = False
        print('Load pretrained model from ', url)

        print("\n", "*" * 100, "\n")

        self.define_module(model)
        self.init_trainable_weights()

    def define_module(self, model):
        self.Conv2d_1a_3x3 = model.Conv2d_1a_3x3
        self.Conv2d_2a_3x3 = model.Conv2d_2a_3x3
        self.Conv2d_2b_3x3 = model.Conv2d_2b_3x3
        self.Conv2d_3b_1x1 = model.Conv2d_3b_1x1
        self.Conv2d_4a_3x3 = model.Conv2d_4a_3x3
        self.Mixed_5b = model.Mixed_5b
        self.Mixed_5c = model.Mixed_5c
        self.Mixed_5d = model.Mixed_5d
        self.Mixed_6a = model.Mixed_6a
        self.Mixed_6b = model.Mixed_6b
        self.Mixed_6c = model.Mixed_6c
        self.Mixed_6d = model.Mixed_6d
        self.Mixed_6e = model.Mixed_6e
        self.Mixed_7a = model.Mixed_7a
        self.Mixed_7b = model.Mixed_7b
        self.Mixed_7c = model.Mixed_7c

        self.emb_features = conv1x1(768, self.nef)
        self.emb_cnn_code = nn.Linear(2048, self.nef)

    def init_trainable_weights(self):
        initrange = 0.1
        self.emb_features.weight.data.uniform_(-initrange, initrange)
        self.emb_cnn_code.weight.data.uniform_(-initrange, initrange)

    def forward(self, x):
        features = None
        # --> fixed-size input: batch x 3 x 299 x 299
        x = nn.Upsample(size=(299, 299), mode='bilinear')(x)
        # 299 x 299 x 3
        x = self.Conv2d_1a_3x3(x)
        # 149 x 149 x 32
        x = self.Conv2d_2a_3x3(x)
        # 147 x 147 x 32
        x = self.Conv2d_2b_3x3(x)
        # 147 x 147 x 64
        x = F.max_pool2d(x, kernel_size=3, stride=2)
        # 73 x 73 x 64
        x = self.Conv2d_3b_1x1(x)
        # 73 x 73 x 80
        x = self.Conv2d_4a_3x3(x)
        # 71 x 71 x 192

        x = F.max_pool2d(x, kernel_size=3, stride=2)
        # 35 x 35 x 192
        x = self.Mixed_5b(x)
        # 35 x 35 x 256
        x = self.Mixed_5c(x)
        # 35 x 35 x 288
        x = self.Mixed_5d(x)
        # 35 x 35 x 288

        x = self.Mixed_6a(x)
        # 17 x 17 x 768
        x = self.Mixed_6b(x)
        # 17 x 17 x 768
        x = self.Mixed_6c(x)
        # 17 x 17 x 768
        x = self.Mixed_6d(x)
        # 17 x 17 x 768
        x = self.Mixed_6e(x)
        # 17 x 17 x 768

        # image region features（把中间某个尺度的特征输出拿了出来，在进一步缩小尺度和降维之前？）
        features = x
        # 17 x 17 x 768

        x = self.Mixed_7a(x)
        # 8 x 8 x 1280
        x = self.Mixed_7b(x)
        # 8 x 8 x 2048
        x = self.Mixed_7c(x)
        # 8 x 8 x 2048
        x = F.avg_pool2d(x, kernel_size=8)
        # 1 x 1 x 2048
        # x = F.dropout(x, training=self.training)
        # 1 x 1 x 2048
        x = x.view(x.size(0), -1)
        # 2048

        # global image features
        # 将维度变成了embedding_dim所需要的维度；注意空间尺寸已经被削减为了1
        cnn_code = self.emb_cnn_code(x)
        # 512
        if features is not None:
            # 将维度变成了embedding_dim所需要的维度；注意这个是有空间尺寸的
            features = self.emb_features(features)
        return features, cnn_code


# ############## G networks ###################
class INIT_STAGE_G(nn.Module):
    # 注意这里输入的ngf实际上是 ngf * 16，降维之后刚好等于ngf
    def __init__(self, ngf, ncf, initial_enlarge_layer):
        super(INIT_STAGE_G, self).__init__()
        self.gf_dim = ngf
        # self.in_dim = cfg.GAN.Z_DIM + ncf  # ncf = cfg.TEXT.EMBEDDING_DIM
        self.initial_enlarge_layer = initial_enlarge_layer
        self.define_module()

    def define_module(self):
        # nz, ngf = self.in_dim, self.gf_dim
        ngf = self.gf_dim
        # 之所以是4 * 4 * 2是因为模型的维度在GLU的处理中减半了
        # 4 * 4展开之后是平面维度，* 2 是给GLU用的
        self.initial_dim = cfg.TREE.BASE_SIZE
        for i in range(self.initial_enlarge_layer):
            self.initial_dim = self.initial_dim / 2

        self.initial_dim = int(round(self.initial_dim, 0))

        # self.fc = nn.Sequential(
        #     nn.Linear(nz, ngf * self.initial_dim * self.initial_dim * 2, bias=False),
        #     nn.BatchNorm1d(ngf * self.initial_dim * self.initial_dim * 2),
        #     GLU())
        self.fc = nn.Sequential(
            nn.Linear(cfg.GAN.Z_DIM, ngf * self.initial_dim * self.initial_dim * 2, bias=False),
            nn.BatchNorm1d(ngf * self.initial_dim * self.initial_dim * 2),
            GLU())

        self.upsample1 = upBlock(ngf, ngf // 2)
        self.upsample2 = upBlock(ngf // 2, ngf // 4)
        self.upsample3 = upBlock(ngf // 4, ngf // 8)
        self.upsample4 = upBlock(ngf // 8, ngf // 16)

    def forward(self, z_code):
        """
        :param z_code: batch x cfg.GAN.Z_DIM（随机噪声）
        :param c_code: batch x cfg.TEXT.EMBEDDING_DIM（文本编码 + 考虑随机性的噪声影响结果）
        :return: batch x ngf/16 x 64 x 64
        """
        # 这里把c_code, z_code 拼接成一个张量，随后通过self.fc 调整维度为cfg.GAN.GF_DIM（也即ngf)
        # c_z_code = torch.cat((c_code, z_code), 1)
        c_z_code = z_code
        # state size ngf x 4 x 4（注意这里的4其实是self.initial_dim = 4 才成立）
        out_code = self.fc(c_z_code)
        out_code = out_code.view(-1, self.gf_dim, self.initial_dim, self.initial_dim)

        # state size ngf/2 x 8 x 8
        out_code = self.upsample1(out_code)
        if self.initial_enlarge_layer >= 2:
            # state size ngf/4 x 16 x 16
            out_code = self.upsample2(out_code)
        if self.initial_enlarge_layer >= 3:
            # state size ngf/8 x 32 x 32
            out_code = self.upsample3(out_code)
        if self.initial_enlarge_layer >= 4:
            # state size ngf/16 x 64 x 64
            out_code = self.upsample4(out_code)
        if self.initial_enlarge_layer >= 5:
            print("Initial Enlarge Layer of Level 5 or Higher is Not Defined")
            raise AssertionError

        return out_code


class NEXT_STAGE_G(nn.Module):
    def __init__(self, ngf, nef, ncf):
        super(NEXT_STAGE_G, self).__init__()
        self.gf_dim = ngf
        self.ef_dim = nef
        self.cf_dim = ncf
        self.num_residual = cfg.GAN.R_NUM
        self.define_module()

    def _make_layer(self, block, channel_num):
        layers = []
        for i in range(cfg.GAN.R_NUM):
            layers.append(block(channel_num))
        return nn.Sequential(*layers)

    def define_module(self):
        # self.ef_dim 其实是 cfg.TEXT.EMBEDDING_DIM
        # ngf 是 cfg.GAN.GF_DIM
        ngf = self.gf_dim
        self.att = ATT_NET(ngf, self.ef_dim)
        self.residual = self._make_layer(ResBlock, ngf * 2)
        self.upsample = upBlock(ngf * 2, ngf)

    def forward(self, h_code, word_embs, mask):
        """
            h_code(query):  batch x ngf(self.gf_dim，也即cfg.GAN.GF_DIM) x ih x iw (queryL=ih x iw)
            word_embs(context): batch x cfg.TEXT.EMBEDDING_DIM x sourceL (sourceL=seq_len)
            att1: batch x sourceL x queryL
        """
        self.att.applyMask(mask)
        c_code, att = self.att(h_code, word_embs)
        h_c_code = torch.cat((h_code, c_code), 1)

        # residual模块每次都将组合后的张量的通道数改为ngf * 2，而upsample则将尺寸增大同时通道数再降回ngf，所以从始至终ngf不变
        # 这样操作方便生成图像，可以用统一尺寸的一组参数进行处理
        out_code = self.residual(h_c_code)
        out_code = self.upsample(out_code)

        return out_code, att


class GET_IMAGE_G(nn.Module):
    def __init__(self, ngf):
        super(GET_IMAGE_G, self).__init__()
        self.gf_dim = ngf
        self.img = nn.Sequential(
            conv3x3(ngf, cfg.output_dim),
            nn.Tanh()
        )

    def forward(self, h_code):
        out_img = self.img(h_code)
        # out_img = out_img.repeat(1, 3, 1, 1)
        return out_img


class G_NET(nn.Module):
    def __init__(self):
        super(G_NET, self).__init__()

        # ngf 代表了初始图像的通道数目，也是生成器网络中的特征维度（Generator Feature Dimension）
        ngf = cfg.GAN.GF_DIM
        # nef 代表文本嵌入的维度（Embedding Dimension），即序列首先被转换成紧凑向量表示的维度
        nef = cfg.TEXT.EMBEDDING_DIM
        # 是条件维度（Condition Dimension），指的是在生成过程中用于条件化生成器和判别器的特征的维度
        # 在文本到图像的生成中，条件化通常指的是使用序列输入作为条件，来指导图像生成过程
        ncf = cfg.GAN.CONDITION_DIM
        # self.ca_net = CA_NET()

        if cfg.TREE.BRANCH_NUM > 0:
            self.h_net1 = INIT_STAGE_G(ngf * (2 ** (cfg.TREE.initial_enlarge_layer)), ncf, cfg.TREE.initial_enlarge_layer)
            self.img_net1 = GET_IMAGE_G(ngf)
        # gf x 64 x 64
        if cfg.TREE.BRANCH_NUM > 1:
            self.h_net2 = NEXT_STAGE_G(ngf, nef, ncf)
            self.img_net2 = GET_IMAGE_G(ngf)
        if cfg.TREE.BRANCH_NUM > 2:
            self.h_net3 = NEXT_STAGE_G(ngf, nef, ncf)
            self.img_net3 = GET_IMAGE_G(ngf)
        if cfg.TREE.BRANCH_NUM > 3:
            self.h_net4 = NEXT_STAGE_G(ngf, nef, ncf)
            self.img_net4 = GET_IMAGE_G(ngf)

    def forward(self, z_code, sent_emb, word_embs, mask):
        """
            :param z_code: batch x cfg.GAN.Z_DIM（这个z_code对应的是输入的noise）
            :param sent_emb: batch x cfg.TEXT.EMBEDDING_DIM
            :param word_embs: batch x cfg.TEXT.EMBEDDING_DIM x seq_len
            :param mask: batch x seq_len
            :return:
        """
        fake_imgs = []
        att_maps = []

        # c_code 是变分自编码器VAE给出的带有随机性的结果，在生成任务中很重要，但回归任务中的角色有待考量
        # c_code, mu, logvar = self.ca_net(sent_emb)

        # 这里的输出是几个list，包括fake_imgs（生成图像）、att_maps（注意力图）和sent_emb的编码结果mu, logvar
        # 前两个输出都是list，从低分辨率到高分辨率，长度等于netD的数目或者tree branch number

        # h_net1给出的输出结果是 B * base_size * base_size * ngf，而img_net1只是做了一次降维到输出维度的操作，给出最终结果
        if cfg.TREE.BRANCH_NUM > 0:
            h_code1 = self.h_net1(z_code)
            fake_img1 = self.img_net1(h_code1)
            fake_imgs.append(fake_img1)

        # 这几个分支里c_code实际上就是没有用到
        if cfg.TREE.BRANCH_NUM > 1:
            h_code2, att1 = \
                self.h_net2(h_code1, word_embs, mask)
            fake_img2 = self.img_net2(h_code2)
            fake_imgs.append(fake_img2)
            if att1 is not None:
                att_maps.append(att1)
        if cfg.TREE.BRANCH_NUM > 2:
            h_code3, att2 = \
                self.h_net3(h_code2, word_embs, mask)
            fake_img3 = self.img_net3(h_code3)
            fake_imgs.append(fake_img3)
            if att2 is not None:
                att_maps.append(att2)
        if cfg.TREE.BRANCH_NUM > 3:
            h_code4, att3 = \
                self.h_net4(h_code3, word_embs, mask)
            fake_img4 = self.img_net3(h_code4)
            fake_imgs.append(fake_img4)
            if att3 is not None:
                att_maps.append(att3)

        return fake_imgs, att_maps


# ############## D networks ##########################
def Block3x3_leakRelu(in_planes, out_planes):
    block = nn.Sequential(
        conv3x3(in_planes, out_planes),
        nn.BatchNorm2d(out_planes),
        nn.LeakyReLU(0.2, inplace=True)
    )
    return block


# Downsale the spatial size by a factor of 2
def downBlock(in_planes, out_planes):
    block = nn.Sequential(
        nn.Conv2d(in_planes, out_planes, 4, 2, 1, bias=False),
        nn.BatchNorm2d(out_planes),
        nn.LeakyReLU(0.2, inplace=True)
    )
    return block


# Downsale the spatial size by a factor of 16
def encode_image_by_16times(ndf):
    encode_img = nn.Sequential(
        # --> state size. ndf x in_size/2 x in_size/2
        nn.Conv2d(3, ndf, 4, 2, 1, bias=False),
        nn.LeakyReLU(0.2, inplace=True),
        # --> state size 2ndf x x in_size/4 x in_size/4
        nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
        nn.BatchNorm2d(ndf * 2),
        nn.LeakyReLU(0.2, inplace=True),
        # --> state size 4ndf x in_size/8 x in_size/8
        nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
        nn.BatchNorm2d(ndf * 4),
        nn.LeakyReLU(0.2, inplace=True),
        # --> state size 8ndf x in_size/16 x in_size/16
        nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
        nn.BatchNorm2d(ndf * 8),
        nn.LeakyReLU(0.2, inplace=True)
    )
    return encode_img


# 这个函数的功能是将输入的h_code和c_code融合，并通过卷积保持维度不变（为ndf * 8，即通道上采样之后的维度）
# 当然，如果没有bcondition，也即没有c_code，那就什么操作都不用做
# 最后在将联合卷积（或者没操作）的样本的维度降维为1，并大幅缩减其最终尺寸、做sigmoid操作，以输出最终判别结果
class D_GET_LOGITS(nn.Module):
    def __init__(self, ndf, nef, bcondition=False):
        super(D_GET_LOGITS, self).__init__()
        self.df_dim = ndf
        self.ef_dim = nef
        self.bcondition = bcondition
        if self.bcondition:
            self.jointConv = Block3x3_leakRelu(ndf * 8 + nef, ndf * 8)

        # self.outlogits = nn.Sequential(
        #     nn.Conv2d(ndf * 8, 1, kernel_size=4, stride=4),
        #     nn.Sigmoid())
        self.outlogits = nn.Sequential(
            nn.Conv2d(ndf * 8, 1, kernel_size=4, stride=4))

    def forward(self, h_code, c_code=None):
        if self.bcondition and c_code is not None:
            # conditioning output
            c_code = c_code.view(-1, self.ef_dim, 1, 1)
            c_code = c_code.repeat(1, 1, 4, 4)
            # state size (ngf+egf) x 4 x 4
            h_c_code = torch.cat((h_code, c_code), 1)
            # state size ngf x in_size x in_size
            h_c_code = self.jointConv(h_c_code)
        else:
            h_c_code = h_code

        output = torch.sigmoid(self.outlogits(h_c_code))
        return output.view(-1)


# For 64 x 64 images
class D_NET64(nn.Module):
    def __init__(self, b_jcu=True):
        super(D_NET64, self).__init__()
        ndf = cfg.GAN.DF_DIM
        nef = cfg.TEXT.EMBEDDING_DIM
        # 这个函数的功能是将模型空间维度下采样16倍，但通道维度上采样8倍
        self.img_code_s16 = encode_image_by_16times(ndf)
        if b_jcu:
            self.UNCOND_DNET = D_GET_LOGITS(ndf, nef, bcondition=False)
        else:
            self.UNCOND_DNET = None
        self.COND_DNET = D_GET_LOGITS(ndf, nef, bcondition=True)

    def forward(self, x_var):
        x_code4 = self.img_code_s16(x_var)  # 4 x 4 x 8df
        return x_code4


# For 128 x 128 images
class D_NET128(nn.Module):
    def __init__(self, b_jcu=True):
        super(D_NET128, self).__init__()
        ndf = cfg.GAN.DF_DIM
        nef = cfg.TEXT.EMBEDDING_DIM
        self.img_code_s16 = encode_image_by_16times(ndf)
        self.img_code_s32 = downBlock(ndf * 8, ndf * 16)
        self.img_code_s32_1 = Block3x3_leakRelu(ndf * 16, ndf * 8)
        #
        if b_jcu:
            self.UNCOND_DNET = D_GET_LOGITS(ndf, nef, bcondition=False)
        else:
            self.UNCOND_DNET = None
        self.COND_DNET = D_GET_LOGITS(ndf, nef, bcondition=True)

    def forward(self, x_var):
        x_code8 = self.img_code_s16(x_var)   # 8 x 8 x 8df
        x_code4 = self.img_code_s32(x_code8)   # 4 x 4 x 16df
        x_code4 = self.img_code_s32_1(x_code4)  # 4 x 4 x 8df
        return x_code4


# For 256 x 256 images
class D_NET256(nn.Module):
    def __init__(self, b_jcu=True):
        super(D_NET256, self).__init__()
        ndf = cfg.GAN.DF_DIM
        nef = cfg.TEXT.EMBEDDING_DIM
        self.img_code_s16 = encode_image_by_16times(ndf)
        self.img_code_s32 = downBlock(ndf * 8, ndf * 16)
        self.img_code_s64 = downBlock(ndf * 16, ndf * 32)
        self.img_code_s64_1 = Block3x3_leakRelu(ndf * 32, ndf * 16)
        self.img_code_s64_2 = Block3x3_leakRelu(ndf * 16, ndf * 8)
        if b_jcu:
            self.UNCOND_DNET = D_GET_LOGITS(ndf, nef, bcondition=False)
        else:
            self.UNCOND_DNET = None
        self.COND_DNET = D_GET_LOGITS(ndf, nef, bcondition=True)

    def forward(self, x_var):
        x_code16 = self.img_code_s16(x_var)
        x_code8 = self.img_code_s32(x_code16)
        x_code4 = self.img_code_s64(x_code8)
        x_code4 = self.img_code_s64_1(x_code4)
        x_code4 = self.img_code_s64_2(x_code4)
        return x_code4
