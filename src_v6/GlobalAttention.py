"""
Global attention takes a matrix and a query metrix.
Based on each query vector q, it computes a parameterized convex combination of the matrix
based.
H_1 H_2 H_3 ... H_n
  q   q   q       q
    |  |   |       |
      \ |   |      /
              .....
          \   |  /
                  a
Constructs a unit mapping.
$$(H_1 + H_n, q) => (a)$$
Where H is of `batch x n x dim` and q is of `batch x dim`.

References:
https://github.com/OpenNMT/OpenNMT-py/tree/fc23dfef1ba2f258858b2765d24565266526dc76/onmt/modules
http://www.aclweb.org/anthology/D15-1166
"""

import torch
import torch.nn as nn


def conv1x1(in_planes, out_planes):
    "1x1 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=1,
                     padding=0, bias=False)


def func_attention(query, context, gamma1):
    """
    query: batch x ndf x queryL
    context: batch x ndf x ih x iw (sourceL=ihxiw)
    mask: batch_size x sourceL
    """
    batch_size, queryL = query.size(0), query.size(2)
    ih, iw = context.size(2), context.size(3)
    sourceL = ih * iw

    # --> batch x sourceL x ndf
    context = context.view(batch_size, -1, sourceL)
    contextT = torch.transpose(context, 1, 2).contiguous()

    # Get attention
    # (batch x sourceL x ndf)(batch x ndf x queryL)
    # -->batch x sourceL x queryL
    attn = torch.bmm(contextT, query) # Eq. (7) in AttnGAN paper
    # --> batch*sourceL x queryL
    attn = attn.view(batch_size * sourceL, queryL)
    attn = nn.Softmax(dim=1)(attn)  # Eq. (8)

    # --> batch x sourceL x queryL
    attn = attn.view(batch_size, sourceL, queryL)
    # --> batch*queryL x sourceL
    attn = torch.transpose(attn, 1, 2).contiguous()
    attn = attn.view(batch_size * queryL, sourceL)
    #  Eq. (9)
    attn = attn * gamma1
    attn = nn.Softmax(dim=1)(attn)
    attn = attn.view(batch_size, queryL, sourceL)
    # --> batch x sourceL x queryL
    attnT = torch.transpose(attn, 1, 2).contiguous()

    # (batch x ndf x sourceL)(batch x sourceL x queryL)
    # --> batch x ndf x queryL
    weightedContext = torch.bmm(context, attnT)

    return weightedContext, attn.view(batch_size, -1, ih, iw)


class GlobalAttentionGeneral(nn.Module):
    def __init__(self, idf, cdf):

        # cdf 其实是 cfg.TEXT.EMBEDDING_DIM，序列映射后的维度
        # idf 是 cfg.GAN.GF_DIM，即生成器特征维度
        super(GlobalAttentionGeneral, self).__init__()
        self.conv_context = conv1x1(cdf, idf)
        self.sm = nn.Softmax(dim=1)
        self.mask = None

    def applyMask(self, mask):
        self.mask = mask  # batch x sourceL

    def forward(self, input, context):
        """
            input: batch x ngf(self.gf_dim，也即cfg.GAN.GF_DIM) x ih x iw (queryL= ih x iw)
            context: batch x cfg.TEXT.EMBEDDING_DIM x sourceL (sourceL=seq_len)
        """
        ih, iw = input.size(2), input.size(3)
        queryL = ih * iw
        batch_size, sourceL = context.size(0), context.size(2)

        # --> batch x queryL x cfg.GAN.GF_DIM
        target = input.view(batch_size, -1, queryL)
        targetT = torch.transpose(target, 1, 2).contiguous()
        # batch x cfg.TEXT.EMBEDDING_DIM x sourceL --> batch x cfg.TEXT.EMBEDDING_DIM x sourceL x 1
        # 这里主要就是做一次维度调整，让两者统一方便后面计算（矩阵乘法运算要求某个维度一致）
        sourceT = context.unsqueeze(3)
        # --> batch x cfg.GAN.GF_DIM x sourceL
        sourceT = self.conv_context(sourceT).squeeze(3)

        # Get attention
        # (batch x queryL x cfg.GAN.GF_DIM) (batch x cfg.GAN.GF_DIM x sourceL)
        # -->batch x queryL x sourceL
        attn = torch.bmm(targetT, sourceT)
        # --> batch * queryL x sourceL
        attn = attn.view(batch_size * queryL, sourceL)

        if self.mask is not None:
            # batch_size x sourceL --> batch_size*queryL x sourceL
            mask = self.mask.squeeze().repeat(queryL, 1)
            attn.data.masked_fill_(mask.data, -float('inf'))

        attn = self.sm(attn)  # Eq. (2)
        # --> batch x queryL x sourceL
        attn = attn.view(batch_size, queryL, sourceL)
        # --> batch x sourceL x queryL
        attn = torch.transpose(attn, 1, 2).contiguous()

        # (batch x cfg.GAN.GF_DIM x sourceL) (batch x sourceL x queryL)
        # --> batch x cfg.GAN.GF_DIM x queryL
        weightedContext = torch.bmm(sourceT, attn)
        weightedContext = weightedContext.view(batch_size, -1, ih, iw)
        attn = attn.view(batch_size, -1, ih, iw)

        return weightedContext, attn
