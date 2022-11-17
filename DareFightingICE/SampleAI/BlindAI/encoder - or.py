# 导入必备的工具包
import torch

# 预定义的网络层torch.nn, 工具开发者已经帮助我们开发好的一些常用层, 
# 比如，卷积层, lstm层, embedding层等, 不需要我们再重新造轮子.
# 数学计算工具包
import math
# torch中变量封装函数Variable.
from torch.autograd import Variable
#画图包（可不用）
import matplotlib.pyplot as plt
import numpy as np
# 用于深度拷贝的copy工具包
import copy
# nn.functional工具包装载了网络层中那些只进行计算, 而没有参数的层
import torch.nn.functional as F
#from torch.nn.functional import log_softmax, pad
import os
from os.path import exists
import math
import time
from torch.optim.lr_scheduler import LambdaLR
import pandas as pd
import altair as alt
from torchtext.data.functional import to_map_style_dataset
from torch.utils.data import DataLoader
from torchtext.vocab import build_vocab_from_iterator
import torchtext.datasets as datasets
import spacy
import GPUtil
import warnings
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP

from abc import ABC, abstractmethod
from torch import nn
import torchaudio

class BaseEncoder(nn.Module, ABC):
    def __init__(self, sampling_rate=48000, fps=60, frame_skip=4):
        super(BaseEncoder, self).__init__()
        self.sampling_rate = sampling_rate
        self.FPS = fps
        self.frame_skip = frame_skip

    def forward(self, x):
        # left side
        left = x[:, :, 0]
        left = self.encode_single_channel(left)
        # right side
        right = x[:, :, 1]
        right = self.encode_single_channel(right)
        return torch.cat((left, right), dim=1)

    @abstractmethod
    def encode_single_channel(self, data):
        pass

class SampleEncoder(BaseEncoder):
    def encode_single_channel(self, data):
        return data

class RawEncoder(BaseEncoder):
    def __init__(self, sampling_rate=48000, fps=60, frame_skip=4):
        super(RawEncoder, self).__init__(sampling_rate, fps, frame_skip)
        self.num_to_subsample = 8
        self.num_samples = (self.sampling_rate / self.FPS) * self.frame_skip
        assert int(self.num_samples) == self.num_samples

        # Encoder (small 1D conv)
        self.pool = torch.nn.MaxPool1d(2)
        self.conv1 = torch.nn.Conv1d(1, 16, kernel_size=16, stride=8)
        self.conv2 = torch.nn.Conv1d(16, 32, kernel_size=16, stride=8)

    def encode_single_channel(self, data):
        """Shape of x: [batch_size, num_samples]"""
        # Subsample
        x = data[:, ::self.num_to_subsample]

        # Add channel dimension
        x = x[:, None, :]
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        if x.shape[2] >= 24:
            x = self.conv2(x)
            x = self.pool(x)
        return x


class MelSpecEncoder(BaseEncoder):
    def __init__(self, sampling_rate=48000, fps=60, frame_skip=4):
        super(MelSpecEncoder, self).__init__(sampling_rate, fps, frame_skip)
        self.window_size = int(self.sampling_rate * 0.025)
        self.hop_size = int(self.sampling_rate * 0.01)
        self.n_fft = int(self.sampling_rate * 0.025)
        self.n_mels = 80

        self.mel_spectrogram = torchaudio.transforms.MelSpectrogram(
            sample_rate=self.sampling_rate,
            n_mels=80,
            n_fft=self.n_fft,
            win_length=self.window_size,
            hop_length=self.hop_size,
            f_min=20,
            f_max=7600,
        )

        # Encoder
        self.conv1 = torch.nn.Conv2d(1, 16, 3, padding=1)
        self.conv2 = torch.nn.Conv2d(16, 32, 3, padding=1)
        self.pool = torch.nn.MaxPool2d(2, 2)

    def encode_single_channel(self, data):
        x = torch.log(self.mel_spectrogram(data) + 1e-5)
        x = torch.reshape(x, (x.shape[0], 1, x.shape[1], x.shape[2]))
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        if x.shape[-1] >= 2:
            x = self.pool(x)
        return x


class FFTEncoder(BaseEncoder):
    def __init__(self, sampling_rate=48000, fps=60, frame_skip=4):
        super(FFTEncoder, self).__init__(sampling_rate, fps, frame_skip)
        self.num_to_subsample = 8
        self.num_samples = (self.sampling_rate / self.FPS) * self.frame_skip
        self.num_frequencies = self.num_samples / 2
        assert int(self.num_samples) == self.num_samples
        self.num_samples = int(self.num_samples)
        self.num_frequencies = int(self.num_frequencies)

        self.hamming_window = torch.hamming_window(self.num_samples)

        # Subsampler
        self.pool = torch.nn.MaxPool1d(self.num_to_subsample)

        # Encoder (small MLP)
        self.linear1 = torch.nn.Linear(int(self.num_frequencies / self.num_to_subsample), 256)
        self.linear2 = torch.nn.Linear(256, 256)

    def _torch_1d_fft_magnitude(self, x):
        """Perform 1D FFT on x with shape (batch_size, num_samples), and return magnitudes"""
        # Apply hamming window
        if x.device != self.hamming_window.device:
            self.hamming_window = self.hamming_window.to(x.device)
        x = x * self.hamming_window
        # Add zero imaginery parts
        x = torch.stack((x, torch.zeros_like(x)), dim=-1)
        #将input c作为复数张量返回
        c = torch.view_as_complex(x)
        ffts = torch.fft.fft(c)
        ffts = torch.view_as_real(ffts)
        # Remove mirrored part
        ffts = ffts[:, :(ffts.shape[1] // 2), :]
        # To magnitudes
        mags = torch.sqrt(ffts[..., 0] ** 2 + ffts[..., 1] ** 2)
        return mags

    def encode_single_channel(self, data):
        """Shape of x: [batch_size, num_samples]"""
        mags = self._torch_1d_fft_magnitude(data)
        mags = torch.log(mags + 1e-5)

        # Add and remove "channel" dim...
        x = self.pool(mags[:, None, :])[:, 0, :]
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        return x


"""



Transformer_encoder_model_class&function



"""


# 定义Embeddings类来实现文本嵌入层，这里s说明代表两个一模一样的嵌入层, 他们共享参数.
# 该类继承nn.Module, 这样就有标准层的一些功能, 这里我们也可以理解为一种模式, 我们自己实现的所有层都会这样去写.
class Embeddings(nn.Module):
    def __init__(self, d_model, vocab):
        """类的初始化函数, 有两个参数, d_model: 指词嵌入的维度, vocab: 指词表的大小."""
        # 接着就是使用super的方式指明继承nn.Module的初始化函数, 我们自己实现的所有层都会这样去写.
        super(Embeddings, self).__init__()
        # 之后就是调用nn中的预定义层Embedding, 获得一个词嵌入对象self.lut
        self.lut = nn.Embedding(vocab, d_model)
        # 最后就是将d_model传入类中
        self.d_model = d_model

    def forward(self, x):
        """可以将其理解为该层的前向传播逻辑，所有层中都会有此函数
        当传给该类的实例化对象参数时, 自动调用该类函数
        参数x: 因为Embedding层是首层, 所以代表输入给模型的文本通过词汇映射后的张量"""

        # 将x传给self.lut并与根号下self.d_model相乘作为结果返回
        return self.lut(x) * math.sqrt(self.d_model)


# 定义位置编码器类, 我们同样把它看做一个层, 因此会继承nn.Module    
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout, max_len=5000):
        """位置编码器类的初始化函数, 共有三个参数, 分别是d_model: 词嵌入维度, 
        dropout: 置0比率, max_len: 每个句子的最大长度"""
        super(PositionalEncoding, self).__init__()

        # 实例化nn中预定义的Dropout层, 并将dropout传入其中, 获得对象self.dropout
        self.dropout = nn.Dropout(p=dropout)

        # 初始化一个位置编码矩阵, 它是一个0阵，矩阵的大小是max_len x d_model.
        pe = torch.zeros(max_len, d_model)

        # 初始化一个绝对位置矩阵, 在我们这里，词汇的绝对位置就是用它的索引去表示. 
        # 所以我们首先使用arange方法获得一个连续自然数向量，然后再使用unsqueeze方法拓展向量维度使其成为矩阵， 
        # 又因为参数传的是1，代表矩阵拓展的位置，会使向量变成一个max_len x 1 的矩阵， 
        position = torch.arange(0, max_len).unsqueeze(1)

        # 绝对位置矩阵初始化之后，接下来就是考虑如何将这些位置信息加入到位置编码矩阵中，
        # 最简单思路就是先将max_len x 1的绝对位置矩阵， 变换成max_len x d_model形状，然后覆盖原来的初始位置编码矩阵即可， 
        # 要做这种矩阵变换，就需要一个1xd_model形状的变换矩阵div_term，我们对这个变换矩阵的要求除了形状外，
        # 还希望它能够将自然数的绝对位置编码缩放成足够小的数字，有助于在之后的梯度下降过程中更快的收敛.  这样我们就可以开始初始化这个变换矩阵了.
        # 首先使用arange获得一个自然数矩阵， 但是细心的同学们会发现， 我们这里并没有按照预计的一样初始化一个1xd_model的矩阵， 
        # 而是有了一个跳跃，只初始化了一半即1xd_model/2 的矩阵。 为什么是一半呢，其实这里并不是真正意义上的初始化了一半的矩阵，
        # 我们可以把它看作是初始化了两次，而每次初始化的变换矩阵会做不同的处理，第一次初始化的变换矩阵分布在正弦波上， 第二次初始化的变换矩阵分布在余弦波上， 
        # 并把这两个矩阵分别填充在位置编码矩阵的偶数和奇数位置上，组成最终的位置编码矩阵.
        div_term = torch.exp(torch.arange(0, d_model, 2) *
                             -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        # 这样我们就得到了位置编码矩阵pe, pe现在还只是一个二维矩阵，要想和embedding的输出（一个三维张量）相加，
        # 就必须拓展一个维度，所以这里使用unsqueeze拓展维度.
        pe = pe.unsqueeze(0)

        # 最后把pe位置编码矩阵注册成模型的buffer，什么是buffer呢，
        # 我们把它认为是对模型效果有帮助的，但是却不是模型结构中超参数或者参数，不需要随着优化步骤进行更新的增益对象. 
        # 注册之后我们就可以在模型保存后重加载时和模型结构与参数一同被加载.
        self.register_buffer('pe', pe)

    def forward(self, x):
        """forward函数的参数是x, 表示文本序列的词嵌入表示"""
        # 在相加之前我们对pe做一些适配工作， 将这个三维张量的第二维也就是句子最大长度的那一维将切片到与输入的x的第二维相同即x.size(1)，
        # 因为我们默认max_len为5000一般来讲实在太大了，很难有一条句子包含5000个词汇，所以要进行与输入张量的适配. 
        # 最后使用Variable进行封装，使其与x的样式相同，但是它是不需要进行梯度求解的，因此把requires_grad设置成false.
        x = x + self.pe[:, : x.size(1)].requires_grad_(False)
        # 最后使用self.dropout对象进行'丢弃'操作, 并返回结果.
        return self.dropout(x)

# 我们使用一个类来实现多头注意力机制的处理
class MultiHeadedAttention(nn.Module):
    def __init__(self, head, d_model, dropout=0.1):
        """在类的初始化时, 会传入三个参数，head代表头数，d_model代表词嵌入的维度， 
        dropout代表进行dropout操作时置0比率，默认是0.1."""
        super(MultiHeadedAttention, self).__init__()

        # 在函数中，首先使用了一个测试中常用的assert语句，判断h是否能被d_model整除，
        # 这是因为我们之后要给每个头分配等量的词特征.也就是d_model/head个.
        assert d_model % head == 0

        # 得到每个头获得的分割词向量维度d_k
        self.d_k = d_model // head

        # 传入头数h
        self.head = head

        # 然后获得线性层对象，通过nn的Linear实例化，它的内部变换矩阵是d_model x d_model，然后使用clones函数克隆四个，
        # 为什么是四个呢，这是因为在多头注意力中，Q，K，V各需要一个，最后拼接的矩阵还需要一个，因此一共是四个.
        self.linears = clones(nn.Linear(d_model, d_model), 4)

        # self.attn为None，它代表最后得到的注意力张量，现在还没有结果所以为None.
        self.attn = None

        # 最后就是一个self.dropout对象，它通过nn中的Dropout实例化而来，置0比率为传进来的参数dropout.
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        """前向逻辑函数, 它的输入参数有四个，前三个就是注意力机制需要的Q, K, V，
        最后一个是注意力机制中可能需要的mask掩码张量，默认是None. """

        # 如果存在掩码张量mask
        if mask is not None:
            # 使用unsqueeze拓展维度
            mask = mask.unsqueeze(1)

        # 接着，我们获得一个batch_size的变量，他是query尺寸的第1个数字，代表有多少条样本.
        nbatches = query.size(0)

        # 之后就进入多头处理环节
        # 首先利用zip将输入QKV与三个线性层组到一起，然后使用for循环，将输入QKV分别传到线性层中，
        # 做完线性变换后，开始为每个头分割输入，这里使用view方法对线性变换的结果进行维度重塑，多加了一个维度h，代表头数，
        # 这样就意味着每个头可以获得一部分词特征组成的句子，其中的-1代表自适应维度，
        # 计算机会根据这种变换自动计算这里的值.然后对第二维和第三维进行转置操作，
        # 为了让代表句子长度维度和词向量维度能够相邻，这样注意力机制才能找到词义与句子位置的关系，
        # 从attention函数中可以看到，利用的是原始输入的倒数第一和第二维.这样我们就得到了每个头的输入.
        query, key, value = [
            lin(x).view(nbatches, -1, self.head, self.d_k).transpose(1, 2)
            for lin, x in zip(self.linears, (query, key, value))
        ]

        # 得到每个头的输入后，接下来就是将他们传入到attention中，
        # 这里直接调用我们之前实现的attention函数.同时也将mask和dropout传入其中.
        x, self.attn = attention(query, key, value, mask=mask, dropout=self.dropout)

        # 通过多头注意力计算后，我们就得到了每个头计算结果组成的4维张量，我们需要将其转换为输入的形状以方便后续的计算，
        # 因此这里开始进行第一步处理环节的逆操作，先对第二和第三维进行转置，然后使用contiguous方法，
        # 这个方法的作用就是能够让转置后的张量应用view方法，否则将无法直接使用，
        # 所以，下一步就是使用view重塑形状，变成和输入形状相同.
        x = x.transpose(1, 2).contiguous().view(nbatches, -1, self.head * self.d_k)

        # 最后使用线性层列表中的最后一个线性层对输入进行线性变换得到最终的多头注意力结构的输出.
        del query
        del key
        del value
        return self.linears[-1](x)

# 通过类PositionwiseFeedForward来实现前馈全连接层
class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        """初始化函数有三个输入参数分别是d_model, d_ff,和dropout=0.1，第一个是线性层的输入维度也是第二个线性层的输出维度，
        因为我们希望输入通过前馈全连接层后输入和输出的维度不变. 第二个参数d_ff就是第二个线性层的输入维度和第一个线性层的输出维度. 
        最后一个是dropout置0比率."""
        super(PositionwiseFeedForward, self).__init__()

        # 首先按照我们预期使用nn实例化了两个线性层对象，self.w1和self.w2
        # 它们的参数分别是d_model, d_ff和d_ff, d_model
        self.w1 = nn.Linear(d_model, d_ff)
        self.w2 = nn.Linear(d_ff, d_model)
        # 然后使用nn的Dropout实例化了对象self.dropout
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """输入参数为x，代表来自上一层的输出"""
        # 首先经过第一个线性层，然后使用Funtional中relu函数进行激活,
        # 之后再使用dropout进行随机置0，最后通过第二个线性层w2，返回最终结果.
        return self.w2(self.dropout(F.relu(self.w1(x))))


# 通过LayerNorm实现规范化层的类
class LayerNorm(nn.Module):
    def __init__(self, features, eps=1e-6):
        """初始化函数有两个参数, 一个是features, 表示词嵌入的维度,
        另一个是eps它是一个足够小的数, 在规范化公式的分母中出现,
        防止分母为0.默认是1e-6."""
        super(LayerNorm, self).__init__()

        # 根据features的形状初始化两个参数张量a2，和b2，第一个初始化为1张量，
        # 也就是里面的元素都是1，第二个初始化为0张量，也就是里面的元素都是0，这两个张量就是规范化层的参数，
        # 因为直接对上一层得到的结果做规范化公式计算，将改变结果的正常表征，因此就需要有参数作为调节因子，
        # 使其即能满足规范化要求，又能不改变针对目标的表征.最后使用nn.parameter封装，代表他们是模型的参数。
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))

        # 把eps传到类中
        self.eps = eps

    def forward(self, x):
        """输入参数x代表来自上一层的输出"""
        # 在函数中，首先对输入变量x求其最后一个维度的均值，并保持输出维度与输入维度一致.
        # 接着再求最后一个维度的标准差，然后就是根据规范化公式，用x减去均值除以标准差获得规范化的结果，
        # 最后对结果乘以我们的缩放参数，即a2，*号代表同型点乘，即对应位置进行乘法操作，加上位移参数b2.返回即可.

        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2



# 使用SublayerConnection来实现子层连接结构的类
class SublayerConnection(nn.Module):
    def __init__(self, size, dropout=0.1):
        """它输入参数有两个, size以及dropout， size一般是都是词嵌入维度的大小， 
        dropout本身是对模型结构中的节点数进行随机抑制的比率， 
        又因为节点被抑制等效就是该节点的输出都是0，因此也可以把dropout看作是对输出矩阵的随机置0的比率.
        """
        super(SublayerConnection, self).__init__()
        # 实例化了规范化对象self.norm
        self.norm = LayerNorm(size)
        # 又使用nn中预定义的droupout实例化一个self.dropout对象.
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, sublayer):
        """前向逻辑函数中, 接收上一个层或者子层的输入作为第一个参数，
        将该子层连接中的子层函数作为第二个参数"""

        # 我们首先对输出进行规范化，然后将结果传给子层处理，之后再对子层进行dropout操作，
        # 随机停止一些网络中神经元的作用，来防止过拟合. 最后还有一个add操作， 
        # 因为存在跳跃连接，所以是将输入x与dropout后的子层输出结果相加作为最终的子层连接输出.
        return x + self.dropout(sublayer(self.norm(x)))


# 使用EncoderLayer类实现编码器层
class EncoderLayer(nn.Module):
    def __init__(self, size, self_attn, feed_forward, dropout):
        """它的初始化函数参数有四个，分别是size，其实就是我们词嵌入维度的大小，它也将作为我们编码器层的大小, 
        第二个self_attn，之后我们将传入多头自注意力子层实例化对象, 并且是自注意力机制, 
        第三个是feed_froward, 之后我们将传入前馈全连接层实例化对象, 最后一个是置0比率dropout."""
        super(EncoderLayer, self).__init__()

        # 首先将self_attn和feed_forward传入其中.
        self.self_attn = self_attn
        self.feed_forward = feed_forward

        # 如图所示, 编码器层中有两个子层连接结构, 所以使用clones函数进行克隆
        self.sublayer = clones(SublayerConnection(size, dropout), 2)
        # 把size传入其中
        self.size = size

    def forward(self, x, mask):
        """forward函数中有两个输入参数，x和mask，分别代表上一层的输出，和掩码张量mask."""
        # 里面就是按照结构图左侧的流程. 首先通过第一个子层连接结构，其中包含多头自注意力子层，
        # 然后通过第二个子层连接结构，其中包含前馈全连接子层. 最后返回结果.
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))
        return self.sublayer[1](x, self.feed_forward)

# 使用Encoder类来实现编码器
class Encoder(nn.Module):
    def __init__(self, layer, N):
        """初始化函数的两个参数分别代表编码器层和编码器层的个数"""
        super(Encoder, self).__init__()
        # 首先使用clones函数克隆N个编码器层放在self.layers中
        self.layers = clones(layer, N)
        # 再初始化一个规范化层, 它将用在编码器的最后面.
        self.norm = LayerNorm(layer.size)

    def forward(self, x, mask):
        """forward函数的输入和编码器层相同, x代表上一层的输出, mask代表掩码张量"""
        # 首先就是对我们克隆的编码器层进行循环，每次都会得到一个新的x，
        # 这个循环的过程，就相当于输出的x经过了N个编码器层的处理. 
        # 最后再通过规范化层的对象self.norm进行处理，最后返回结果. 
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)

def subsequent_mask(size):
    """生成向后遮掩的掩码张量, 参数size是掩码张量最后两个维度的大小, 它的最后两维形成一个方阵"""
    # 在函数中, 首先定义掩码张量的形状
    attn_shape = (1, size, size)

    # 然后使用np.ones方法向这个形状中添加1元素,形成上三角阵, 最后为了节约空间, 
    # 再使其中的数据类型变为无符号8位整形unit8 
    subsequent_mask = torch.triu(torch.ones(attn_shape), diagonal=1).type(torch.uint8)

    # 最后将numpy类型转化为torch中的tensor, 内部做一个1 - 的操作, 
    # 在这个其实是做了一个三角阵的反转, subsequent_mask中的每个元素都会被1减, 
    # 如果是0, subsequent_mask中的该位置由0变成1
    # 如果是1, subsequent_mask中的该位置由1变成0 
    #return torch.from_numpy(1 - subsequent_mask)
    return subsequent_mask == 0

def attention(query, key, value, mask=None, dropout=None):
    """
    注意力机制的实现, 输入分别是query, key, value, mask: 掩码张量, 
    dropout是nn.Dropout层的实例化对象, 默认为None
    """
    # 在函数中, 首先取query的最后一维的大小, 一般情况下就等同于我们的词嵌入维度, 命名为d_k
    d_k = query.size(-1)
    # 按照注意力公式, 将query与key的转置相乘, 这里面key是将最后两个维度进行转置, 再除以缩放系数根号下d_k, 这种计算方法也称为缩放点积注意力计算.
    # 得到注意力得分张量scores
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)

    # 接着判断是否使用掩码张量
    if mask is not None:
        # 使用tensor的masked_fill方法, 将掩码张量和scores张量每个位置一一比较, 如果掩码张量处为0
        # 则对应的scores张量用-1e9这个值来替换, 如下演示
        scores = scores.masked_fill(mask == 0, -1e9)

    # 对scores的最后一维进行softmax操作, 使用F.softmax方法, 第一个参数是softmax对象, 第二个是目标维度.
    # 这样获得最终的注意力张量
    p_attn = F.softmax(scores, dim = -1)

    # 之后判断是否使用dropout进行随机置0
    if dropout is not None:
        # 将p_attn传入dropout对象中进行'丢弃'处理
        p_attn = dropout(p_attn)

    # 最后, 根据公式将p_attn与value张量相乘获得最终的query注意力表示, 同时返回注意力张量
    return torch.matmul(p_attn, value), p_attn


# 首先需要定义克隆函数, 因为在多头注意力机制的实现中, 用到多个结构相同的线性层.
# 我们将使用clone函数将他们一同初始化在一个网络层列表对象中. 之后的结构中也会用到该函数.
def clones(module, N):
    """用于生成相同网络层的克隆函数, 它的参数module表示要克隆的目标网络层, N代表需要克隆的数量"""
    # 在函数中, 我们通过for循环对module进行N次深度拷贝, 使其每个module成为独立的层,
    # 然后将其放在nn.ModuleList类型的列表中存放.
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


class TransformerEncoder(BaseEncoder):
    def __init__(self, sampling_rate=48000, fps=60, frame_skip=4):
        super(TransformerEncoder, self).__init__(sampling_rate, fps, frame_skip)
        #parameter of linear
        self.num_to_subsample = 8
        self.num_samples = (self.sampling_rate / self.FPS) * self.frame_skip
        self.num_frequencies = self.num_samples / 2
        assert int(self.num_samples) == self.num_samples
        self.num_samples = int(self.num_samples)
        self.num_frequencies = int(self.num_frequencies)
        # Encoder_linear (small MLP)

        


    def encode_single_channel(self, data):
        # add some transformer stuffs here
        input_data=torch.from_numpy(1000*(data+1)).long()
        # 词表大小是1000/Tensor数量+1
        vocab_size = input_data.numel()+1
        # 词嵌入维度是512维
        d_model = 64
        #随机抛除系数Dropout
        dropout = 0.12
        # 句子最大长度
        max_len=input_data.shape[1]
        #掩码张量
        mask = torch.ones(1, 1, max_len)
        #多头注意力机制头数
        head = 2
        #前馈全连接中间层Neuron
        d_ff = 16
        #Deepcoopy实例化
        c = copy.deepcopy
        #实例化Embedding和Positional Coding
        embedding = Embeddings(d_model,vocab_size)
        #实例化Positional Coding
        pe = PositionalEncoding(d_model, dropout)
        #MultiHeadedAttention实例化
        attn = MultiHeadedAttention(head, d_model)
        #前馈全连接层FeedForward实例化
        ff = PositionwiseFeedForward(d_model, d_ff, dropout)
        #EncoderLayer实例化
        layer = EncoderLayer(d_model, c(attn), c(ff), dropout)
        # 编码器中编码器层的个数N EncoderLayer个数
        N = 3
        #Encoder实例化
        Enco = Encoder(layer, N)
        #-------------实例化结束，导入数据-----------------#
        #输入x进行Embedding
        input_data=embedding(input_data)
        #输入x进行Positional Coding
        input_data = pe(input_data)
        #将embeddeddata输入Transformer_Encoder得出结果
        Transformer_Model_Encoder=Enco(input_data,mask)
        #finished the transformer_encoder
        #pass to 2-linear 
        Transformer_Model_Encoder = torch.flatten(Transformer_Model_Encoder)
        linear1 = torch.nn.Linear(Transformer_Model_Encoder.numel(), 512)
        Transformer_Model_Encoder = F.relu(linear1(Transformer_Model_Encoder))

        return Transformer_Model_Encoder



if __name__ == '__main__':
    # TODO: compare data value vs pytorch version
    encoder = FFTEncoder()
    import numpy as np
    np.random.seed(0)
    # 这是一个1*3200*2的三维数据
    #data = np.random.randn(1, 3200, 2)

