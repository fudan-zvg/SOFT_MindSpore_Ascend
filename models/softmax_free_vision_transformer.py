# Copyright 2022 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""Vision Transformer implementation."""
import math
from importlib import import_module
from easydict import EasyDict as edict
import numpy as np

import mindspore
import mindspore.nn as nn
from mindspore.common.parameter import Parameter
from mindspore.common.initializer import initializer
import mindspore.ops as ops
from .softmax_free_transformer import SoftmaxFreeTrasnformerBlock

class DropPath(nn.Cell):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """

    def __init__(self, drop_prob=None, seed=0):
        super(DropPath, self).__init__()
        self.keep_prob = 1 - drop_prob
        seed = min(seed, 0) # always be 0
        self.rand = ops.UniformReal(seed=seed) # seed must be 0, if set to other value, it's not rand for multiple call
        self.shape = ops.Shape()
        self.floor = ops.Floor()

    def construct(self, x):
        if self.training:
            x_shape = self.shape(x) # B N C
            random_tensor = self.rand((x_shape[0], 1, 1))
            random_tensor = random_tensor + self.keep_prob
            random_tensor = self.floor(random_tensor)
            x = x / self.keep_prob
            x = x * random_tensor
        return x


class SOFTConfig:
    """
    SOFTConfig
    """

    def __init__(self, configs):
        self.configs = configs

        # network init
        self.network_dropout_rate = 0.0
        self.network = SoftmaxFreeVisionTransformer
        self.network_init = mindspore.common.initializer.TruncatedNormal(sigma=0.02)

        # body
        self.body_drop_path_rate = 0.0

        # body attention
        self.attention_init = mindspore.common.initializer.TruncatedNormal(sigma=0.02)
        self.attention_activation = mindspore.nn.Softmax()
        self.attention_dropout_rate = 0.0
        self.project_dropout_rate = 0.0
        self.attention = Attention
        self.qkv_bias = True

        # body feedforward
        self.feedforward_init = mindspore.common.initializer.TruncatedNormal(sigma=0.02)
        self.feedforward_activation = mindspore.nn.GELU()
        self.feedforward_dropout_rate = 0.0
        self.feedforward = FeedForward

        # head
        self.head_init = mindspore.common.initializer.TruncatedNormal(sigma=0.02)


class ResidualCell(nn.Cell):
    """Cell which implements x + f(x) function."""
    def __init__(self, cell):
        super().__init__()
        self.cell = cell

    def construct(self, x):
        return self.cell(x) + x

class FeedForward(nn.Cell):
    """FeedForward layer implementation."""

    def __init__(self, soft_config, d_model, mlp_ratio):
        super().__init__()

        hidden_dim = int(d_model * mlp_ratio)

        initialization = soft_config.feedforward_init
        activation = soft_config.feedforward_activation
        dropout_rate = soft_config.feedforward_dropout_rate


        self.ff1 = nn.Dense(d_model, hidden_dim, weight_init=initialization)
        self.activation = activation
        self.dropout = nn.Dropout(keep_prob=1. - dropout_rate)
        self.ff2 = nn.Dense(hidden_dim, d_model, weight_init=initialization)

    def construct(self, x):
        x = self.ff1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.ff2(x)
        x = self.dropout(x)
        return x


class PatchEmbed(nn.Cell):
    """ Image to Patch Embedding """

    def __init__(self, img_size=224, kernel_size=7, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        # self.img_size = img_size
        # self.patch_size = patch_size
        assert img_size % patch_size == 0, \
            f"img_size {img_size} should be divided by patch_size {patch_size}."
        he_uniform = mindspore.common.initializer.HeUniform(math.sqrt(5))
        self.H = self.W = img_size // patch_size
        self.num_patches = self.H * self.W
        self.kernel_size = kernel_size
        self.conv1 = nn.Conv2d(in_channels=in_chans, out_channels=32, kernel_size=3, stride=2,
                               pad_mode='pad', padding=1, has_bias=False, weight_init=he_uniform)
        self.norm1 = nn.BatchNorm2d(num_features=32)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1,
                               pad_mode='pad', padding=1, has_bias=False, weight_init=he_uniform)
        self.norm2 = nn.BatchNorm2d(num_features=32)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=embed_dim, kernel_size=3, stride=2,
                               pad_mode='pad', padding=1, has_bias=False, weight_init=he_uniform)
        self.norm3 = nn.BatchNorm2d(num_features=embed_dim)
        self.relu = nn.ReLU()
        self.conv = nn.Conv2d(in_channels=in_chans, out_channels=embed_dim, kernel_size=3, stride=2,
                              pad_mode='pad', padding=1, has_bias=False, weight_init=he_uniform)
        self.norm = nn.BatchNorm2d(num_features=embed_dim)

    def construct(self, x):
        if self.kernel_size == 7:
            x = self.relu(self.norm1(self.conv1(x)))
            x = self.relu(self.norm2(self.conv2(x)))
            x = self.relu(self.norm3(self.conv3(x)))
        else:
            x = self.relu(self.norm(self.conv(x)))
        return x


class Attention(nn.Cell):
    """Attention layer implementation."""

    def __init__(self, soft_config, d_model, num_heads, sr_ratio, fea_size):
        super().__init__()
        assert d_model % num_heads == 0, f"dim {d_model} should be divided by num_heads {num_heads}."
        dim_head = d_model // num_heads

        initialization = soft_config.attention_init
        activation = soft_config.attention_activation  # softmax
        attn_drop = soft_config.attention_dropout_rate
        proj_drop = soft_config.project_dropout_rate
        qkv_bias = soft_config.qkv_bias

        inner_dim = num_heads * dim_head
        self.dim_head = dim_head
        self.num_heads = num_heads
        self.scale = dim_head ** -0.5
        self.H, self.W = fea_size

        self.q = nn.Dense(d_model, inner_dim, has_bias=qkv_bias, weight_init=initialization)
        self.kv = nn.Dense(d_model, inner_dim * 2, has_bias=qkv_bias, weight_init=initialization)

        self.sr_ratio = sr_ratio
        if self.sr_ratio > 1:
            self.sr = nn.Conv2d(in_channels=d_model, out_channels=inner_dim, kernel_size=sr_ratio,
                                stride=sr_ratio, pad_mode='pad', has_bias=True)
            self.norm = nn.LayerNorm(normalized_shape=(d_model))

        self.proj = nn.Dense(inner_dim, d_model, weight_init=initialization)
        self.attn_drop = nn.Dropout(1 - attn_drop)
        self.proj_drop = nn.Dropout(1 - proj_drop)
        self.activation = activation

        # auxiliary functions
        self.unstack = ops.Unstack(0)
        self.q_matmul_k = ops.BatchMatMul(transpose_b=True)
        self.attn_matmul_v = ops.BatchMatMul()

    def construct(self, x):
        '''x size - BxNxd_model'''
        bs, seq_len, d_model, h, d = x.shape[0], x.shape[1], x.shape[2], self.num_heads, self.dim_head

        q = ops.transpose(ops.reshape(self.q(x), (bs, seq_len, h, d)), (0, 2, 1, 3))

        if self.sr_ratio > 1:
            x_ = ops.reshape(ops.transpose(x, (0, 2, 1)), (bs, d_model, self.H, self.W))
            x_ = ops.transpose(ops.reshape(self.sr(x_), (bs, d_model, -1)), (0, 2, 1))
            x_ = self.norm(x_)
            kv = ops.transpose(ops.reshape(self.kv(x_), (bs, -1, 2, h, d)), (2, 0, 3, 1, 4))
        else:
            kv = ops.transpose(ops.reshape(self.kv(x), (bs, -1, 2, h, d)), (2, 0, 3, 1, 4))
        k, v = self.unstack(kv)
        # k, v [bs, h, seq_len, d]

        attn = self.q_matmul_k(q, k) * self.scale # [bs, h, seq_len, seq_len]
        attn = self.activation(attn)
        attn = self.attn_drop(attn)

        # attn : [bs, h, seq_len, seq_len] v : [bs, h, seq_len, d]
        x = self.attn_matmul_v(attn, v) # [bs, h, seq_len, d]
        x = ops.reshape(ops.transpose(x, (0, 2, 1, 3)), (bs, seq_len, d_model))
        x = self.proj(x)
        x = self.proj_drop(x)

        return x



class Block(nn.Cell):

    def __init__(self, soft_config, d_model, num_heads, mlp_ratio, sr_ratio, fea_size, dpr):
        super().__init__()

        attention = soft_config.attention(soft_config, d_model, num_heads, sr_ratio, fea_size)
        feedforward = soft_config.feedforward(soft_config, d_model, mlp_ratio)

        if dpr > 0:
            self.layers = nn.SequentialCell([
                ResidualCell(nn.SequentialCell([nn.LayerNorm((d_model,)),
                                                attention,
                                                DropPath(dpr)])),
                ResidualCell(nn.SequentialCell([nn.LayerNorm((d_model,)),
                                                feedforward,
                                                DropPath(dpr)]))
            ])
        else:
            self.layers = nn.SequentialCell([
                ResidualCell(nn.SequentialCell([nn.LayerNorm((d_model,)),
                                                attention])),
                ResidualCell(nn.SequentialCell([nn.LayerNorm((d_model,)),
                                                feedforward]))
            ])

    def construct(self, x):
        return self.layers(x)


class SoftmaxFreeVisionTransformer(nn.Cell):
    def __init__(self, soft_config):
        super().__init__()
        num_classes = soft_config.configs.num_classes
        img_size = soft_config.configs.img_size
        patch_size = soft_config.configs.patch_size
        in_chans = soft_config.configs.in_chans
        embed_dims = soft_config.configs.embed_dims
        num_heads = soft_config.configs.num_heads
        mlp_ratios = soft_config.configs.mlp_ratios
        drop_path_rate = soft_config.body_drop_path_rate
        drop_rate = soft_config.network_dropout_rate
        depths = soft_config.configs.depths
        sr_ratios = soft_config.configs.sr_ratios
        newton_max_iter = soft_config.configs.newton_max_iter
        kernel_method = soft_config.configs.kernel_method

        self.concat = ops.Concat(1)
        truncated_normal = soft_config.network_init
        initialization = mindspore.common.initializer.Normal(sigma=1.0)

        # patch_embed
        in_chans = (in_chans, embed_dims[0], embed_dims[1], embed_dims[2])
        self.patch_embed1 = PatchEmbed(img_size=img_size, kernel_size=7, patch_size=patch_size[0],
                                       in_chans=in_chans[0], embed_dim=embed_dims[0])
        self.patch_embed2 = PatchEmbed(img_size=img_size // 4, kernel_size=3, patch_size=patch_size[1],
                                       in_chans=in_chans[1], embed_dim=embed_dims[1])
        self.patch_embed3 = PatchEmbed(img_size=img_size // 8, kernel_size=3, patch_size=patch_size[2],
                                       in_chans=in_chans[2], embed_dim=embed_dims[2])
        self.patch_embed4 = PatchEmbed(img_size=img_size // 16, kernel_size=3, patch_size=patch_size[3],
                                       in_chans=in_chans[3], embed_dim=embed_dims[3])

        # pos_embed
        self.pos_embed1 = Parameter(initializer(initialization, (1, 3136, embed_dims[0])))
        self.pos_drop1 = nn.Dropout(keep_prob=1.0 - drop_rate)

        self.pos_embed2 = Parameter(initializer(initialization, (1, 784, embed_dims[1])))
        self.pos_drop2 = nn.Dropout(keep_prob=1.0 - drop_rate)

        self.pos_embed3 = Parameter(initializer(initialization, (1, 196, embed_dims[2])))
        self.pos_drop3 = nn.Dropout(keep_prob=1.0 - drop_rate)

        self.pos_embed4 = Parameter(initializer(initialization, (1, 49 + 1, embed_dims[3])))
        self.pos_drop4 = nn.Dropout(keep_prob=1.0 - drop_rate)

        # transformer encoder
        dpr = [x.item() for x in np.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule
        cur = 0
        self.block1 = nn.SequentialCell(
            [SoftmaxFreeTrasnformerBlock(dim=embed_dims[0], num_heads=num_heads[0],
                                         drop_path=dpr[cur + i], H=56, W=56, conv_size=9,
                                         max_iter=newton_max_iter,
                                         kernel_method=kernel_method) for i in range(depths[0])])

        cur += depths[0]
        self.block2 = nn.SequentialCell(
            [SoftmaxFreeTrasnformerBlock(dim=embed_dims[1], num_heads=num_heads[1],
                                         drop_path=dpr[cur + i], H=28, W=28, conv_size=5,
                                         max_iter=newton_max_iter,
                                         kernel_method=kernel_method) for i in range(depths[1])])

        cur += depths[1]
        self.block3 = nn.SequentialCell(
            [SoftmaxFreeTrasnformerBlock(dim=embed_dims[2], num_heads=num_heads[2],
                                         drop_path=dpr[cur + i], H=14, W=14, conv_size=3,
                                         max_iter=newton_max_iter,
                                         kernel_method=kernel_method) for i in range(depths[2])])

        cur += depths[2]
        self.block4 = nn.SequentialCell(
            [Block(soft_config, embed_dims[3], num_heads[3], mlp_ratios[3],
                   sr_ratios[3],
                   (img_size//(patch_size[0]*patch_size[1]*patch_size[2]*patch_size[3]),
                    img_size // (patch_size[0]*patch_size[1]*patch_size[2]*patch_size[3])),
                   dpr[cur + i]) for i in range(depths[3])])

        self.norm = nn.LayerNorm(normalized_shape=(embed_dims[3],), epsilon=1e-6)

        # cls_token
        self.cls_token = Parameter(initializer(initialization, (1, 1, embed_dims[3])),
                                   name='cls', requires_grad=True)

        # classification head
        self.head = nn.Dense(in_channels=embed_dims[3],
                             out_channels=num_classes,
                             weight_init=truncated_normal) if num_classes > 0 else ops.Identity()


    def no_weight_decay(self):
        return {'cls_token'}

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=''):
        truncated_normal = mindspore.common.initializer.TruncatedNormal(sigma=0.02)
        self.num_classes = num_classes
        if num_classes > 0:
            self.head = nn.Dense(in_channels=self.embed_dim, out_channels=num_classes, weight_init=truncated_normal)

    def forward_features(self, x):

        # stage 1
        x = self.patch_embed1(x)
        B, C, H, W = x.shape
        x = ops.transpose(ops.reshape(x, (B, C, H*W)), (0, 2, 1))
        x = x + self.pos_embed1
        x = self.pos_drop1(x)
        x = self.block1(x)
        x = ops.transpose(ops.reshape(x, (B, H, W, -1)), (0, 3, 1, 2))

        # stage 2
        x = self.patch_embed2(x)
        B, C, H, W = x.shape
        x = ops.transpose(ops.reshape(x, (B, C, H*W)), (0, 2, 1))
        x = x + self.pos_embed2
        x = self.pos_drop2(x)
        x = self.block2(x)
        x = ops.transpose(ops.reshape(x, (B, H, W, -1)), (0, 3, 1, 2))

        # stage 3
        x = self.patch_embed3(x)
        B, C, H, W = x.shape
        x = ops.transpose(ops.reshape(x, (B, C, H*W)), (0, 2, 1))
        x = x + self.pos_embed3
        x = self.pos_drop3(x)
        x = self.block3(x)
        x = ops.transpose(ops.reshape(x, (B, H, W, -1)), (0, 3, 1, 2))

        # stage 4
        x = self.patch_embed4(x)
        B, C, H, W = x.shape
        x = ops.transpose(ops.reshape(x, (B, C, H*W)), (0, 2, 1))

        cls_tokens = ops.BroadcastTo((B, -1, -1))(self.cls_token)
        x = self.concat((cls_tokens, x))
        x = x + self.pos_embed4
        x = self.pos_drop4(x)
        x = self.block4(x)

        x = self.norm(x)
        return x[:, 0]

    def construct(self, x):

        x = self.forward_features(x)
        x = self.head(x)

        return x



def load_function(func_name):
    """Load function using its name."""
    modules = func_name.split(".")
    if len(modules) > 1:
        module_path = ".".join(modules[:-1])
        name = modules[-1]
        module = import_module(module_path)
        return getattr(module, name)
    return func_name


soft_cfg = edict({
    'img_size': 224,
    'patch_size': (4, 2, 2, 2),
    'embed_dims': (64, 128, 256, 512),
    'depths': (3, 4, 6, 3),
    'num_heads': (1, 2, 4, 8),
    'mlp_ratios': (4, 4, 4, 4),
    'sr_ratios': (8, 4, 2, 1),
    'in_chans': 3,
    'num_classes': 1001,
    'newton_max_iter': 20,
    'kernel_method': "ms"
})


def soft_tiny(args):
    """soft_tiny"""
    soft_cfg.img_size = args.train_image_size
    soft_cfg.patch_size = (4, 2, 2, 2)
    soft_cfg.embed_dims = (64, 128, 320, 512)
    soft_cfg.depths = (1, 2, 3, 2)
    soft_cfg.num_heads = (2, 4, 10, 16)
    soft_cfg.mlp_ratios = (8, 8, 4, 4)
    soft_cfg.sr_ratios = (8, 4, 2, 1)
    soft_cfg.in_chans = 3
    soft_cfg.num_classes = args.class_num
    soft_cfg.newton_max_iter = 20
    soft_cfg.kernel_method = "ms"

    if args.soft_config_path != '':
        print("get soft_config_path")
        soft_config = load_function(args.soft_config_path)(soft_cfg)
    else:
        print("get default_soft_cfg")
        soft_config = SOFTConfig(soft_cfg)

    model = soft_config.network(soft_config)
    return model

def soft_small(args):
    """soft_small"""
    soft_cfg.img_size = args.train_image_size
    soft_cfg.patch_size = (4, 2, 2, 2)
    soft_cfg.embed_dims = (64, 128, 320, 512)
    soft_cfg.depths = (1, 3, 20, 4)
    soft_cfg.num_heads = (2, 4, 10, 16)
    soft_cfg.mlp_ratios = (8, 8, 4, 4)
    soft_cfg.sr_ratios = (8, 4, 2, 1)
    soft_cfg.in_chans = 3
    soft_cfg.num_classes = args.class_num
    soft_cfg.newton_max_iter = 20
    soft_cfg.kernel_method = "ms"

    if args.soft_config_path != '':
        print("get soft_config_path")
        soft_config = load_function(args.soft_config_path)(soft_cfg)
    else:
        print("get default_soft_cfg")
        soft_config = SOFTConfig(soft_cfg)

    model = soft_config.network(soft_config)
    return model


def soft_medium(args):
    """soft_medium"""
    soft_cfg.img_size = args.train_image_size
    soft_cfg.patch_size = (4, 2, 2, 2)
    soft_cfg.embed_dims = (64, 128, 288, 512)
    soft_cfg.depths = (1, 3, 29, 5)
    soft_cfg.num_heads = (2, 4, 9, 16)
    soft_cfg.mlp_ratios = (8, 8, 4, 4)
    soft_cfg.sr_ratios = (8, 4, 2, 1)
    soft_cfg.in_chans = 3
    soft_cfg.num_classes = args.class_num
    soft_cfg.newton_max_iter = 20
    soft_cfg.kernel_method = "ms"

    if args.soft_config_path != '':
        print("get soft_config_path")
        soft_config = load_function(args.soft_config_path)(soft_cfg)
    else:
        print("get default_soft_cfg")
        soft_config = SOFTConfig(soft_cfg)

    model = soft_config.network(soft_config)
    return model


def soft_large(args):
    """soft_tiny"""
    soft_cfg.img_size = args.train_image_size
    soft_cfg.patch_size = (4, 2, 2, 2)
    soft_cfg.embed_dims = (64, 128, 320, 512)
    soft_cfg.depths = (1, 3, 40, 5)
    soft_cfg.num_heads = (2, 4, 10, 16)
    soft_cfg.mlp_ratios = (8, 8, 4, 4)
    soft_cfg.sr_ratios = (8, 4, 2, 1)
    soft_cfg.in_chans = 3
    soft_cfg.num_classes = args.class_num
    soft_cfg.newton_max_iter = 20
    soft_cfg.kernel_method = "ms"

    if args.soft_config_path != '':
        print("get soft_config_path")
        soft_config = load_function(args.soft_config_path)(soft_cfg)
    else:
        print("get default_soft_cfg")
        soft_config = SOFTConfig(soft_cfg)

    model = soft_config.network(soft_config)
    return model



def get_network(backbone_name, args):
    """get_network"""
    if backbone_name == 'soft_tiny':
        backbone = soft_tiny(args)
    elif backbone_name == 'soft_small':
        backbone = soft_small(args)
    elif backbone_name == 'soft_medium':
        backbone = soft_medium(args)
    elif backbone_name == 'soft_large':
        backbone = soft_large(args)
    else:
        raise NotImplementedError

    return backbone
