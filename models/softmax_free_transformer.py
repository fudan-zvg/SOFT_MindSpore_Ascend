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
import math
import mindspore
import mindspore.nn as nn
import mindspore.ops as ops


class ApproxGeLU(nn.Cell):
    def __init__(self):
        super().__init__()
        self.grad_checkpointing = True

    def func(self, x):
        return 0.5 * x * (1 + ops.tanh((2 / math.pi) ** 0.5 * (x + 0.044715 * ops.pow(x, 3))))

    def construct(self, x):
        x = self.func(x)
        return x


def subtraction_gaussian_kernel_torch(q, k):
    # [B, H, H1*W1, C] @ [C, H2*W2] -> [B, H, H1*W1, H2*W2]
    matA_square = ops.matmul(q ** 2., ops.Ones()(k.shape[-2:], mindspore.float16))
    # [H1*W1, C] @ [B, H, C, H2*W2] -> [B, H, H1*W1, H2*W2]
    matB_square = ops.matmul(ops.Ones()(q.shape[-2:], mindspore.float16), k ** 2.)
    return matA_square + matB_square - 2. * ops.matmul(q, k)

class SoftmaxFreeAttentionKernel(nn.Cell):
    def __init__(self, dim, num_heads, q_len, k_len, num_landmark, use_conv, max_iter=20, kernel_method="ms"):
        super().__init__()

        self.head_dim = dim // num_heads
        self.num_head = num_heads
        self.num_landmarks = num_landmark
        self.q_seq_len = q_len
        self.q_seq_len_sqrt = int((self.q_seq_len)**0.5)
        self.k_seq_len = k_len
        self.max_iter = max_iter
        self.concat = ops.Concat(2)
        self.eye = ops.Eye()
        self.matmul = ops.BatchMatMul()

        self.kernel_function = subtraction_gaussian_kernel_torch

        ratio = int((self.q_seq_len // self.num_landmarks)**0.5)
        if ratio == 1:
            self.Qlandmark_op = nn.Dense(in_channels=self.head_dim, out_channels=self.head_dim, has_bias=False)
            self.Qnorm_act = nn.SequentialCell([nn.LayerNorm(normalized_shape=[self.head_dim],
                                                             epsilon=1e-05), nn.GELU()])
        else:
            self.Qlandmark_op = nn.Conv2d(in_channels=self.head_dim, out_channels=self.head_dim,
                                          kernel_size=ratio, stride=ratio,
                                          pad_mode='pad', has_bias=False)
            self.Qnorm_act = nn.SequentialCell([nn.LayerNorm(normalized_shape=[self.head_dim],
                                                             epsilon=1e-05), nn.GELU()])

        self.use_conv = use_conv
        if self.use_conv:
            self.conv = nn.Conv2d(in_channels=self.num_head, out_channels=self.num_head,
                                  kernel_size=(self.use_conv, self.use_conv), pad_mode='pad',
                                  padding=(self.use_conv // 2, self.use_conv // 2,
                                           self.use_conv // 2, self.use_conv // 2),
                                  group=self.num_head, has_bias=False)

    def construct(self, Q, V):
        # b, nhead, seq_len, headdim = Q.shape
        # Q: [b, num_head, N, head_dim]
        Q = Q / (self.head_dim ** 0.25)
        # K=Q
        if self.num_landmarks == self.q_seq_len:
            b = Q.shape[0]
            Q_landmarks = ops.reshape(Q, (b * self.num_head, self.q_seq_len_sqrt * self.q_seq_len_sqrt + 1,
                                          self.head_dim))
            Q_landmarks = self.Qlandmark_op(Q_landmarks)
            Q_landmarks = ops.reshape(self.Qnorm_act(Q_landmarks), \
                                    (b, self.num_head, self.num_landmarks + 1, self.head_dim))
            K_landmarks = Q_landmarks
            attn = self.kernel_function(Q_landmarks, ops.transpose(K_landmarks, (0, 1, 3, 2)))
            attn = self.exp(-attn / 2)
            X = ops.matmul(attn, V)

            # h = w = int(seq_len ** 0.5)
            if self.use_conv:
                b, nhead, _, headdim = Q.shape
                h = w = self.q_seq_len_sqrt
                V_ = V[:, :, 1:, :]
                cls_token = ops.expand_dims(V[:, :, 0, :], 2)
                V_ = ops.reshape(V_, (b, nhead, h, w, headdim))
                V_ = ops.reshape(ops.transpose(V_, (0, 4, 1, 2, 3,)), (b * headdim, nhead, h, w))
                out = ops.transform(ops.reshape(self.conv(V_), (b, headdim, nhead, h * w)), (0, 2, 3, 1))
                out = self.concat([cls_token, out])
                X += out
        else:
            b = Q.shape[0]
            Q_landmarks = ops.transpose(ops.reshape(Q, (b * self.num_head, self.q_seq_len_sqrt, \
                self.q_seq_len_sqrt, self.head_dim)), (0, 3, 1, 2))
            Q_landmarks = self.Qlandmark_op(Q_landmarks) # [64, 32, 7, 7]

            Q_landmarks = ops.transpose(ops.reshape(Q_landmarks, (b, self.num_head, self.head_dim, \
                self.num_landmarks)), (0, 1, 3, 2))
            Q_landmarks = self.Qnorm_act(Q_landmarks)
            K_landmarks = Q_landmarks

            # Q (32, 2, 3136, 32)    K_landmarks  (32, 2, 49, 32)
            kernel_1_ = self.kernel_function(Q, ops.transpose(K_landmarks, (0, 1, 3, 2))) # [32, 2, 3136, 49]
            kernel_1_ = ops.tensor_exp(-kernel_1_ / 2)

            # Q_landmarks (32, 2, 49, 32)  K_landmarks  (32, 2, 49, 32)
            kernel_2_ = self.kernel_function(Q_landmarks, ops.transpose(K_landmarks, (0, 1, 3, 2))) # [32, 2, 49, 49]
            kernel_2_ = ops.tensor_exp(-kernel_2_ / 2)

            kernel_3_ = ops.transpose(kernel_1_, (0, 1, 3, 2)) # (32, 2, 49, 3136)

            X = self.matmul(self.matmul(kernel_1_, self.newton_inv(kernel_2_)), self.matmul(kernel_3_, V))

            # h = w = int(seq_len ** 0.5)
            if self.use_conv:
                b, nhead, _, headdim = Q.shape
                h = w = self.q_seq_len_sqrt
                V = ops.reshape(V, (b, nhead, h, w, headdim))
                V = ops.reshape(ops.transpose(V, (0, 4, 1, 2, 3)), (b*headdim, nhead, h, w))
                X += ops.transpose(ops.reshape(self.conv(V), (b, headdim, nhead, h * w)), (0, 2, 3, 1))

        return X

    def newton_inv(self, mat):
        P = mat
        I = self.eye(mat.shape[-1], mat.shape[-1], mindspore.float16)
        alpha = 0.9 * 2 / (ops.reduce_sum(mat, -1)).max(axis=-1) ** 2
        P += 0.01 * I
        V = ops.BroadcastTo(P.shape)(alpha[..., None, None]) * P

        for _ in range(self.max_iter):
            V = 2 * V - self.matmul(self.matmul(V, P), V)
        return V


class SoftmaxFreeAttention(nn.Cell):
    def __init__(self, dim, num_heads, q_len, k_len, num_landmark, conv_size, max_iter=20, kernel_method="ms"):
        super().__init__()

        self.grad_checkpointing = True
        self.dim = dim
        self.head_dim = dim // num_heads
        self.num_head = num_heads

        self.W_q = nn.Dense(in_channels=self.dim, out_channels=self.num_head * self.head_dim)
        self.W_v = nn.Dense(in_channels=self.dim, out_channels=self.num_head * self.head_dim)

        self.attn = SoftmaxFreeAttentionKernel(dim, num_heads, q_len, k_len, \
            num_landmark, conv_size, max_iter, kernel_method)

        self.ff = nn.Dense(in_channels=self.num_head * self.head_dim, out_channels=self.dim)

    def construct(self, X, return_QKV=False):

        Q = self.split_heads(self.W_q(X))
        V = self.split_heads(self.W_v(X))
        attn_out = self.attn(Q, V)
        attn_out = self.combine_heads(attn_out)

        out = self.ff(attn_out)

        if return_QKV:
            return out, (Q, V)
        return out

    def combine_heads(self, X):

        X = ops.transpose(X, (0, 2, 1, 3))
        X = ops.reshape(X, (X.shape[0], X.shape[1], self.num_head * self.head_dim))
        return X

    def split_heads(self, X):
        X = ops.reshape(X, (X.shape[0], X.shape[1], self.num_head, self.head_dim))

        X = ops.transpose(X, (0, 2, 1, 3))
        return X


class SoftmaxFreeTransformer(nn.Cell):
    def __init__(self, dim, num_heads, q_len, k_len, num_landmark, conv_size,
                 drop_path=0., max_iter=20, kernel_method="ms"):
        super().__init__()
        self.dim = dim
        self.hidden_dim = 4 * dim

        self.mha = SoftmaxFreeAttention(dim, num_heads, q_len, k_len, num_landmark, conv_size, max_iter, kernel_method)

        self.dropout1 = nn.Dropout(keep_prob=1.0 - drop_path)
        self.norm1 = nn.LayerNorm(normalized_shape=[self.dim], epsilon=1e-05)

        self.ff1 = nn.Dense(in_channels=self.dim, out_channels=self.hidden_dim)
        self.act = nn.GELU()
        self.ff2 = nn.Dense(in_channels=self.hidden_dim, out_channels=self.dim)

        self.dropout2 = nn.Dropout(keep_prob=1.0 - drop_path)
        self.norm2 = nn.LayerNorm(normalized_shape=[self.dim], epsilon=1e-05)

    def construct(self, X, return_QKV=False):
        QKV = None
        if return_QKV:
            mha_out, QKV = self.mha(X, return_QKV=True)
        else:
            mha_out = self.mha(X)

        mha_out = self.norm1(X + self.dropout1(mha_out))
        ff_out = self.ff2(self.act(self.ff1(mha_out)))
        mha_out = self.norm2(mha_out + self.dropout2(ff_out))

        if return_QKV:
            print('return_QKV')
            return mha_out, QKV
        return mha_out


class SoftmaxFreeTrasnformerBlock(nn.Cell):
    def __init__(self, dim, num_heads, H, W, drop_path=0., conv_size=3, max_iter=20, kernel_method="ms"):
        super().__init__()
        seq_len = 49
        self.att = SoftmaxFreeTransformer(dim, num_heads, H*W, H*W, seq_len, conv_size, \
                                          drop_path, max_iter, kernel_method)

    def construct(self, x):
        x = self.att(x)
        return x
