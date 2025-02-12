import torch
import spconv.pytorch as spconv
import os
import sys
from copy import deepcopy

from modules.netvlad import NetVLADLoupe

p = os.path.dirname(os.path.dirname((os.path.abspath(__file__))))
if p not in sys.path:
    sys.path.append(p)


def attention(query, key, value):
    dim = query.shape[1]
    scores = torch.einsum('bdhn,bdhm->bhnm', query, key) / dim**.5
    prob = torch.nn.functional.softmax(scores, dim=-1)
    return torch.einsum('bhnm,bdhm->bdhn', prob, value), prob


def MLP(channels: list, do_bn=True):
    """ Multi-layer perceptron """
    n = len(channels)
    layers = []
    for i in range(1, n):
        layers.append(
            torch.nn.Conv1d(channels[i - 1],
                            channels[i],
                            kernel_size=1,
                            bias=True))
        if i < (n - 1):
            if do_bn:
                layers.append(torch.nn.InstanceNorm1d(channels[i]))
            layers.append(torch.nn.ReLU())
    return torch.nn.Sequential(*layers)


class MultiHeadedAttention(torch.nn.Module):
    """ Multi-head attention to increase model expressivitiy """

    def __init__(self, num_heads: int, d_model: int):
        super().__init__()
        assert d_model % num_heads == 0
        self.dim = d_model // num_heads
        self.num_heads = num_heads
        self.merge = torch.nn.Conv1d(d_model, d_model, kernel_size=1)
        self.proj = torch.nn.ModuleList(
            [deepcopy(self.merge) for _ in range(3)])

    def forward(self, query, key, value):
        batch_dim = query.size(0)
        query, key, value = [
            l(x).view(batch_dim, self.dim, self.num_heads, -1)
            for l, x in zip(self.proj, (query, key, value))
        ]
        x, _ = attention(query, key, value)
        return self.merge(x.contiguous().view(batch_dim,
                                              self.dim * self.num_heads, -1))


class AttentionalPropagation(torch.nn.Module):

    def __init__(self, feature_dim: int, num_heads: int):
        super().__init__()
        self.attn = MultiHeadedAttention(num_heads, feature_dim)
        self.mlp = MLP([feature_dim * 2, feature_dim * 2, feature_dim])
        torch.nn.init.constant_(self.mlp[-1].bias, 0.0)

    def forward(self, x, source):
        message = self.attn(x, source, source)
        return self.mlp(torch.cat([x, message], dim=1))


class BottleneckSparse2D(torch.nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size) -> None:
        super(BottleneckSparse2D, self).__init__()
        self.conv = spconv.SparseSequential(
            spconv.SubMConv2d(in_channels, out_channels // 4, 1),
            torch.nn.BatchNorm1d(out_channels // 4), torch.nn.ReLU(),
            spconv.SubMConv2d(out_channels // 4, out_channels // 4,
                              kernel_size),
            torch.nn.BatchNorm1d(out_channels // 4), torch.nn.ReLU(),
            spconv.SubMConv2d(out_channels // 4, out_channels, 1),
            torch.nn.BatchNorm1d(out_channels))
        self.shotcut_conv = spconv.SparseSequential(
            spconv.SubMConv2d(in_channels, out_channels, 1),
            torch.nn.BatchNorm1d(out_channels))
        self.relu = spconv.SparseSequential(torch.nn.ReLU())

    def forward(self, x):
        y = self.conv(x)
        shortcut = self.shotcut_conv(x)
        y = spconv.functional.sparse_add(y, shortcut)
        return self.relu(y)


class Backbone(torch.nn.Module):
    def __init__(self, inchannels=64) -> None:
        super(Backbone, self).__init__()
        self.dconv_down1 = BottleneckSparse2D(inchannels, inchannels * 2, 11)
        self.dconv_down1_1 = BottleneckSparse2D(inchannels * 2, inchannels * 2,
                                                11)
        self.dconv_down2 = BottleneckSparse2D(inchannels * 2, inchannels * 4,
                                              7)
        self.dconv_down2_1 = BottleneckSparse2D(inchannels * 4, inchannels * 4,
                                                7)
        self.dconv_down3 = BottleneckSparse2D(inchannels * 4, inchannels * 8,
                                              5)
        self.dconv_down3_1 = BottleneckSparse2D(inchannels * 8, inchannels * 8,
                                                5)
        self.dconv_down4 = spconv.SubMConv2d(inchannels * 8,
                                             inchannels * 16,
                                             3,
                                             bias=True)
        self.maxpool1 = spconv.SparseMaxPool2d(3, 2, 1, indice_key='up1')
        self.maxpool2 = spconv.SparseMaxPool2d(3, 2, 1, indice_key='up2')
        self.maxpool3 = spconv.SparseMaxPool2d(3, 2, 1, indice_key='up3')

    def forward(self, x):
        x = spconv.SparseConvTensor.from_dense(x)
        conv1 = self.dconv_down1(x)
        x = self.maxpool1(conv1)
        x = self.dconv_down1_1(x)
        conv2 = self.dconv_down2(x)
        x = self.maxpool2(conv2)
        x = self.dconv_down2_1(x)
        conv3 = self.dconv_down3(x)
        x = self.maxpool3(conv3)
        x = self.dconv_down3_1(x)
        x = self.dconv_down4(x)
        return x.dense()


class FeatureFuse(torch.nn.Module):

    def __init__(self, feature_dim, num_heads=1) -> None:
        super(FeatureFuse, self).__init__()
        self.mutihead_attention = AttentionalPropagation(
            feature_dim, num_heads)

    def forward(self, x, source):
        return (x + self.mutihead_attention(x, source))


class AttnVLADHead(torch.nn.Module):
    def __init__(self) -> None:
        super(AttnVLADHead, self).__init__()
        self.self_attention = FeatureFuse(512)
        self.vlad = NetVLADLoupe(
            feature_size=512,
            max_samples=1024,
            cluster_size=32,
            output_dim=1024,
            gating=True,
            add_batch_norm=True,
            is_training=True)

    def forward(self, x):
        x = x.squeeze(1)
        x = x.reshape(x.shape[0], x.shape[1], -1, 1).squeeze(-1)
        x = self.self_attention(x, x)
        return self.vlad(x.reshape(x.shape[0], x.shape[1], -1, 1))

    def pre_dis(self, x):
        dis = self.mlp(x)
        return dis


class OverlapHead(torch.nn.Module):
    def __init__(self, inchannels=64) -> None:
        super(OverlapHead, self).__init__()
        self.fusenet16 = FeatureFuse(inchannels * 16)
        self.last_conv16 = spconv.SparseSequential(
            spconv.SubMConv2d(inchannels * 16, inchannels * 8, 3, bias=True),
            torch.nn.BatchNorm1d(inchannels * 8), torch.nn.ReLU(),
            spconv.SubMConv2d(inchannels * 8, 1, 3, bias=True))

    def forward(self, x):
        x40 = spconv.SparseConvTensor.from_dense(x)
        feature = x40.features
        mask0 = (x40.indices[:, 0] == 0)
        mask1 = (x40.indices[:, 0] == 1)
        fea1 = self.fusenet16(feature[mask0].permute(1, 0).unsqueeze(0),
                              feature[mask1].permute(
                                  1, 0).unsqueeze(0)).squeeze(0).permute(1, 0)
        fea2 = self.fusenet16(feature[mask1].permute(1, 0).unsqueeze(0),
                              feature[mask0].permute(
                                  1, 0).unsqueeze(0)).squeeze(0).permute(1, 0)
        x40 = x40.replace_feature(torch.cat([fea1, fea2], dim=0))
        out4 = self.last_conv16(x40)
        out4 = out4.replace_feature(torch.sigmoid(out4.features))
        score0 = out4.features[mask0]
        score1 = out4.features[mask1]

        # im0 = torch.zeros(x40.spatial_shape).float().to(fea1.device)
        # indi0 = x40.indices[mask0].long()
        # im0[indi0[:,1], indi0[:,2]] = score0.reshape(-1)
        # plt.subplot(1, 2, 1)
        # plt.imshow(im0.detach().cpu().numpy(), cmap = "Reds")
        # # plt.show()

        # im1 = torch.zeros(x40.spatial_shape).float().to(fea1.device)
        # indi1 = x40.indices[mask1].long()
        # im1[indi1[:,1], indi1[:,2]] = score1.reshape(-1)
        # plt.subplot(1, 2, 2)
        # plt.imshow(im1.detach().cpu().numpy(), cmap = "Reds")
        # plt.show()

        score_sum0 = torch.sum(score0) / len(score0)
        score_sum1 = torch.sum(score1) / len(score1)
        return (score_sum0 + score_sum1) / 2., out4


if __name__ == "__main__":
    pass
