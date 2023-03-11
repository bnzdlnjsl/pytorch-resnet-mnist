import torch
import torch.nn as nn
import torch.nn.functional as F


class Conv_BN_ReLU(nn.Module):
    def __init__(
        self, in_channels, out_channels, kernel_size,
        stride=1, padding=0, dilation=1, groups=1, bias=False, padding_mode='zeros',
        eps=1e-05, momentum=0.1, affine=True, track_running_stats=True,
        need_relu=True, inplace=True,
        device=None, dtype=None
    ):
        super(Conv_BN_ReLU, self).__init__()
        conv = nn.Conv2d(
            in_channels, out_channels, kernel_size,
            stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias,
            padding_mode=padding_mode,
            device=device, dtype=dtype
        )
        bn = nn.BatchNorm2d(
            out_channels, eps=eps, momentum=momentum, affine=affine,
            track_running_stats=track_running_stats,
            device=device, dtype=dtype
        )
        if need_relu:
            relu = nn.ReLU(inplace=inplace)
            self.seq = nn.Sequential(
                conv,
                bn,
                relu
            )
        else:
            self.seq = nn.Sequential(
                conv,
                bn
            )
        return

    def forward(self, x):
        return self.seq(x)


# class Conv_BN_ReLU(nn.Module):
#     def __init__(
#         self, in_channels, out_channels, kernel_size,
#         stride=1, padding=0, dilation=1, groups=1, bias=False, padding_mode='zeros',
#         eps=1e-05, momentum=0.1, affine=True, track_running_stats=True,
#         need_relu=True, inplace=True,
#         device=None, dtype=None
#     ):
#         super(Conv_BN_ReLU, self).__init__()
#         self.conv = nn.Conv2d(
#             in_channels, out_channels, kernel_size,
#             stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias,
#             padding_mode=padding_mode,
#             device=device, dtype=dtype
#         )
#         self.bn = nn.BatchNorm2d(
#             out_channels, eps=eps, momentum=momentum, affine=affine,
#             track_running_stats=track_running_stats,
#             device=device, dtype=dtype
#         )
#         self.need_relu = need_relu
#         if self.need_relu:
#             self.relu = nn.ReLU(inplace=inplace)
#         return

#     def forward(self, x):
#         if self.need_relu:
#             return nn.Sequential(
#                 self.conv,
#                 self.bn,
#                 self.relu
#             )(x)
#         else:
#             return nn.Sequential(
#                 self.conv,
#                 self.bn,
#             )(x)


class Normal(nn.Module):
    def __init__(self, out_channels, manual: bool = False, stride=None, in_channels=None, device=None, dtype=None):
        super(Normal, self).__init__()

        if manual:
            if not stride or not in_channels:
                raise ValueError("must explicitly specify stride and number of in channels in manual mode")
        else:
            stride = 1
            in_channels = out_channels

        KERNEL_SIZE = 3
        PADDING = KERNEL_SIZE // 2
        conv_in = Conv_BN_ReLU(
            in_channels, out_channels, KERNEL_SIZE, stride=stride, padding=PADDING,
            device=device, dtype=dtype
        )
        conv_out = Conv_BN_ReLU(
            out_channels, out_channels, KERNEL_SIZE, padding=PADDING,
            need_relu=False,
            device=device, dtype=dtype
        )
        self.branch_residual = nn.Sequential(
            conv_in,
            conv_out
        )

        if in_channels != out_channels or stride != 1:
            KERNEL_SIZE_SHORT_CUT = 1
            PADDING_SHORT_CUT = KERNEL_SIZE_SHORT_CUT // 2
            self.branch_short_cut = Conv_BN_ReLU(
                in_channels, out_channels, KERNEL_SIZE_SHORT_CUT, stride=stride, padding=PADDING_SHORT_CUT,
                need_relu=False,
                device=device, dtype=dtype
            )
        else:
            # self.branch_short_cut = lambda x: x
            self.branch_short_cut = nn.Sequential()

        self.relu_merge = nn.ReLU(inplace=True)

        return

    def forward(self, x):
        out_merge = self.branch_residual(x) + self.branch_short_cut(x)
        out = self.relu_merge(out_merge)
        return out


class Bottleneck(nn.Module):
    def __init__(self, mid_channels, manual: bool = False, stride=None, in_channels=None, device=None, dtype=None):
        super(Bottleneck, self).__init__()

        if manual:
            if not stride or not in_channels:
                raise ValueError("must explicitly specify stride and number of in channels in manual mode")
        else:
            stride = 1
            in_channels = mid_channels * 4

        KERNEL_SIZE_IN = 1
        KERNEL_SIZE_MID = 3
        KERNEL_SIZE_OUT = 1
        PADDING_IN = KERNEL_SIZE_IN // 2
        PADDING_MID = KERNEL_SIZE_MID // 2
        PADDING_OUT = KERNEL_SIZE_OUT // 2

        out_channels = mid_channels * 4

        conv_in = Conv_BN_ReLU(
            in_channels, mid_channels, KERNEL_SIZE_IN, padding=PADDING_IN,
            device=device, dtype=dtype
        )
        conv_mid = Conv_BN_ReLU(
            mid_channels, mid_channels, KERNEL_SIZE_MID, stride=stride, padding=PADDING_MID,
            device=device, dtype=dtype
        )
        conv_out = Conv_BN_ReLU(
            mid_channels, out_channels, KERNEL_SIZE_OUT, padding=PADDING_OUT,
            need_relu=False,
            device=device, dtype=dtype
        )
        self.branch_residual = nn.Sequential(
            conv_in,
            conv_mid,
            conv_out
        )

        if in_channels != out_channels or stride != 1:
            KERNEL_SIZE_SHORT_CUT = 1
            PADDING_SHORT_CUT = KERNEL_SIZE_SHORT_CUT // 2
            self.branch_short_cut = Conv_BN_ReLU(
                in_channels, out_channels, KERNEL_SIZE_SHORT_CUT, stride=stride, padding=PADDING_SHORT_CUT,
                need_relu=False,
                device=device, dtype=dtype
            )
        else:
            # self.branch_short_cut = lambda x: x
            self.branch_short_cut = nn.Sequential()

        self.relu_merge = nn.ReLU(inplace=True)

        return

    def forward(self, x):
        out_merge = self.branch_residual(x) + self.branch_short_cut(x)
        out = self.relu_merge(out_merge)
        return out


class BlockGroup(nn.Module):
    def __init__(self, building_block_name, block_group_size: int, in_channels: int, channel_level: int, stride: int, device=None, dtype=None):
        super(BlockGroup, self).__init__()

        if building_block_name == "normal":
            BB_Class = Normal
        elif building_block_name == "bottleneck":
            BB_Class = Bottleneck
        else:
            raise ValueError("invalid building block name")

        self.seq = nn.Sequential()
        conv = BB_Class(channel_level, manual=True, stride=stride, in_channels=in_channels, device=device, dtype=dtype)
        self.seq.append(conv)
        for _ in range(1, block_group_size):
            conv = BB_Class(channel_level, device=device, dtype=dtype)
            self.seq.append(conv)
        del conv
        return

    def forward(self, x):
        return self.seq(x)


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.shape[0], -1)


class ResNet(nn.Module):
    def __init__(self, in_channels: int, building_block_name: str, block_group_sizes: list or tuple, num_classes: int, device=None, dtype=None):
        assert len(block_group_sizes) == 4, "must specify the sizes of exactly four block groups"

        super(ResNet, self).__init__()

        KERNEL_SIZE_CONV_1 = 7
        PADDING_CONV_1 = KERNEL_SIZE_CONV_1 // 2
        KERNEL_SIZE_MAX_POOL = 3
        PADDING_MAX_POOL = KERNEL_SIZE_MAX_POOL // 2

        def get_out_channels_pre(channel_level):
            if building_block_name == "normal":
                return channel_level
            elif building_block_name == "bottleneck":
                return 4 * channel_level
            else:
                raise ValueError("invalid building block name")

        channel_level = 64
        conv_1 = Conv_BN_ReLU(in_channels, channel_level, KERNEL_SIZE_CONV_1, stride=2, padding=PADDING_CONV_1, device=device, dtype=dtype)
        out_channels_pre = channel_level

        channel_level *= 1
        conv_2 = nn.Sequential(
            nn.MaxPool2d(KERNEL_SIZE_MAX_POOL, stride=2, padding=PADDING_MAX_POOL),
            BlockGroup(building_block_name, block_group_sizes[0], out_channels_pre, channel_level, 1, device=device, dtype=dtype)
        )
        out_channels_pre = get_out_channels_pre(channel_level)

        channel_level *= 2
        conv_3 = BlockGroup(building_block_name, block_group_sizes[1], out_channels_pre, channel_level, 2, device=device, dtype=dtype)
        out_channels_pre = get_out_channels_pre(channel_level)

        channel_level *= 2
        conv_4 = BlockGroup(building_block_name, block_group_sizes[2], out_channels_pre, channel_level, 2, device=device, dtype=dtype)
        out_channels_pre = get_out_channels_pre(channel_level)

        channel_level *= 2
        conv_5 = BlockGroup(building_block_name, block_group_sizes[3], out_channels_pre, channel_level, 2, device=device, dtype=dtype)
        out_channels_pre = get_out_channels_pre(channel_level)

        shape_GAP = (1, 1)
        gap = nn.AdaptiveAvgPool2d(shape_GAP)
        flatten = Flatten()

        fc = nn.Linear(out_channels_pre, num_classes, device=device, dtype=dtype)

        self.seq = nn.Sequential(
            conv_1,
            conv_2,
            conv_3,
            conv_4,
            conv_5,
            gap,
            flatten,
            fc
        )
        return

    def forward(self, x):
        return self.seq(x)


def ResNet_18(in_channels, num_classes, device=None, dtype=None):
    return ResNet(in_channels, "normal", [2, 2, 2, 2], num_classes, device=device, dtype=dtype)


def ResNet_34(in_channels, num_classes, device=None, dtype=None):
    return ResNet(in_channels, "normal", [3, 4, 6, 3], num_classes, device=device, dtype=dtype)


def ResNet_50(in_channels, num_classes, device=None, dtype=None):
    return ResNet(in_channels, "bottleneck", [3, 4, 6, 3], num_classes, device=device, dtype=dtype)


def ResNet_101(in_channels, num_classes, device=None, dtype=None):
    return ResNet(in_channels, "bottleneck", [3, 4, 23, 3], num_classes, device=device, dtype=dtype)


def ResNet_152(in_channels, num_classes, device=None, dtype=None):
    return ResNet(in_channels, "bottleneck", [3, 8, 36, 3], num_classes, device=device, dtype=dtype)


def Size2ResNet(resnet_size: int, in_channels, num_classes, device=None, dtype=None):
    if resnet_size == 18:
        return ResNet_18(in_channels, num_classes, device=device, dtype=dtype)
    elif resnet_size == 34:
        return ResNet_34(in_channels, num_classes, device=device, dtype=dtype)
    elif resnet_size == 50:
        return ResNet_50(in_channels, num_classes, device=device, dtype=dtype)
    elif resnet_size == 101:
        return ResNet_101(in_channels, num_classes, device=device, dtype=dtype)
    elif resnet_size == 152:
        return ResNet_152(in_channels, num_classes, device=device, dtype=dtype)
    else:
        raise ValueError("invalid resnet size")
