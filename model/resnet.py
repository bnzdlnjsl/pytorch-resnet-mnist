import torch
import torch.nn as nn
import torch.nn.functional as F


class Conv_BN_ReLU(nn.Module):
    r"""Helper module for combining the three layer into one, with the ReLU layer being optional."""
    
    def __init__(
        self, in_channels, out_channels, kernel_size,
        stride=1, padding=0, dilation=1, groups=1, bias=False, padding_mode='zeros',
        eps=1e-05, momentum=0.1, affine=True, track_running_stats=True,
        need_relu=True, inplace=True,
        device=None, dtype=None
    ):
        super(Conv_BN_ReLU, self).__init__()
        
        """
        The bias of conv layer defaults to False since the batchnorm layer comes right after it.
        """
        conv = nn.Conv2d(
            in_channels, out_channels, kernel_size,
            stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias,
            padding_mode=padding_mode,
            device=device, dtype=dtype
        )
        """
        Manually initialize at the bottom level of the entire net using kaiming initialization.
        """
        nn.init.kaiming_normal_(conv.weight)
        
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


class Normal(nn.Module):
    r"""The Basic building block."""
    
    def __init__(self, out_channels, manual: bool = False, stride=None, 
                 in_channels=None, device=None, dtype=None):
        
        super(Normal, self).__init__()

        """
        If manual is set to True then stride and in_channels (i.e. out_channels of
        previous layer) will be specified by the caller, otherwise in_channels is
        deduced by out_channels for consistency and stride defaults to 1.
        """
        if manual:
            if not stride or not in_channels:
                raise ValueError(
                    "must explicitly specify stride and number of in channels in manual mode")
        else:
            stride = 1
            in_channels = out_channels

        """
        The residual path.
        """
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

        """
        The short cut path. Generated only if the shapes of input and output
        is not the same, otherwise the identity mapping is used.
        """
        if in_channels != out_channels or stride != 1:
            """
            Projection mapping.
            """
            KERNEL_SIZE_SHORT_CUT = 1
            PADDING_SHORT_CUT = KERNEL_SIZE_SHORT_CUT // 2
            self.branch_short_cut = Conv_BN_ReLU(
                in_channels, out_channels, KERNEL_SIZE_SHORT_CUT, stride=stride, padding=PADDING_SHORT_CUT,
                need_relu=False,
                device=device, dtype=dtype
            )
        else:
            """
            Identity mapping.
            """
            # self.branch_short_cut = lambda x: x
            self.branch_short_cut = nn.Sequential()

        self.relu_merge = nn.ReLU(inplace=True)

        return

    def forward(self, x):
        out_merge = self.branch_residual(x) + self.branch_short_cut(x)
        out = self.relu_merge(out_merge)
        return out


class Bottleneck(nn.Module):
    r"""The Bottleneck building block."""
    
    def __init__(self, mid_channels, manual: bool = False, stride=None, in_channels=None, device=None, dtype=None):
        super(Bottleneck, self).__init__()


        """
        If manual is set to True then stride and in_channels (i.e. out_channels of
        previous layer) will be specified by the caller, otherwise in_channels is
        deduced by mid_channels for consistency and stride defaults to 1.
        """
        if manual:
            if not stride or not in_channels:
                raise ValueError(
                    "must explicitly specify stride and number of in channels in manual mode")
        else:
            stride = 1
            in_channels = mid_channels * 4

        """
        For Bottleneck, the (third) restoring 1x1 conv layer has a out_channels that is always 4x the out_channels of the (second) 3x3 worker conv layer (i.e. mid_channels).
        """
        out_channels = mid_channels * 4

        """
        The residual path.
        """
        KERNEL_SIZE_IN = 1
        KERNEL_SIZE_MID = 3
        KERNEL_SIZE_OUT = 1
        PADDING_IN = KERNEL_SIZE_IN // 2
        PADDING_MID = KERNEL_SIZE_MID // 2
        PADDING_OUT = KERNEL_SIZE_OUT // 2
        """
        This layer has a stride of 1.
        
        In the original paper seems that this layer is to have a stride of 2, since otherwise the number of parameters can not matched correctly, however the official pytorch pretrained ResNet model uses a stride of 1 in this layer and a stride of 2 in the next worker layer so I chose to believe the latter.
        """
        conv_in = Conv_BN_ReLU(
            in_channels, mid_channels, KERNEL_SIZE_IN, padding=PADDING_IN,
            device=device, dtype=dtype
        )
        """
        This layer has a stride of 2.
        """
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

        """
        The short cut path. Generated only if the shapes of input and output
        is not the same, otherwise the identity mapping is used.
        """
        if in_channels != out_channels or stride != 1:
            """
            Projection mapping.
            """
            KERNEL_SIZE_SHORT_CUT = 1
            PADDING_SHORT_CUT = KERNEL_SIZE_SHORT_CUT // 2
            self.branch_short_cut = Conv_BN_ReLU(
                in_channels, out_channels, KERNEL_SIZE_SHORT_CUT, stride=stride, padding=PADDING_SHORT_CUT,
                need_relu=False,
                device=device, dtype=dtype
            )
        else:
            """
            Identity mapping.
            """
            # self.branch_short_cut = lambda x: x
            self.branch_short_cut = nn.Sequential()

        self.relu_merge = nn.ReLU(inplace=True)

        return

    def forward(self, x):
        out_merge = self.branch_residual(x) + self.branch_short_cut(x)
        out = self.relu_merge(out_merge)
        return out


class BBlockGroup(nn.Module):
    r"""Helper module for grouping some building blocks of the same type together."""
    
    def __init__(self, building_block_name: str, block_group_size: int, in_channels: int, channel_level: int, stride: int, device=None, dtype=None):
        super(BBlockGroup, self).__init__()

        if building_block_name == "normal":
            BB_Class = Normal
        elif building_block_name == "bottleneck":
            BB_Class = Bottleneck
        else:
            raise ValueError("invalid building block name")

        self.seq = nn.Sequential()
        """
        Append the first block.
        
        The first block of this group will be accepting the output of the previous block group or maybe some other module that very likely has a inconsistent shape, so the manual mode is always required for this block.
        """
        conv = BB_Class(channel_level, manual=True, stride=stride, in_channels=in_channels, device=device, dtype=dtype)
        self.seq.append(conv)
        """
        Append the blocks left.
        """
        for _ in range(1, block_group_size):
            conv = BB_Class(channel_level, device=device, dtype=dtype)
            self.seq.append(conv)
        del conv
        
        return

    def forward(self, x):
        return self.seq(x)


class Flatten(nn.Module):
    r"""Helper module for flattening tensors."""
    
    def forward(self, x):
        return x.view(x.shape[0], -1)


class ResNet(nn.Module):
    r"""ResNet model."""
    
    def __init__(self, in_channels: int, building_block_name: str, block_group_sizes: list, num_classes: int, device=None, dtype=None):

        """
        Compute the out_channels depending on the channel_level of a block group.
        """
        def get_out_channels_pre(channel_level):
            if building_block_name == "normal":
                return channel_level
            elif building_block_name == "bottleneck":
                return 4 * channel_level
            else:
                raise ValueError("invalid building block name")
        
        assert len(block_group_sizes) == 4, "must specify the sizes of exactly four block groups"

        super(ResNet, self).__init__()
        
        KERNEL_SIZE_CONV_1 = 7
        PADDING_CONV_1 = KERNEL_SIZE_CONV_1 // 2
        KERNEL_SIZE_MAX_POOL = 3
        PADDING_MAX_POOL = KERNEL_SIZE_MAX_POOL // 2
        """
        Block group Conv1_x.
        """
        channel_level = 64
        conv_1 = Conv_BN_ReLU(in_channels, channel_level, KERNEL_SIZE_CONV_1, stride=2, padding=PADDING_CONV_1, device=device, dtype=dtype)
        out_channels_pre = channel_level
        """
        Block group Conv2_x.
        """
        channel_level *= 1
        conv_2 = nn.Sequential(
            nn.MaxPool2d(KERNEL_SIZE_MAX_POOL, stride=2, padding=PADDING_MAX_POOL),
            BBlockGroup(building_block_name, block_group_sizes[0], out_channels_pre, channel_level, 1, device=device, dtype=dtype)
        )
        out_channels_pre = get_out_channels_pre(channel_level)
        """
        Block group Conv3_x - Conv5_x.
        """
        conv_35 = []
        for i in range(3):
            channel_level *= 2
            conv_i = BBlockGroup(building_block_name, block_group_sizes[i + 1], out_channels_pre, channel_level, 2, device=device, dtype=dtype)
            conv_35.append(conv_i)
            out_channels_pre = get_out_channels_pre(channel_level)
        """
        Global average pooling.
        """
        shape_GAP = (1, 1)
        gap = nn.AdaptiveAvgPool2d(shape_GAP)
        """
        Flattening the output of GAP into a 2-D matrix for FC layer.
        """
        flatten = Flatten()
        """
        FC layer.
        """
        fc = nn.Linear(out_channels_pre, num_classes, device=device, dtype=dtype)
        nn.init.kaiming_normal_(fc.weight)

        """
        For MNIST, the feature map size is already 1x1 right before the last
        global average pooling, so disable it as we want.
        """
        self.seq = nn.Sequential(
            conv_1,
            conv_2,
            *conv_35,
            # gap,
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
