import torch
import torch.nn as nn

__all__ = ["Pose2Sign"]


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=dilation,
        groups=groups,
        bias=False,
        dilation=dilation,
    )


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1
    __constants__ = ["downsample"]

    def __init__(
        self,
        inplanes,
        planes,
        stride=1,
        downsample=None,
        groups=1,
        base_width=3,
        dilation=1,
        norm_layer=None,
    ):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 3:
            raise ValueError("BasicBlock only supports groups=1 and base_width=3")
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    def __init__(
        self,
        block,
        layers,
        num_classes=1000,
        zero_init_residual=False,
        groups=1,
        width_per_group=3,
        replace_stride_with_dilation=None,
        norm_layer=None,
        dropout_keep_prob=0.0,
    ):
        super(ResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 3
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError(
                f"replace_stride_with_dilation should be None "
                f"or a 3-element tuple, got {replace_stride_with_dilation}"
            )
        self.groups = groups
        self.base_width = width_per_group
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 3, layers[0])
        self.layer2 = self._make_layer(
            block, 128, layers[1], stride=2, dilate=replace_stride_with_dilation[0]
        )
        self.layer3 = self._make_layer(
            block, 256, layers[2], stride=2, dilate=replace_stride_with_dilation[1]
        )
        self.layer4 = self._make_layer(
            block, 512, layers[3], stride=2, dilate=replace_stride_with_dilation[2]
        )
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)
        self.dropout = nn.Dropout(dropout_keep_prob)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(
            block(
                self.inplanes,
                planes,
                stride,
                downsample,
                self.groups,
                self.base_width,
                previous_dilation,
                norm_layer,
            )
        )
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(
                block(
                    self.inplanes,
                    planes,
                    groups=self.groups,
                    base_width=self.base_width,
                    dilation=self.dilation,
                    norm_layer=norm_layer,
                )
            )

        return nn.Sequential(*layers)

    def _forward_impl(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.fc(x)

        return x

    def forward(self, x):
        return self._forward_impl(x)


class Pose2Sign(nn.Module):
    """ Shallow network that classifies signs with pose series as input
    """

    # def __init__(self, num_classes, input_dim, hidden_dim):
    def __init__(self, num_classes):
        """
        Args:
            input_dim: Integer indicating the input size.
            hidden_dim: List of integers indicating the size of hidden dimensions.
            num_classes: Integer indicating the number of classes.
        """
        super().__init__()

        # layers = []
        # layers.append(nn.Linear(input_dim, hidden_dim[0]))
        # layers.append(nn.LeakyReLU(0.2, inplace=True))
        # for i in range(1, len(hidden_dim)):
        #     layers.append(nn.Linear(hidden_dim[i - 1], hidden_dim[i]))
        #     layers.append(nn.LeakyReLU(0.2, inplace=True))
        # layers.append(nn.Linear(hidden_dim[-1], num_classes))
        # self.classification = nn.Sequential(*layers)
        self.classification = ResNet(
            BasicBlock, [2, 2, 2, 2], num_classes=num_classes, dropout_keep_prob=0.5
        )

    def forward(self, x):
        # x is of shape [batch_size, input_dim]
        # [batch_size, num_classes]
        return self.classification(x)


if __name__ == "__main__":
    # model = Pose2Sign(num_classes=1064, input_dim=3*16*130, hidden_dim=[10])
    model = Pose2Sign(num_classes=1064)
    inp = torch.rand(4, 3, 16, 130)
    out = model(inp)
    # If input 130 pose keypoints (body + hands + face)
    # torch.Size([4, 3, 16, 130])  # 6240
    # torch.Size([4, 3, 16, 130])  # 6240
    # torch.Size([4, 128, 8, 65])  # 66560
    # torch.Size([4, 256, 4, 33])  # 33792
    # torch.Size([4, 512, 2, 17])  # 17408
    # torch.Size([4, 512, 1, 1])   # 512
    # torch.Size([4, 512])         # 512
    # torch.Size([4, 1064])        # 512

    # If input 60 pose keypoints (body + hands)
    # torch.Size([4, 3, 16, 60])   # 2880
    # torch.Size([4, 3, 16, 60])   # 2880
    # torch.Size([4, 128, 8, 30])  # 30720
    # torch.Size([4, 256, 4, 15])  # 15360
    # torch.Size([4, 512, 2, 8])   # 8192
    # torch.Size([4, 512, 1, 1])   # 512
    # torch.Size([4, 512])         # 512
    # torch.Size([4, 1064])        # 1064
    import pdb

    pdb.set_trace()
