import MinkowskiEngine as ME
import MinkowskiEngine.MinkowskiFunctional as MEF
from MinkowskiEngine.MinkowskiConvolution import MinkowskiConvolution
import torch

from .normalizations import NHNConv3D, NHNConv3DTranspose, BNHNConv3D, BNHNConv3DTranspose


class BlockBase(torch.nn.Module):
    def __init__(self,
                 conv_module,
                 inplanes,
                 planes,
                 stride=1,
                 dilation=1,
                 downsample=None,
                 bn_momentum=0.05):
        super(BlockBase, self).__init__()
        self.conv1 = conv_module(
            inplanes, planes,
            kernel_size=3,
            stride=stride,
            dilation=1,
            use_bias=True,
            bn_momentum=bn_momentum)

        self.conv2 = conv_module(
            planes, planes,
            kernel_size=3,
            stride=1,
            dilation=dilation,
            use_bias=True,
            bn_momentum=bn_momentum)

        self.downsample = downsample

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = MEF.relu(out)

        out = self.conv2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = MEF.relu(out)

        return out


class MinkResUNet(ME.MinkowskiNetwork):
    # To use the model, must call initialize_coords before forward pass.
    # Once data is processed, call clear to reset the model before calling initialize_coords
    def __init__(self,
                 in_channels=1,
                 out_channels=32,
                 down_channels=(None, 32, 64, 128, 256),
                 up_channels=(None, 64, 64, 64, 128),
                 bn_momentum=0.05,
                 pre_conv_num=3,
                 after_pre_channels=1,
                 conv1_kernel_size=None,
                 norm_type=None,
                 upsample_type=None,
                 epsilon=1.0e-8,
                 D=3):
        ME.MinkowskiNetwork.__init__(self, D)

        self.register_buffer('epsilon', torch.tensor(epsilon).float(), persistent=False)
        self.upsample_type = upsample_type.lower()
        self.norm_type = norm_type.lower()

        if self.upsample_type != "pool" and self.upsample_type != "transpose":
            print(f"upsampling type {self.upsample_type} not supported")

        if self.norm_type == "bnhn":
            conv_module = BNHNConv3D
            conv_module_tr = BNHNConv3DTranspose
        elif self.norm_type == "nhn":
            conv_module = NHNConv3D
            conv_module_tr = NHNConv3DTranspose
        else:
            raise AttributeError(f"normalization type {self.norm_type} is not supported")

        self._pre_conv_num = pre_conv_num
        self._pre_conv_list = torch.nn.ModuleList()
        self._pre_conv_list.append(MinkowskiConvolution(
            in_channels=in_channels,
            out_channels=down_channels[1],
            kernel_size=conv1_kernel_size,
            stride=1,
            dilation=1,
            bias=True,
            dimension=D))
        for _ in range(pre_conv_num - 2):
            self._pre_conv_list.append(MinkowskiConvolution(
                in_channels=down_channels[1],
                out_channels=down_channels[1],
                kernel_size=3,
                stride=1,
                dilation=1,
                bias=True,
                dimension=D))
        self._pre_conv_list.append(MinkowskiConvolution(
            in_channels=down_channels[1],
            out_channels=after_pre_channels,
            kernel_size=3,
            stride=1,
            dilation=1,
            bias=True,
            dimension=D))

        self.conv1 = conv_module(
            in_channels=after_pre_channels,
            out_channels=down_channels[1],
            kernel_size=conv1_kernel_size,
            stride=1,
            dilation=1,
            bn_momentum=bn_momentum
        )
        self.block1 = BlockBase(conv_module, down_channels[1], down_channels[1], stride=1, dilation=1,
                                bn_momentum=bn_momentum)

        self.conv2 = conv_module(
            in_channels=down_channels[1],
            out_channels=down_channels[2],
            kernel_size=3,
            stride=2,
            dilation=1,
            bn_momentum=bn_momentum
        )
        self.block2 = BlockBase(conv_module, down_channels[2], down_channels[2], stride=1, dilation=1,
                                bn_momentum=bn_momentum)

        self.conv3 = conv_module(
            in_channels=down_channels[2],
            out_channels=down_channels[3],
            kernel_size=3,
            stride=2,
            dilation=1,
            bn_momentum=bn_momentum
        )
        self.block3 = BlockBase(conv_module, down_channels[3], down_channels[3], stride=1, dilation=1,
                                bn_momentum=bn_momentum)

        self.conv4 = conv_module(
            in_channels=down_channels[3],
            out_channels=down_channels[4],
            kernel_size=3,
            stride=2,
            dilation=1,
            bn_momentum=bn_momentum
        )
        self.block4 = BlockBase(conv_module, down_channels[4], down_channels[4], stride=1, dilation=1,
                                bn_momentum=bn_momentum)

        if self.upsample_type == "pool":
            self.pool4 = ME.MinkowskiPoolingTranspose(kernel_size=2, stride=2, dilation=1, dimension=D)
            self.conv4_tr = conv_module(
                in_channels=down_channels[4],
                out_channels=up_channels[4],
                kernel_size=3,
                stride=1,
                dilation=1,
                bn_momentum=bn_momentum
            )
            self.block4_tr = BlockBase(conv_module, up_channels[4], up_channels[4], stride=1, dilation=1,
                                       bn_momentum=bn_momentum)

            self.pool3 = ME.MinkowskiPoolingTranspose(kernel_size=2, stride=2, dilation=1, dimension=D)
            self.conv3_tr = conv_module(
                in_channels=down_channels[3] + up_channels[4],
                out_channels=up_channels[3],
                kernel_size=3,
                stride=1,
                dilation=1,
                bn_momentum=bn_momentum
            )
            self.block3_tr = BlockBase(conv_module, up_channels[3], up_channels[3], stride=1, dilation=1,
                                       bn_momentum=bn_momentum)

            self.pool2 = ME.MinkowskiPoolingTranspose(kernel_size=2, stride=2, dilation=1, dimension=D)
            self.conv2_tr = conv_module(
                in_channels=down_channels[2] + up_channels[3],
                out_channels=up_channels[2],
                kernel_size=3,
                stride=1,
                dilation=1,
                bn_momentum=bn_momentum
            )
            self.block2_tr = BlockBase(conv_module, up_channels[2], up_channels[2], stride=1, dilation=1,
                                       bn_momentum=bn_momentum)

            self.conv1_tr = MinkowskiConvolution(
                in_channels=down_channels[1] + up_channels[2],
                out_channels=up_channels[1],
                kernel_size=1,
                stride=1,
                dilation=1,
                bias=True,
                dimension=D)

            self.final = MinkowskiConvolution(
                in_channels=up_channels[1],
                out_channels=out_channels,
                kernel_size=1,
                stride=1,
                dilation=1,
                bias=True,
                dimension=D)

        elif self.upsample_type == "transpose":
            self.conv4_tr = conv_module_tr(
                in_channels=down_channels[4],
                out_channels=up_channels[4],
                kernel_size=3,
                stride=2,
                dilation=1,
                bn_momentum=bn_momentum
            )
            self.block4_tr = BlockBase(conv_module, up_channels[4], up_channels[4], stride=1, dilation=1,
                                       bn_momentum=bn_momentum)

            self.conv3_tr = conv_module_tr(
                in_channels=down_channels[3] + up_channels[4],
                out_channels=up_channels[3],
                kernel_size=3,
                stride=2,
                dilation=1,
                bn_momentum=bn_momentum
            )
            self.block3_tr = BlockBase(conv_module, up_channels[3], up_channels[3], stride=1, dilation=1,
                                       bn_momentum=bn_momentum)

            self.conv2_tr = conv_module_tr(
                in_channels=down_channels[2] + up_channels[3],
                out_channels=up_channels[2],
                kernel_size=3,
                stride=2,
                dilation=1,
                bn_momentum=bn_momentum
            )
            self.block2_tr = BlockBase(conv_module, up_channels[2], up_channels[2], stride=1, dilation=1,
                                       bn_momentum=bn_momentum)

            self.conv1_tr = MinkowskiConvolution(
                in_channels=down_channels[1] + up_channels[2],
                out_channels=up_channels[1],
                kernel_size=1,
                stride=1,
                dilation=1,
                bias=True,
                dimension=D)

            self.final = MinkowskiConvolution(
                in_channels=up_channels[1],
                out_channels=out_channels,
                kernel_size=1,
                stride=1,
                dilation=1,
                bias=True,
                dimension=D)

    def forward(self, x):
        for i, pre_conv in enumerate(self._pre_conv_list):
            x = pre_conv(x)
            if i < self._pre_conv_num - 1:
                x = MEF.relu(x)

        out_s1 = self.conv1(x)
        out_s1 = self.block1(out_s1)

        out_s2 = self.conv2(out_s1)
        out_s2 = self.block2(out_s2)

        out_s4 = self.conv3(out_s2)
        out_s4 = self.block3(out_s4)

        out_s8 = self.conv4(out_s4)
        out_s8 = self.block4(out_s8)

        if self.upsample_type == "pool":
            out = self.conv4_tr(out_s8)
            out = self.pool4(out)
            out_s4_tr = self.block4_tr(out)
            out = ME.cat(out_s4_tr, out_s4)

            out = self.conv3_tr(out)
            out = self.pool3(out)
            out_s2_tr = self.block3_tr(out)
            out = ME.cat(out_s2_tr, out_s2)

            out = self.conv2_tr(out)
            out = self.pool2(out)
            out_s1_tr = self.block2_tr(out)
            out = ME.cat(out_s1_tr, out_s1)

            out = self.conv1_tr(out)
            out = MEF.relu(out)
            out = self.final(out)
        elif self.upsample_type == "transpose":
            out = self.conv4_tr(out_s8)
            out_s4_tr = self.block4_tr(out)
            out = ME.cat(out_s4_tr, out_s4)

            out = self.conv3_tr(out)
            out_s2_tr = self.block3_tr(out)
            out = ME.cat(out_s2_tr, out_s2)

            out = self.conv2_tr(out)
            out_s1_tr = self.block2_tr(out)
            out = ME.cat(out_s1_tr, out_s1)

            out = self.conv1_tr(out)
            out = MEF.relu(out)
            out = self.final(out)

        return ME.SparseTensor(
            out.F / (torch.norm(out.F, p=2, dim=1, keepdim=True) + self._buffers['epsilon']),
            coordinate_map_key=out.coordinate_map_key,
            coordinate_manager=out.coordinate_manager)
