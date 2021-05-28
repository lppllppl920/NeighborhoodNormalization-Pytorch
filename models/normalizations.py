import torch
import torch.nn as nn
import MinkowskiEngine as ME
from MinkowskiEngine.MinkowskiConvolution import MinkowskiConvolution, MinkowskiConvolutionTranspose

from .mink_convolutions import CustomMinkowskiConvolution, CustomMinkowskiConvolutionTranspose
from utils import tuplify


class NHNConv3D(torch.nn.Module):
    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size,
            stride=1,
            dilation=1,
            use_bias=True,
            epsilon=1.0e-5,
            bn_momentum=None):
        super(NHNConv3D, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.dilation = dilation
        self.use_bias = use_bias

        self.beta = nn.Parameter(torch.zeros(self.in_channels, dtype=torch.float32), requires_grad=True)
        self.gamma = nn.Parameter(torch.ones(self.in_channels, dtype=torch.float32), requires_grad=True)

        self.full_kernel_size = tuplify(kernel_size, 3)
        self.conv = MinkowskiConvolution(
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            kernel_size=self.kernel_size,
            stride=self.stride,
            bias=False,
            dimension=3)

        epsilon = torch.tensor(epsilon).cuda()
        self.register_buffer('epsilon', epsilon, persistent=False)

        window = (torch.ones(
            1).view(1, 1, 1, 1, 1).expand(1, in_channels, self.full_kernel_size[0], self.full_kernel_size[1],
                                          self.full_kernel_size[2])
                  .contiguous().cuda())
        window = window.reshape(in_channels, 1, -1).permute(2, 0, 1).contiguous()
        self.register_buffer('window', window, persistent=False)

        if self.use_bias:
            self.bias = nn.Parameter(torch.zeros(self.out_channels, dtype=torch.float32), requires_grad=True)

        # B x 1 x D' x H' x W'
        self.average_conv = CustomMinkowskiConvolution(kernel_size=self.kernel_size,
                                                       stride=self.stride, dilation=self.dilation,
                                                       dimension=3)

    def forward(self, x, coords=None):
        # Need to consider the number of occupied voxels in the kernel volume
        # N x 1
        ones = ME.SparseTensor(features=torch.ones_like(x.F),
                               coordinate_map_key=x.coordinate_map_key,
                               coordinate_manager=x.coordinate_manager)
        voxels = self.average_conv(self._buffers['window'], ones, coords)

        # N x 1
        means = self.average_conv(self._buffers['window'], x, coords) / voxels

        # N x 1
        square_means = self.average_conv(self._buffers['window'], x * x, coords) / voxels

        # N x 1
        biased_stds = torch.sqrt(torch.relu(square_means.F - means.F * means.F) + self._buffers['epsilon'])

        voxels_count = torch.clamp_min(voxels.F, min=2)
        stds = ME.SparseTensor(torch.sqrt(voxels_count / (voxels_count - 1)) * biased_stds,
                               coordinate_map_key=square_means.coordinate_map_key,
                               coordinate_manager=square_means.coordinate_manager)

        # N x C_in
        temp = x * self.gamma.view(1, self.in_channels)

        # N x C_out
        conv = self.conv(temp, coords) / stds

        # N x C_out
        gamma_kernel_sum = self.conv(ones * self.gamma.reshape(1, -1), coords)

        # N x C_out
        kernel_weighted_means = gamma_kernel_sum / stds * means.F

        # N x C_out
        beta_kernel_sum = self.conv(ones * self.beta.reshape(1, -1), coords)

        # N x C_out
        x = conv - kernel_weighted_means + beta_kernel_sum

        if self.use_bias:
            x = x + self.bias.view(1, -1)

        return x


class NHNConv3DTranspose(torch.nn.Module):
    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size,
            stride=1,
            dilation=1,
            use_bias=True,
            epsilon=1.0e-5,
            bn_momentum=None):
        super(NHNConv3DTranspose, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.dilation = dilation
        self.use_bias = use_bias

        self.beta = nn.Parameter(torch.zeros(self.in_channels, dtype=torch.float32), requires_grad=True)
        self.gamma = nn.Parameter(torch.ones(self.in_channels, dtype=torch.float32), requires_grad=True)

        self.full_kernel_size = tuplify(kernel_size, 3)
        self.conv = MinkowskiConvolutionTranspose(
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            kernel_size=self.kernel_size,
            stride=self.stride,
            bias=False,
            dimension=3)

        epsilon = torch.tensor(epsilon).cuda()
        self.register_buffer('epsilon', epsilon, persistent=False)

        window = (torch.ones(
            1).view(1, 1, 1, 1, 1).expand(1, in_channels, self.full_kernel_size[0], self.full_kernel_size[1],
                                          self.full_kernel_size[2])
                  .contiguous().cuda())
        window = window.reshape(in_channels, 1, -1).permute(2, 0, 1).contiguous()
        self.register_buffer('window', window, persistent=False)

        if self.use_bias:
            self.bias = nn.Parameter(torch.zeros(self.out_channels, dtype=torch.float32), requires_grad=True)

        # B x 1 x D' x H' x W'
        self.average_conv = CustomMinkowskiConvolutionTranspose(kernel_size=self.kernel_size,
                                                                stride=self.stride, dilation=self.dilation,
                                                                dimension=3)

    def forward(self, x, coords=None):
        # Need to consider the number of occupied voxels in the kernel volume
        # N x 1
        ones = ME.SparseTensor(features=torch.ones_like(x.F),
                               coordinate_map_key=x.coordinate_map_key,
                               coordinate_manager=x.coordinate_manager)
        voxels = self.average_conv(self._buffers['window'], ones, coords)

        # N x 1
        means = self.average_conv(self._buffers['window'], x, coords) / voxels

        # N x 1
        square_means = self.average_conv(self._buffers['window'], x * x, coords) / voxels

        # N x 1
        biased_stds = torch.sqrt(torch.relu(square_means.F - means.F * means.F) + self._buffers['epsilon'])

        voxels_count = torch.clamp_min(voxels.F, min=2)
        stds = ME.SparseTensor(torch.sqrt(voxels_count / (voxels_count - 1)) * biased_stds,
                               coordinate_map_key=square_means.coordinate_map_key,
                               coordinate_manager=square_means.coordinate_manager)

        # N x C_in
        temp = x * self.gamma.view(1, self.in_channels)

        # N x C_out
        conv = self.conv(temp, coords) / stds

        # N x C_out
        gamma_kernel_sum = self.conv(ones * self.gamma.reshape(1, -1), coords)

        # N x C_out
        kernel_weighted_means = means * gamma_kernel_sum / stds

        # N x C_out
        beta_kernel_sum = self.conv(ones * self.beta.reshape(1, -1), coords)

        # N x C_out
        x = conv - kernel_weighted_means + beta_kernel_sum

        if self.use_bias:
            x = x + self.bias.view(1, -1)

        return x


class BNHNConv3D(torch.nn.Module):
    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size,
            stride=1,
            dilation=1,
            use_bias=True,
            epsilon=1.0e-5,
            bn_momentum=0.05):

        super(BNHNConv3D, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.dilation = dilation
        self.use_bias = use_bias
        self.beta = nn.Parameter(torch.zeros(self.in_channels, dtype=torch.float32), requires_grad=True)
        self.gamma = nn.Parameter(torch.ones(self.in_channels, dtype=torch.float32), requires_grad=True)

        self.full_kernel_size = tuplify(kernel_size, 3)
        self.conv = MinkowskiConvolution(
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            kernel_size=self.kernel_size,
            stride=self.stride,
            bias=False,
            dimension=3)

        window = (torch.ones(
            1).view(1, 1, 1, 1, 1).expand(1, in_channels, self.full_kernel_size[0], self.full_kernel_size[1],
                                          self.full_kernel_size[2])
                  .contiguous().cuda())
        window = window.reshape(in_channels, 1, -1).permute(2, 0, 1).contiguous()
        self.register_buffer('window', window, persistent=False)

        if self.use_bias:
            self.bias = nn.Parameter(torch.zeros(self.out_channels, dtype=torch.float32), requires_grad=True)

        self.average_conv = CustomMinkowskiConvolution(kernel_size=self.kernel_size,
                                                       stride=self.stride, dilation=self.dilation,
                                                       dimension=3)

        self.mean_att = torch.nn.Parameter(0.5 * torch.ones(1, 1).float(), requires_grad=True)
        self.std_att = torch.nn.Parameter(0.5 * torch.ones(1, 1).float(), requires_grad=True)

        self.bn_momentum = bn_momentum

        self.register_buffer("moving_avg", torch.zeros(self.in_channels, dtype=torch.float32), persistent=True)
        self.register_buffer("moving_var", torch.ones(self.in_channels, dtype=torch.float32), persistent=True)

        self.register_buffer("one", torch.ones(1, dtype=torch.float32), persistent=False)

        epsilon = torch.tensor(epsilon).cuda()
        self.register_buffer('epsilon', epsilon, persistent=True)

    def forward(self, x, coords=None):
        # N_out x 1
        ones = ME.SparseTensor(features=torch.ones_like(x.F),
                               coordinate_map_key=x.coordinate_map_key,
                               coordinate_manager=x.coordinate_manager)
        voxels = self.average_conv(self._buffers['window'], ones, coords)

        # N_out x 1
        means = self.average_conv(self._buffers['window'], x, coords) / voxels

        # N_out x 1
        square_means = self.average_conv(self._buffers['window'], x * x, coords) / voxels

        # N_out x 1
        voxels_count = torch.clamp_min(voxels.F, min=2)
        stds = torch.sqrt(
            voxels_count / (voxels_count - 1) * torch.relu(square_means.F - means.F * means.F) + self._buffers[
                'epsilon'])

        if self.training:
            # Calculate channel-wise batch norm statistics
            bn_mean = torch.mean(x.F, dim=0).reshape(self.in_channels, )
            bn_var = torch.var(x.F, dim=0, unbiased=True).reshape(self.in_channels, )
            bn_std = torch.sqrt(bn_var + self._buffers['epsilon']).reshape(1, self.in_channels)
            with torch.no_grad():
                self._buffers['moving_var'] = self._buffers['moving_var'] * (
                        1 - self.bn_momentum) + bn_var * self.bn_momentum
                self._buffers['moving_avg'] = self._buffers['moving_avg'] * (
                        1 - self.bn_momentum) + bn_mean * self.bn_momentum
        else:
            bn_mean = self._buffers['moving_avg']
            bn_var = self._buffers['moving_var']
            bn_std = torch.sqrt(bn_var + self._buffers['epsilon']).reshape(1, self.in_channels)

        mean_att = self.mean_att
        std_att = self.std_att

        # 1 x 1 or 1 x C_out
        alpha_1 = mean_att[:, 0:1].reshape(1, -1)
        # 1 x 1
        alpha_2 = std_att[:, 0:1].reshape(1, -1)

        # 1 x 1 or 1 x C_out
        alpha_1_oppo = 1.0 - alpha_1
        # 1 x 1
        alpha_2_oppo = 1.0 - alpha_2

        # Here we use weighted geometric mean instead of arithmetic mean for weighting nhn std and bn std
        # N_out x C_out
        beta_term = self.conv(ones * self.beta.reshape(1, self.in_channels), coords)

        # N_out x C_out
        gamma_term_2 = self.conv(ones * self.gamma.reshape(1, self.in_channels) / bn_std.pow(alpha_2), coords) * \
                       (means.F * (-alpha_1_oppo) / stds.pow(alpha_2_oppo))

        if alpha_1.shape[1] != 1:
            # N_out x C_out
            gamma_term_1 = self.conv(ones * (self.gamma.reshape(1, self.in_channels)
                                             * bn_mean.reshape(1, self.in_channels)) /
                                     bn_std.pow(alpha_2), coords) * (-alpha_1 / stds.pow(alpha_2_oppo))

            # N_out x C_out
            x_term = self.conv(x * self.gamma.reshape(1, self.in_channels) / bn_std.pow(alpha_2), coords) * \
                     (1.0 / stds.pow(alpha_2_oppo))
            gamma_1_x_term = gamma_term_1 + x_term
        else:
            gamma_1_x_term = self.conv((x - bn_mean.reshape(1, self.in_channels) * alpha_1) *
                                       (self.gamma.reshape(1, self.in_channels) / bn_std.pow(alpha_2)), coords) * \
                             (1.0 / stds.pow(alpha_2_oppo))
        temp = gamma_1_x_term + gamma_term_2 + beta_term
        return temp


class BNHNConv3DTranspose(torch.nn.Module):
    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size,
            stride=1,
            dilation=1,
            use_bias=True,
            epsilon=1.0e-5,
            bn_momentum=0.05):

        super(BNHNConv3DTranspose, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.dilation = dilation
        self.use_bias = use_bias
        self.beta = nn.Parameter(torch.zeros(self.in_channels, dtype=torch.float32), requires_grad=True)
        self.gamma = nn.Parameter(torch.ones(self.in_channels, dtype=torch.float32), requires_grad=True)

        self.full_kernel_size = tuplify(kernel_size, 3)
        self.conv = MinkowskiConvolutionTranspose(
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            kernel_size=self.kernel_size,
            stride=self.stride,
            bias=False,
            dimension=3)

        window = (torch.ones(
            1).view(1, 1, 1, 1, 1).expand(1, in_channels, self.full_kernel_size[0], self.full_kernel_size[1],
                                          self.full_kernel_size[2])
                  .contiguous().cuda())
        window = window.reshape(in_channels, 1, -1).permute(2, 0, 1).contiguous()
        self.register_buffer('window', window, persistent=False)

        if self.use_bias:
            self.bias = nn.Parameter(torch.zeros(self.out_channels, dtype=torch.float32), requires_grad=True)

        # B x 1 x D' x H' x W'
        self.average_conv = CustomMinkowskiConvolutionTranspose(kernel_size=self.kernel_size,
                                                                stride=self.stride, dilation=self.dilation,
                                                                dimension=3)

        self.mean_att = torch.nn.Parameter(0.5 * torch.ones(1, 1).float(), requires_grad=True)
        self.std_att = torch.nn.Parameter(0.5 * torch.ones(1, 1).float(), requires_grad=True)

        self.bn_momentum = bn_momentum

        self.register_buffer("moving_avg", torch.zeros(self.in_channels, dtype=torch.float32), persistent=True)
        self.register_buffer("moving_var", torch.ones(self.in_channels, dtype=torch.float32), persistent=True)

        self.register_buffer("one", torch.ones(1, dtype=torch.float32), persistent=False)
        epsilon = torch.tensor(epsilon).cuda()
        self.register_buffer('epsilon', epsilon, persistent=True)

    def forward(self, x, coords=None):
        # N_out x 1
        ones = ME.SparseTensor(features=torch.ones_like(x.F),
                               coordinate_map_key=x.coordinate_map_key,
                               coordinate_manager=x.coordinate_manager)
        voxels = self.average_conv(self._buffers['window'], ones, coords)

        # N_out x 1
        means = self.average_conv(self._buffers['window'], x, coords) / voxels

        # N_out x 1
        square_means = self.average_conv(self._buffers['window'], x * x, coords) / voxels

        # N_out x 1
        voxels_count = torch.clamp_min(voxels.F, min=2)
        stds = torch.sqrt(
            voxels_count / (voxels_count - 1) * torch.relu(square_means.F - means.F * means.F) + self._buffers[
                'epsilon'])

        if self.training:
            # Calculate channel-wise batch norm statistics
            bn_mean = torch.mean(x.F, dim=0).reshape(self.in_channels, )
            bn_var = torch.var(x.F, dim=0, unbiased=True).reshape(self.in_channels, )
            bn_std = torch.sqrt(bn_var + self._buffers['epsilon']).reshape(1, self.in_channels)
            with torch.no_grad():
                self._buffers['moving_var'] = self._buffers['moving_var'] * (
                        1 - self.bn_momentum) + bn_var * self.bn_momentum
                self._buffers['moving_avg'] = self._buffers['moving_avg'] * (
                        1 - self.bn_momentum) + bn_mean * self.bn_momentum
        else:
            if torch.any(torch.isinf(self._buffers['moving_var'])):
                print("wtf")
                self._buffers['moving_var'] = \
                    torch.where(torch.isinf(self._buffers['moving_var']), self.one, self._buffers['moving_var'])
            bn_mean = self._buffers['moving_avg']
            bn_var = self._buffers['moving_var']
            bn_std = torch.sqrt(bn_var + self._buffers['epsilon']).reshape(1, self.in_channels)

        mean_att = self.mean_att
        std_att = self.std_att

        # 1 x 1 or 1 x C_out
        alpha_1 = mean_att[:, 0:1].reshape(1, -1)
        # 1 x 1
        alpha_2 = std_att[:, 0:1].reshape(1, -1)

        # 1 x 1 or 1 x C_out
        alpha_1_oppo = 1.0 - alpha_1
        # 1 x 1
        alpha_2_oppo = 1.0 - alpha_2

        # Here we use weighted geometric mean instead of arithmetic mean for weighting patch std and bn std
        # N_out x C_out
        beta_term = self.conv(ones * self.beta.reshape(1, self.in_channels), coords)

        # N_out x C_out
        gamma_term_2 = self.conv(ones * self.gamma.reshape(1, self.in_channels) / bn_std.pow(alpha_2), coords) * \
                       (means.F * (-alpha_1_oppo) / stds.pow(alpha_2_oppo))

        if alpha_1.shape[1] != 1:
            # N_out x C_out
            gamma_term_1 = self.conv(ones * (self.gamma.reshape(1, self.in_channels)
                                             * bn_mean.reshape(1, self.in_channels)) /
                                     bn_std.pow(alpha_2), coords) * (-alpha_1 / stds.pow(alpha_2_oppo))

            # N_out x C_out
            x_term = self.conv(x * self.gamma.reshape(1, self.in_channels) / bn_std.pow(alpha_2), coords) * \
                     (1.0 / stds.pow(alpha_2_oppo))
            gamma_1_x_term = gamma_term_1 + x_term
        else:
            gamma_1_x_term = self.conv((x - bn_mean.reshape(1, self.in_channels) * alpha_1) *
                                       (self.gamma.reshape(1, self.in_channels) / bn_std.pow(alpha_2)), coords) * \
                             (1.0 / stds.pow(alpha_2_oppo))

        return gamma_1_x_term + gamma_term_2 + beta_term
