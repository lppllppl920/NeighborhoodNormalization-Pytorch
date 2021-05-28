import torch
import MinkowskiEngine as ME
from MinkowskiEngine.MinkowskiSparseTensor import SparseTensor, _get_coordinate_map_key
from MinkowskiEngine.MinkowskiKernelGenerator import KernelGenerator
from MinkowskiEngine.MinkowskiCommon import convert_to_int_tensor
from MinkowskiEngine.MinkowskiConvolution import MinkowskiConvolutionFunction, MinkowskiConvolutionTransposeFunction
from MinkowskiEngineBackend._C import ConvolutionMode
from typing import Union


class CustomMinkowskiConvolutionTranspose(torch.nn.Module):
    def __init__(self, kernel_size, stride,
                 dilation, dimension, kernel_generator=None):
        super(CustomMinkowskiConvolutionTranspose, self).__init__()

        if kernel_generator is None:
            kernel_generator = KernelGenerator(
                kernel_size=kernel_size,
                stride=stride,
                dilation=dilation,
                dimension=dimension,
                is_transpose=True)
        else:
            kernel_size = kernel_generator.kernel_size

        stride = convert_to_int_tensor(stride, dimension)
        kernel_size = convert_to_int_tensor(kernel_size, dimension)
        dilation = convert_to_int_tensor(dilation, dimension)

        kernel_volume = kernel_generator.kernel_volume

        self.kernel_size = kernel_size
        self.kernel_volume = kernel_volume
        self.stride = stride
        self.dilation = dilation
        self.kernel_generator = kernel_generator
        self.dimension = dimension
        self.use_mm = False  # use matrix multiplication when kernel is 1

        if torch.prod(kernel_size) == 1 and torch.prod(stride) == 1:
            self.use_mm = True

        self.region_type_ = None
        self.region_offset_ = None

    def forward(self, kernel, input: SparseTensor,
                coords: Union[torch.IntTensor, ME.CoordinateMapKey, SparseTensor] = None):
        assert input.D == self.dimension

        # # Create a region_offset
        # self.region_type_, self.region_offset_, _ = \
        #     self.kernel_generator.get_kernel(input.tensor_stride, True)

        # Get a new coords key or extract one from the coords
        conv_transpose = MinkowskiConvolutionTransposeFunction()
        if self.use_mm and coords is None:
            # If the kernel_size == 1, the convolution is simply a matrix
            # multiplication
            if kernel.dim() == 3:
                kernel = kernel.reshape(kernel.shape[1], kernel.shape[2])
            outfeat = input.F.mm(kernel)
            out_coords_key = input.coordinate_map_key
        else:
            out_coords_key = _get_coordinate_map_key(input, coords, tensor_stride=1)

            '''
                    input_features: torch.Tensor,
        kernel_weights: torch.Tensor,
        kernel_generator: KernelGenerator,
        convolution_mode: ConvolutionMode,
        in_coordinate_map_key: CoordinateMapKey,
        out_coordinate_map_key: CoordinateMapKey = None,
        coordinate_manager: CoordinateManager = None,
            '''
            # input.tensor_stride,
            #                                            self.stride, self.kernel_size, self.dilation,
            #                                            self.region_type_, self.region_offset_,
            #                                            False,  # generate_new_coords
            outfeat = conv_transpose.apply(input.F, kernel,
                                           self.kernel_generator,
                                           ConvolutionMode(0),
                                           input.coordinate_map_key, out_coords_key,
                                           input.coordinate_manager)

        return SparseTensor(
            outfeat, coordinate_map_key=out_coords_key,
            coordinate_manager=input.coordinate_manager)


class CustomMinkowskiConvolution(torch.nn.Module):
    def __init__(self, kernel_size, stride,
                 dilation, dimension, kernel_generator=None):
        super(CustomMinkowskiConvolution, self).__init__()

        if kernel_generator is None:
            kernel_generator = KernelGenerator(
                kernel_size=kernel_size,
                stride=stride,
                dilation=dilation,
                dimension=dimension)
        else:
            kernel_size = kernel_generator.kernel_size

        stride = convert_to_int_tensor(stride, dimension)
        kernel_size = convert_to_int_tensor(kernel_size, dimension)
        dilation = convert_to_int_tensor(dilation, dimension)

        kernel_volume = kernel_generator.kernel_volume

        # kernel size: volume x in_channel x out_channel
        self.kernel_size = kernel_size
        self.kernel_volume = kernel_volume
        self.stride = stride
        self.dilation = dilation
        self.kernel_generator = kernel_generator
        self.dimension = dimension
        self.use_mm = False  # use matrix multiplication when kernel is 1

        if torch.prod(kernel_size) == 1 and torch.prod(stride) == 1:
            self.use_mm = True

        self.region_type_ = None
        self.region_offset_ = None

    def forward(self, kernel, input: SparseTensor,
                coords: Union[torch.IntTensor, ME.CoordinateMapKey, SparseTensor] = None):
        assert input.D == self.dimension

        # Create a region_offset
        self.region_type_, self.region_offset_, _ = \
            self.kernel_generator.get_kernel(input.tensor_stride, False)

        # Get a new coords key or extract one from the coords
        out_coords_key = _get_coordinate_map_key(input, coords)
        conv = MinkowskiConvolutionFunction()
        if self.use_mm and coords is None:
            # If the kernel_size == 1, the convolution is simply a matrix
            # multiplication
            kernel = kernel.reshape(kernel.shape[1], kernel.shape[2])
            outfeat = input.F.mm(kernel)
            out_coords_key = input.coordinate_map_key
        else:
            outfeat = conv.apply(input.F, kernel,
                                 self.kernel_generator,
                                 ConvolutionMode(0),
                                 input.coordinate_map_key,
                                 out_coords_key,
                                 input.coordinate_manager)

        return SparseTensor(
            outfeat, coordinate_map_key=out_coords_key,
            coordinate_manager=input.coordinate_manager)


class CustomMinkowskiChannelwiseConvolution(torch.nn.Module):
    def __init__(self,
                 kernel_size=-1,
                 stride=1,
                 dilation=1,
                 kernel_generator=None,
                 dimension=-1):
        r"""convolution on a sparse tensor

        Args:
            :attr:`in_channels` (int): the number of input channels in the
            input tensor.

            :attr:`kernel_size` (int, optional): the size of the kernel in the
            output tensor. If not provided, :attr:`region_offset` should be
            :attr:`RegionType.CUSTOM` and :attr:`region_offset` should be a 2D
            matrix with size :math:`N\times D` such that it lists all :math:`N`
            offsets in D-dimension.

            :attr:`stride` (int, or list, optional): stride size of the
            convolution layer. If non-identity is used, the output coordinates
            will be at least :attr:`stride` :math:`\times` :attr:`tensor_stride`
            away. When a list is given, the length must be D; each element will
            be used for stride size for the specific axis.

            :attr:`dilation` (int, or list, optional): dilation size for the
            convolution kernel. When a list is given, the length must be D and
            each element is an axis specific dilation. All elements must be > 0.

            :attr:`has_bias` (bool, optional): if True, the convolution layer
            has a bias.

            :attr:`kernel_generator` (:attr:`MinkowskiEngine.KernelGenerator`,
            optional): defines the custom kernel shape.

            :attr:`dimension` (int): the spatial dimension of the space where
            all the inputs and the network are defined. For example, images are
            in a 2D space, meshes and 3D shapes are in a 3D space.

        """

        super(CustomMinkowskiChannelwiseConvolution, self).__init__()
        assert dimension > 0, f"dimension must be a positive integer, {dimension}"

        if kernel_generator is None:
            kernel_generator = KernelGenerator(
                kernel_size=kernel_size,
                stride=stride,
                dilation=dilation,
                dimension=dimension)
        else:
            kernel_size = kernel_generator.kernel_size

        stride = convert_to_int_tensor(stride, dimension)
        kernel_size = convert_to_int_tensor(kernel_size, dimension)
        dilation = convert_to_int_tensor(dilation, dimension)

        kernel_volume = kernel_generator.kernel_volume

        self.kernel_size = kernel_size
        self.kernel_volume = kernel_volume
        self.stride = stride
        self.dilation = dilation
        self.kernel_generator = kernel_generator
        self.dimension = dimension

    def forward(self,
                kernel,
                input: SparseTensor):
        r"""
        :attr:`input` (`MinkowskiEngine.SparseTensor`): Input sparse tensor to apply a
        convolution on.

        :attr:`coords` ((`torch.IntTensor`, `MinkowskiEngine.CoordsKey`,
        `MinkowskiEngine.SparseTensor`), optional): If provided, generate
        results on the provided coordinates. None by default.

        """
        assert ((kernel.dim() == 2) or (kernel.dim() == 3))
        assert input.D == self.dimension

        # Create a region_offset
        self.region_type_, self.region_offset_, _ = \
            self.kernel_generator.get_kernel(input.tensor_stride, False)

        cm = input.coordinate_manager
        in_key = input.coordinate_map_key
        on_gpu = input.device.type != 'cpu'

        out_key = cm.stride(in_key, self.stride)
        N_out = cm.get_coords_size_by_coords_key(out_key)
        tensor_stride = convert_to_int_tensor(input.tensor_stride, self.dimension)

        in_maps, out_maps = cm.get_kernel_map(
            in_key,
            out_key,
            self.stride,
            self.kernel_size,
            self.dilation,
            self.region_type_,
            self.region_offset_,
            is_transpose=False,
            is_pool=False,
            on_gpu=on_gpu)

        # Get the discrete output coordinate values so that we can look up the associated kernel values
        output_coords = cm.get_coords(out_key)
        index_array = torch.flip(torch.arange(self.dimension), dims=(0,))
        index_array = torch.pow(self.kernel_size[0], index_array)

        in_feats = input.F
        if input.F.dim() == 2:
            in_feats = in_feats.unsqueeze(dim=-1)

        if self.kernel_size.prod() == 1 and kernel.dim() == 2:
            kernel = kernel.unsqueeze(dim=0)
        elif self.kernel_size.prod() != 1 and kernel.dim() == 2:
            raise ValueError("wtf")

        out_F = input._F.new(N_out, *kernel.shape[1:]).zero_()

        for k in range(len(in_maps)):
            kernel_coord = (input.C[in_maps[k], 1:] - output_coords[out_maps[k], 1:]) // \
                           tensor_stride.reshape(1, -1) + (
                                   self.kernel_size.reshape(1, -1) - 1) // 2
            kernel_1d_coord = torch.sum(index_array.reshape(1, -1) * kernel_coord, dim=1, keepdim=False)
            out_F[out_maps[k]] += in_feats[in_maps[k]] * kernel[kernel_1d_coord]

        return SparseTensor(out_F.reshape(N_out, -1),
                            coordinate_map_key=out_key,
                            coordinate_manager=cm)
