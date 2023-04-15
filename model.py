class ConvOffset2D(nn.Conv2d):
    """ConvOffset2D

    Convolutional layer responsible for learning the 2D offsets and output the
    deformed feature map using bilinear interpolation

    Note that this layer does not perform convolution on the deformed feature
    map. See get_deform_cnn in cnn.py for usage
    """
    def __init__(self, filters, init_normal_stddev=0.01, **kwargs):
        """Init

        Parameters
        ----------
        filters : int
            Number of channel of the input feature map
        init_normal_stddev : float
            Normal kernel initialization
        **kwargs:
            Pass to superclass. See Con2d layer in pytorch
        """
        self.filters = filters
        self._grid_param = None
        super(ConvOffset2D, self).__init__(self.filters, self.filters*2, 3, padding=1, bias=False, **kwargs)
        self.weight.data.copy_(self._init_weights(self.weight, init_normal_stddev))

    def forward(self, x):
        """Return the deformed featured map"""
        x_shape = x.size()
        offsets = super(ConvOffset2D, self).forward(x)

        # offsets: (b*c, h, w, 2)
        offsets = self._to_bc_h_w_2(offsets, x_shape)

        # x: (b*c, h, w)
        x = self._to_bc_h_w(x, x_shape)

        # X_offset: (b*c, h, w)
        x_offset = th_batch_map_offsets(x, offsets, grid=self._get_grid(self,x))

        # x_offset: (b, h, w, c)
        x_offset = self._to_b_c_h_w(x_offset, x_shape)
        # print("ok")
        return x_offset

    @staticmethod
    def _get_grid(self, x):
        batch_size, input_height, input_width = x.size(0), x.size(1), x.size(2)
        dtype, cuda = x.data.type(), x.data.is_cuda
        if self._grid_param == (batch_size, input_height, input_width, dtype, cuda):
            return self._grid
        self._grid_param = (batch_size, input_height, input_width, dtype, cuda)
        self._grid = th_generate_grid(batch_size, input_height, input_width, dtype, cuda)
        return self._grid

    @staticmethod
    def _init_weights(weights, std):
        fan_out = weights.size(0)
        fan_in = weights.size(1) * weights.size(2) * weights.size(3)
        w = np.random.normal(0.0, std, (fan_out, fan_in))
        return torch.from_numpy(w.reshape(weights.size()))

    @staticmethod
    def _to_bc_h_w_2(x, x_shape):
        """(b, 2c, h, w) -> (b*c, h, w, 2)"""
        x = x.contiguous().view(-1, int(x_shape[2]), int(x_shape[3]), 2)
        return x

    @staticmethod
    def _to_bc_h_w(x, x_shape):
        """(b, c, h, w) -> (b*c, h, w)"""
        x = x.contiguous().view(-1, int(x_shape[2]), int(x_shape[3]))
        return x

    @staticmethod
    def _to_b_c_h_w(x, x_shape):
        """(b*c, h, w) -> (b, c, h, w)"""
        x = x.contiguous().view(-1, int(x_shape[1]), int(x_shape[2]), int(x_shape[3]))
        return x


def th_generate_grid(batch_size, input_height, input_width, dtype, cuda):
    grid = np.meshgrid(
        range(input_height), range(input_width), indexing='ij'
    )
    grid = np.stack(grid, axis=-1)
    grid = grid.reshape(-1, 2)

    grid = np_repeat_2d(grid, batch_size)
    grid = torch.from_numpy(grid).type(dtype)
    if cuda:
        grid = grid.cuda()
    return Variable(grid, requires_grad=False)
def np_repeat_2d(a, repeats):
    """Tensorflow version of np.repeat for 2D"""

    assert len(a.shape) == 2
    a = np.expand_dims(a, 0)
    a = np.tile(a, [repeats, 1, 1])
    return a
def th_batch_map_offsets(input, offsets, grid=None, order=1):
    """Batch map offsets into input
    Parameters
    ---------
    input : torch.Tensor. shape = (b, s, s)
    offsets: torch.Tensor. shape = (b, s, s, 2)
    Returns
    -------
    torch.Tensor. shape = (b, s, s)
    """
    batch_size = input.size(0)
    input_height = input.size(1)
    input_width = input.size(2)

    offsets = offsets.view(batch_size, -1, 2)
    if grid is None:
        grid = th_generate_grid(batch_size, input_height, input_width, offsets.data.type(), offsets.data.is_cuda)

    coords = offsets + grid

    mapped_vals = th_batch_map_coordinates(input, coords)
    return mapped_vals
def th_repeat(a, repeats, axis=0):
    """Torch version of np.repeat for 1D"""
    assert len(a.size()) == 1
    return th_flatten(torch.transpose(a.repeat(repeats, 1), 0, 1))
def th_flatten(a):
    """Flatten tensor"""
    return a.contiguous().view(a.nelement())
def th_batch_map_coordinates(input, coords, order=1):
    """Batch version of th_map_coordinates
    Only supports 2D feature maps
    Parameters
    ----------
    input : tf.Tensor. shape = (b, s, s)
    coords : tf.Tensor. shape = (b, n_points, 2)
    Returns
    -------
    tf.Tensor. shape = (b, s, s)
    """

    batch_size = input.size(0)
    input_height = input.size(1)
    input_width = input.size(2)

    n_coords = coords.size(1)
    coords = torch.cat((torch.clamp(coords.narrow(2, 0, 1), 0, input_height - 1), torch.clamp(coords.narrow(2, 1, 1), 0, input_width - 1)), 2)

    assert (coords.size(1) == n_coords)

    coords_lt = coords.floor().long()
    coords_rb = coords.ceil().long()
    coords_lb = torch.stack([coords_lt[..., 0], coords_rb[..., 1]], 2)
    coords_rt = torch.stack([coords_rb[..., 0], coords_lt[..., 1]], 2)
    idx = th_repeat(torch.arange(0, batch_size), n_coords).long()
    idx = Variable(idx, requires_grad=False)
    if input.is_cuda:
        idx = idx.cuda()

    def _get_vals_by_coords(input, coords):
        indices = torch.stack([
            idx, th_flatten(coords[..., 0]), th_flatten(coords[..., 1])
        ], 1)
        inds = indices[:, 0]*input.size(1)*input.size(2)+ indices[:, 1]*input.size(2) + indices[:, 2]
        vals = th_flatten(input).index_select(0, inds)
        vals = vals.view(batch_size, n_coords)
        return vals

    vals_lt = _get_vals_by_coords(input, coords_lt.detach())
    vals_rb = _get_vals_by_coords(input, coords_rb.detach())
    vals_lb = _get_vals_by_coords(input, coords_lb.detach())
    vals_rt = _get_vals_by_coords(input, coords_rt.detach())

    coords_offset_lt = coords - coords_lt.type(coords.data.type())
    vals_t = coords_offset_lt[..., 0]*(vals_rt - vals_lt) + vals_lt
    vals_b = coords_offset_lt[..., 0]*(vals_rb - vals_lb) + vals_lb
    mapped_vals = coords_offset_lt[..., 1]* (vals_b - vals_t) + vals_t
    return mapped_vals


class BasicBlock(nn.Module):

    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, groups=1, bias=False):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size, stride, padding, groups=groups, bias=bias)
        self.bn1 = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_planes, out_planes, kernel_size, 1, padding, groups=groups, bias=bias)
        self.bn2 = nn.BatchNorm2d(out_planes)
        self.downsample = None
        if stride > 1:
            self.downsample = nn.Sequential(nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False),
                            nn.BatchNorm2d(out_planes),)

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Encoder(nn.Module):

    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, groups=1, bias=False):
        super(Encoder, self).__init__()
        self.block1 = BasicBlock(in_planes, out_planes, kernel_size, stride, padding, groups, bias)
        self.block2 = BasicBlock(out_planes, out_planes, kernel_size, 1, padding, groups, bias)

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)

        return x


class Decoder(nn.Module):

    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, output_padding=0, groups=1, bias=False):

        super(Decoder, self).__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(in_planes, in_planes//4, 1, 1, 0, bias=bias),
                                nn.BatchNorm2d(in_planes//4),
                                nn.ReLU(inplace=True),)
        self.tp_conv = nn.Sequential(nn.ConvTranspose2d(in_planes//4, in_planes//4, kernel_size, stride, padding, output_padding, bias=bias),
                                nn.BatchNorm2d(in_planes//4),
                                nn.ReLU(inplace=True),)
        self.conv2 = nn.Sequential(nn.Conv2d(in_planes//4, out_planes, 1, 1, 0, bias=bias),
                                nn.BatchNorm2d(out_planes),
                                nn.ReLU(inplace=True),)

    def forward(self, x):
        x = self.conv1(x)
        x = self.tp_conv(x)
        x = self.conv2(x)

        return x

class Change_channel(nn.Module):
    """
    降通道   in_planes, out_planes,
    """
    def __init__(self, in_planes, out_planes, bias=False,relu = False):
        super(Change_channel, self).__init__()

        self.change_channel = nn.Sequential(nn.Conv2d(in_planes, out_planes, 1, bias=bias),
                                            nn.BatchNorm2d(out_planes),
                                             # nn.ReLU(inplace=True)
                                            )
    def forward(self, x):
        x = self.change_channel(x)
        return x

class C(nn.Module):
    '''
    This class is for a convolutional layer.
    '''

    def __init__(self, nIn, nOut, kSize, stride=1):
        '''

        :param nIn: number of input channels
        :param nOut: number of output channels
        :param kSize: kernel size
        :param stride: optional stride rate for down-sampling
        '''
        super().__init__()
        padding = int((kSize - 1) / 2)
        self.conv = nn.Conv2d(nIn, nOut, (kSize, kSize), stride=stride, padding=(padding, padding), bias=False)

    def forward(self, input):
        '''
        :param input: input feature map
        :return: transformed feature map
        '''
        output = self.conv(input)
        return output

class BR(nn.Module):
    '''
        This class groups the batch normalization and PReLU activation
    '''

    def __init__(self, nOut):
        '''
        :param nOut: output feature maps
        '''
        super().__init__()
        self.bn = nn.BatchNorm2d(nOut, eps=1e-03)
        self.act = nn.PReLU(nOut)
    

    def forward(self, input):
        '''
        :param input: input feature map
        :return: normalized and thresholded feature map
        '''
        output = self.bn(input)
        output = self.act(output)
        return output

class C3block(nn.Module):
    '''
    This class defines the dilated convolution.
    '''

    def __init__(self, nIn, nOut, kSize, stride=1, d=1):
        '''
        :param nIn: number of input channels
        :param nOut: number of output channels
        :param kSize: kernel size
        :param stride: optional stride rate for down-sampling
        :param d: optional dilation rate
        '''
        super().__init__()
        padding = int((kSize - 1) / 2) * d
        if d == 1:
            self.conv = nn.Sequential(
                nn.Conv2d(nIn, nIn, (kSize, kSize), stride=stride, padding=(padding, padding), groups=nIn, bias=False,
                          dilation=d),
                nn.Conv2d(nIn, nOut, kernel_size=1, stride=1, bias=False)
            )
        else:
            combine_kernel = 2 * d - 1

            self.conv = nn.Sequential(

                nn.Conv2d(nIn, nIn, kernel_size=(combine_kernel, 1), stride=stride, padding=(padding - 1, 0),
                          groups=nIn, bias=False),
                nn.BatchNorm2d(nIn),
                nn.PReLU(nIn),


                nn.Conv2d(nIn, nIn, kernel_size=(1, combine_kernel), stride=stride, padding=(0, padding - 1),
                          groups=nIn, bias=False),
                nn.BatchNorm2d(nIn),


                nn.Conv2d(nIn, nIn, (kSize, kSize), stride=stride, padding=(padding, padding), groups=nIn, bias=False,
                          dilation=d),
                nn.Conv2d(nIn, nOut, kernel_size=1, stride=1, bias=False))

    def forward(self, input):
        '''
        :param input: input feature map
        :return: transformed feature map
        '''
        output = self.conv(input)
        return output


class Down_advancedC3(nn.Module):
    def __init__(self, nIn, nOut, ratio=[2,4,8]):
        super().__init__()
        n = int(nOut // 3)
        n1 = nOut - 3 * n
        self.c1 = C(nIn, n, 3, 2)

        self.d1 = C3block(n, n+n1, 3, 1, ratio[0])
        self.d2 = C3block(n, n, 3, 1, ratio[1])
        self.d3 = C3block(n, n, 3, 1, ratio[2])

        self.bn = nn.BatchNorm2d(nOut, eps=1e-3)
        self.act = nn.PReLU(nOut)

    def forward(self, input):
        output1 = self.c1(input)
        d1 = self.d1(output1)
        d2 = self.d2(output1)
        d3 = self.d3(output1)

        combine = torch.cat([d1, d2, d3], 1)

        output = self.bn(combine)
        output = self.act(output)
        return output

class AdvancedC3(nn.Module):
    '''
    This class defines the ESP block, which is based on the following principle
        Reduce ---> Split ---> Transform --> Merge
    '''

    def __init__(self, nIn, nOut, add=True, ratio=[2,4,8]):
        '''
        :param nIn: number of input channels
        :param nOut: number of output channels
        :param add: if true, add a residual connection through identity operation. You can use projection too as
                in ResNet paper, but we avoid to use it if the dimensions are not the same because we do not want to
                increase the module complexity
        '''
        super().__init__()
        n = int(nOut // 3)
        n1 = nOut - 3 * n
        self.c1 = C(nIn, n, 1, 1)

        self.d1 = C3block(n, n + n1, 3, 1, ratio[0])
        self.d2 = C3block(n, n, 3, 1, ratio[1])
        self.d3 = C3block(n, n, 3, 1, ratio[2])

        self.bn = BR(nOut)
        self.add = add

    def forward(self, input):
        '''
        :param input: input feature map
        :return: transformed feature map
        '''
        # reduce
        output1 = self.c1(input)
        d1 = self.d1(output1)
        d2 = self.d2(output1)
        d3 = self.d3(output1)

        combine = torch.cat([d1, d2, d3], 1)

        if self.add:
            combine = input + combine
        output = self.bn(combine)
        return output

class GlobalContextBlock(nn.Module):
    # def __init__(self, inplanes = 64, ratio = 0.25, pooling_type='att', fusion_types=('channel_add', )):
    def __init__(self, inplanes = 64, ratio = 0.25):
        super(GlobalContextBlock, self).__init__()

        self.inplanes = inplanes
        self.ratio = ratio
        self.planes = int(inplanes * ratio)

        # 'att'
        self.conv_mask = nn.Conv2d(inplanes, 1, kernel_size=1)
        self.softmax = nn.Softmax(dim=2)

        # 'channel_add'
        self.Transform = nn.Sequential(
            nn.Conv2d(self.inplanes, self.planes, kernel_size=1),
            nn.LayerNorm([self.planes, 1, 1]),
            nn.ReLU(inplace=True),  # yapf: disable
            nn.Conv2d(self.planes, self.inplanes, kernel_size=1))

    def ContextModel(self, x):
        batch, channel, height, width = x.size()

        input_x = x
        # [N, C, H * W]
        input_x = input_x.view(batch, channel, height * width)
        # [N, 1, C, H * W]
        input_x = input_x.unsqueeze(1)
        # [N, 1, H, W]
        context_mask = self.conv_mask(x)
        # [N, 1, H * W]
        context_mask = context_mask.view(batch, 1, height * width)
        # [N, 1, H * W]
        context_mask = self.softmax(context_mask)
        # [N, 1, H * W, 1]
        context_mask = context_mask.unsqueeze(-1)
        # [N, 1, C, 1]
        context = torch.matmul(input_x, context_mask)
        # [N, C, 1, 1]
        context = context.view(batch, channel, 1, 1)

        return context

    def forward(self, x):
        # [N, C, 1, 1]
        context = self.ContextModel(x)
        out = x
        if self.Transform is not None:
            # [N, C, 1, 1]
            Transform = self.Transform(context)
            out = out + Transform

        return out

class GhostModule(nn.Module):
    def __init__(self, inp, oup ):
        super(GhostModule, self).__init__()

        self.primary_conv = nn.Sequential( nn.Conv2d(inp, inp/2, 1, bias=False),
                                           nn.BatchNorm2d(inp/2),
                                           nn.ReLU(inplace=True)
                                           )

        self.cheap_operation = nn.Sequential( nn.Conv2d(inp/2, oup, 3, 1, 1, groups=inp/2, bias=False),
                                              nn.BatchNorm2d(oup),
                                              nn.ReLU(inplace=True)
                                              )

    def forward(self, x):
        x1 = self.primary_conv(x)
        x2 = self.cheap_operation(x1)
        out = torch.cat([x1, x2], dim=1)

        return out

class StripPooling(nn.Module):
    """
    Reference:
    """
    def __init__(self, in_channels = 64, norm_layer =nn.BatchNorm2d):
        super(StripPooling, self).__init__()

        self.pool_h = nn.AdaptiveAvgPool2d((1, None))

        inter_channels = int(in_channels/4)
        self.low_channel = nn.Sequential(nn.Conv2d(in_channels, inter_channels, 1, bias=False),
                                         norm_layer(inter_channels),
                                         nn.ReLU(True))

        self.conv_h = nn.Sequential(nn.Conv2d(inter_channels, inter_channels, (1, 3), 1, (0, 1), bias=False),  norm_layer(inter_channels))

        self.refine_relu = nn.Sequential(nn.Conv2d(inter_channels, inter_channels, 3, 1, 1, bias=False),  norm_layer(inter_channels),nn.ReLU(True))
        self.high_channel = nn.Sequential(nn.Conv2d(inter_channels*2, in_channels, 1, bias=False), norm_layer(in_channels))

    def forward(self, x):
        _, _, h, w = x.size()
        low_channel = self.low_channel(x)      
        strip_h_V = F.interpolate(self.conv_h(self.pool_h(low_channel)), (h, w), mode="bilinear", align_corners=True)

        out = self.high_channel(torch.cat([low_channel , strip_h_V], dim=1))

        return  x * out


# region GALD
class GlobalContextBlock_repair(nn.Module):
    def __init__(self, inplanes = 64, ratio = 0.25):
        super(GlobalContextBlock_repair, self).__init__()

        self.inplanes = inplanes
        self.ratio = ratio
        self.planes = int(inplanes * ratio)

        # 'att'
        self.conv_mask = nn.Conv2d(inplanes, 1, kernel_size=1)
        self.softmax = nn.Softmax(dim=2)

        # 'channel_add'
        self.Transform = nn.Sequential(
            nn.Conv2d(self.inplanes, self.planes, kernel_size=1),
            nn.LayerNorm([self.planes, 1, 1]),
            nn.ReLU(inplace=True),  # yapf: disable
            nn.Conv2d(self.planes, self.inplanes, kernel_size=1))

    def ContextModel(self, x):
        batch, channel, height, width = x.size()

        input_x = x
        # [N, C, H * W]
        input_x = input_x.view(batch, channel, height * width)
        # [N, 1, C, H * W]
        input_x = input_x.unsqueeze(1)
        # [N, 1, H, W]
        context_mask = self.conv_mask(x)
        # [N, 1, H * W]
        context_mask = context_mask.view(batch, 1, height * width)
        # [N, 1, H * W]
        context_mask = self.softmax(context_mask)
        # [N, 1, H * W, 1]
        context_mask = context_mask.unsqueeze(-1)
        # [N, 1, C, 1]
        context = torch.matmul(input_x, context_mask)
        # [N, C, 1, 1]
        context = context.view(batch, channel, 1, 1)

        return context

    def forward(self, x):

        context = self.ContextModel(x)
        out = x
        if self.Transform is not None:

            Transform = self.Transform(context)
            out = out + Transform

        return out

class LocalAttenModule_repair(nn.Module):
    def __init__(self, inplane):
        super(LocalAttenModule_repair, self).__init__()
        self.dconv = AdvancedC3(inplane, inplane, ratio=[2, 4, 8])
        self.sigmoid_spatial = nn.Sigmoid()

    def forward(self, x):
        res1 = x
        res2 = x
        x = self.dconv(x)
        x_mask = self.sigmoid_spatial(x)
        res1 = res1 * x_mask

        return res2 + res1

class GALDBlock_repair(nn.Module):
    def __init__(self, inplane):
        super(GALDBlock_repair, self).__init__()
        self.GA = GlobalContextBlock_repair(inplane)
        self.LD = LocalAttenModule_repair(inplane)

    def forward(self, x):
        GA = self.GA(x)
        LD = self.LD(x)
        return LD + GA

class LinkNet_C3_4block_SCNN_offset_GHost(nn.Module):
    """
    designed for (h,w) 1024,2048
    """
    def __init__(self, model_id, project_dir,ms_ks,status = "train", n_classes=2):
        """
        Model initialization
        :param x_n: number of input neurons
        :type x_n: int
        """
        super(LinkNet_C3_4block_SCNN_offset_GHost, self).__init__()

        self.status = status
        self.model_id = model_id
        self.project_dir = project_dir
        self.create_model_dirs(status=self.status)

        self.net_init(ms_ks = 9)

        self.conv1 = nn.Conv2d(3, 64, 7, 2, 3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(3, 2, 1)

    
        self.encoder1 = nn.Sequential(AdvancedC3(64, 64, ratio=[1, 3, 4]),
                                      AdvancedC3(64, 64, ratio=[1, 3, 4]),
           
                                      )
        self.encoder2 = nn.Sequential(Down_advancedC3(64, 128, ratio=[1, 3, 5]),
                                      AdvancedC3(128, 128, ratio=[1, 3, 5]),
                                
                                      )
        self.encoder3 = nn.Sequential(Down_advancedC3(128, 256, ratio=[2, 4, 8]),
                                      AdvancedC3(256, 256, ratio=[2, 4, 8]))
        self.offset_256 = ConvOffset2D(256)

        self.encoder4 = nn.Sequential(Down_advancedC3(256, 512, ratio=[2, 4, 8]),
                                      AdvancedC3(512, 512, ratio=[2, 4, 8]))

        self.channel_512_64 =Change_channel(512,64)
        self.channel_512_128 =Change_channel(512,128)

        self.cheap_operation = nn.Sequential(nn.Conv2d(64, 64, 3, 1, 1, groups=64, bias=False),
                                             nn.BatchNorm2d(64),
                                             nn.ReLU(inplace=True)
                                             )

        self.channel_64_512 =Change_channel(64,512)
        self.channel_128_512 =Change_channel(128,512)
        self.channel_256_512 =Change_channel(256,512)

        self.decoder1 = Decoder(64, 64, 3, 1, 1, 0)
        self.decoder2 = Decoder(128, 64, 3, 2, 1, 1)
        self.decoder3 = Decoder(256, 128, 3, 2, 1, 1)
        self.decoder4 = Decoder(512, 256, 3, 2, 1, 1)

        # Classifier
        self.dropout = nn.Dropout2d(0.5)
        self.refine = nn.Sequential(nn.Conv2d(64, 32, 1),
                                    nn.BatchNorm2d(32),
                                    nn.ReLU(inplace=True), )
        self.dropout = nn.Dropout2d(0.5)
        self.fc = nn.Sequential(nn.Conv2d(32, n_classes, 1))

        self.lsm = nn.LogSoftmax(dim=1)

    def forward(self, x):
        # print("x.size(): ",x.size())
        h,w = x.size()[-2:]
        # Initial block
        x = self.conv1(x)
 

        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)



        # Encoder blocks
        e1 = self.encoder1(x)
        e2 = self.encoder2(e1)
        e3 = self.encoder3(e2)

        e3_offset = self.offset_256(e3)
        e4 = self.encoder4(e3_offset)
       

        # GHost_SCNN
        channel_512_64 = self.channel_512_64(e4)


        primary_conv = self.message_passing_forward(channel_512_64)           # SCNN
        cheap_operation1 = self.cheap_operation(primary_conv)
        cat_1 = torch.cat([primary_conv, cheap_operation1], dim=1)
       
        channel_128_512 = self.channel_128_512(cat_1)
       
        e4 = channel_128_512

     

        # Decoder blocks
        d4 = e3_offset + self.decoder4(e4)
        d3 = e2 + self.decoder3(d4)
        d2 = e1 + self.decoder2(d3)
        d1 = x + self.decoder1(d2)
        

        # Classifier
        y = self.refine(d1)
        

        y = self.dropout(y)
        y = self.fc(y)

        y = F.interpolate(y, (h, w), mode="bilinear", align_corners=True)
        y = self.lsm(y)


        return y

    def net_init(self, ms_ks):

        channel = 64
        self.message_passing = nn.ModuleList()
        self.message_passing.add_module('up_down',
                                        nn.Conv2d(channel, channel, (1, ms_ks), padding=(0, ms_ks // 2), bias=False))
        self.message_passing.add_module('down_up',
                                        nn.Conv2d(channel, channel, (1, ms_ks), padding=(0, ms_ks // 2), bias=False))
        self.message_passing.add_module('left_right',
                                        nn.Conv2d(channel, channel, (ms_ks, 1), padding=(ms_ks // 2, 0), bias=False))
        self.message_passing.add_module('right_left',
                                        nn.Conv2d(channel, channel, (ms_ks, 1), padding=(ms_ks // 2, 0), bias=False))


    def message_passing_forward(self, x):
        Vertical = [True, True, False, False]
        Reverse = [ False,True, False, True]

        for ms_conv, v, r in zip(self.message_passing, Vertical, Reverse):
            x = self.message_passing_once(x, ms_conv, v, r)
        return x

    def message_passing_once(self, x, conv, vertical=True, reverse=False):
        nB, C, H, W = x.shape

        if vertical:
          
            slices = [x[:, :, i:(i + 1), :] for i in range(H)]       
            dim = 2
        else:
            slices = [x[:, :, :, i:(i + 1)] for i in range(W)]         
            dim = 3
        if reverse:
            slices = slices[::-1]                  

        out = [slices[0]]
        for i in range(1, len(slices)):
            out.append(slices[i] + F.relu(conv(out[i - 1])))
        if reverse:
            out = out[::-1]
        return torch.cat(out, dim=dim)

    def create_model_dirs(self,status = "train"):        

        # train
        self.logs_trainmodel_dir = self.project_dir + "/datasets/training_logs/model"
        self.model_dir = self.logs_trainmodel_dir + "/" + self.model_id
        self.model_logger = self.model_dir + "/Logs"
        self.checkpoints_dir = self.model_dir + "/weightfiles"

        #test
        self.logs_test_dir = self.project_dir + "/datasets/training_logs/pred"
        self.test_dir = self.logs_test_dir +"/" + self.model_id
        self.test_logger = self.test_dir + "/Logs"


        if status == "train":  
            if not os.path.exists(self.logs_trainmodel_dir):
                os.makedirs(self.logs_trainmodel_dir)

            if not os.path.exists(self.model_dir):
                os.makedirs(self.model_dir)
                os.makedirs(self.model_logger)
                os.makedirs(self.checkpoints_dir)

        if status == "test":  

            if not os.path.exists(self.logs_test_dir):
                os.makedirs(self.logs_test_dir)

            if not os.path.exists(self.test_dir):
                os.makedirs(self.test_dir)
                os.makedirs(self.test_logger)