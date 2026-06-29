'''by lyuwenyu
'''
import torch
import torch.nn as nn
import torch.nn.functional as F

from collections import OrderedDict
from pathlib import Path

from .common import get_activation, ConvNormLayer, FrozenBatchNorm2d

from src.core import register


__all__ = ['PResNet']


ResNet_cfg = {
    18: [2, 2, 2, 2],
    34: [3, 4, 6, 3],
    50: [3, 4, 6, 3],
    101: [3, 4, 23, 3],
    # 152: [3, 8, 36, 3],
}


donwload_url = {
    18: 'https://github.com/lyuwenyu/storage/releases/download/v0.1/ResNet18_vd_pretrained_from_paddle.pth',
    34: 'https://github.com/lyuwenyu/storage/releases/download/v0.1/ResNet34_vd_pretrained_from_paddle.pth',
    50: 'https://github.com/lyuwenyu/storage/releases/download/v0.1/ResNet50_vd_ssld_v2_pretrained_from_paddle.pth',
    101: 'https://github.com/lyuwenyu/storage/releases/download/v0.1/ResNet101_vd_ssld_pretrained_from_paddle.pth',
}

# Local-first pretrained weights. This keeps the original pretrained=True logic
# while avoiding repeated downloads if the local file exists.
local_pretrained_path = {
    18: Path(r'D:\Learn\RTDETR\RT-DETR-main\PretrainingWeight\ResNet18_vd_pretrained_from_paddle.pth'),
}


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, ch_in, ch_out, stride, shortcut, act='relu', variant='b'):
        super().__init__()

        self.shortcut = shortcut

        if not shortcut:
            if variant == 'd' and stride == 2:
                self.short = nn.Sequential(OrderedDict([
                    ('pool', nn.AvgPool2d(2, 2, 0, ceil_mode=True)),
                    ('conv', ConvNormLayer(ch_in, ch_out, 1, 1))
                ]))
            else:
                self.short = ConvNormLayer(ch_in, ch_out, 1, stride)

        self.branch2a = ConvNormLayer(ch_in, ch_out, 3, stride, act=act)
        self.branch2b = ConvNormLayer(ch_out, ch_out, 3, 1, act=None)
        self.act = nn.Identity() if act is None else get_activation(act)

    def forward(self, x):
        out = self.branch2a(x)
        out = self.branch2b(out)
        if self.shortcut:
            short = x
        else:
            short = self.short(x)

        out = out + short
        out = self.act(out)

        return out


class BottleNeck(nn.Module):
    expansion = 4

    def __init__(self, ch_in, ch_out, stride, shortcut, act='relu', variant='b'):
        super().__init__()

        if variant == 'a':
            stride1, stride2 = stride, 1
        else:
            stride1, stride2 = 1, stride

        width = ch_out

        self.branch2a = ConvNormLayer(ch_in, width, 1, stride1, act=act)
        self.branch2b = ConvNormLayer(width, width, 3, stride2, act=act)
        self.branch2c = ConvNormLayer(width, ch_out * self.expansion, 1, 1)

        self.shortcut = shortcut
        if not shortcut:
            if variant == 'd' and stride == 2:
                self.short = nn.Sequential(OrderedDict([
                    ('pool', nn.AvgPool2d(2, 2, 0, ceil_mode=True)),
                    ('conv', ConvNormLayer(ch_in, ch_out * self.expansion, 1, 1))
                ]))
            else:
                self.short = ConvNormLayer(ch_in, ch_out * self.expansion, 1, stride)

        self.act = nn.Identity() if act is None else get_activation(act)

    def forward(self, x):
        out = self.branch2a(x)
        out = self.branch2b(out)
        out = self.branch2c(out)

        if self.shortcut:
            short = x
        else:
            short = self.short(x)

        out = out + short
        out = self.act(out)

        return out


class PartialConv3(nn.Module):
    """PConv used by Faster-Block.

    Only a fraction of channels are processed by a 3x3 convolution, while the
    remaining channels are directly propagated. The default n_div=4 follows
    the paper setting r=1/4.
    """
    def __init__(self, channels, n_div=4):
        super().__init__()
        if n_div <= 1:
            raise ValueError('n_div should be greater than 1 for PartialConv3.')
        dim_conv = channels // n_div
        if dim_conv < 1:
            raise ValueError(f'channels={channels} is too small for n_div={n_div}.')
        self.dim_conv = dim_conv
        self.dim_untouched = channels - dim_conv
        self.partial_conv3 = nn.Conv2d(
            dim_conv, dim_conv, kernel_size=3, stride=1, padding=1, bias=False
        )

    def forward(self, x):
        x_conv, x_untouched = torch.split(
            x, [self.dim_conv, self.dim_untouched], dim=1
        )
        x_conv = self.partial_conv3(x_conv)
        return torch.cat((x_conv, x_untouched), dim=1)


class FasterBlock(nn.Module):
    """Faster-Block adapted from FasterNet for RT-DETR-R18 backbone.

    Structure follows the paper idea:
    PConv 3x3 -> 1x1 PWConv -> BN + ReLU -> 1x1 PWConv -> shortcut add.

    This implementation supports stage transition. When stride=2 or channel
    number changes, the input is first projected to the target shape, then the
    Faster-Block operates at the target resolution/channels.
    """
    expansion = 1

    def __init__(
        self,
        ch_in,
        ch_out,
        stride,
        shortcut,
        act='relu',
        variant='b',
        n_div=4,
        mlp_ratio=2.0,
    ):
        super().__init__()

        self.need_proj = (stride != 1) or (ch_in != ch_out)
        if self.need_proj:
            if variant == 'd' and stride == 2:
                self.proj = nn.Sequential(OrderedDict([
                    ('pool', nn.AvgPool2d(2, 2, 0, ceil_mode=True)),
                    ('conv', ConvNormLayer(ch_in, ch_out, 1, 1, act=None))
                ]))
            else:
                self.proj = ConvNormLayer(ch_in, ch_out, 1, stride, act=None)
        else:
            self.proj = nn.Identity()

        hidden_dim = int(ch_out * mlp_ratio)
        hidden_dim = max(hidden_dim, ch_out)

        self.spatial_mixing = PartialConv3(ch_out, n_div=n_div)
        self.pwconv1 = nn.Conv2d(ch_out, hidden_dim, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(hidden_dim)
        self.act = nn.Identity() if act is None else get_activation(act)
        self.pwconv2 = nn.Conv2d(hidden_dim, ch_out, kernel_size=1, bias=False)

    def forward(self, x):
        identity = self.proj(x)

        out = self.spatial_mixing(identity)
        out = self.pwconv1(out)
        out = self.bn(out)
        out = self.act(out)
        out = self.pwconv2(out)

        return identity + out


class Blocks(nn.Module):
    def __init__(
        self,
        block,
        ch_in,
        ch_out,
        count,
        stage_num,
        act='relu',
        variant='b',
        use_fasterblock=False,
        fasterblock_replace_mode='last',
        fasterblock_n_div=4,
        fasterblock_mlp_ratio=2.0,
    ):
        super().__init__()

        self.blocks = nn.ModuleList()
        for i in range(count):
            if fasterblock_replace_mode == 'all':
                replace_this_block = use_fasterblock
            elif fasterblock_replace_mode == 'last':
                replace_this_block = use_fasterblock and (i == count - 1)
            elif fasterblock_replace_mode == 'first':
                replace_this_block = use_fasterblock and (i == 0)
            elif fasterblock_replace_mode in ('none', None):
                replace_this_block = False
            else:
                raise ValueError(f'Unsupported fasterblock_replace_mode: {fasterblock_replace_mode}')

            cur_block = FasterBlock if replace_this_block else block

            common_kwargs = dict(
                ch_in=ch_in,
                ch_out=ch_out,
                stride=2 if i == 0 and stage_num != 2 else 1,
                shortcut=False if i == 0 else True,
                variant=variant,
                act=act,
            )

            if cur_block is FasterBlock:
                self.blocks.append(
                    cur_block(
                        **common_kwargs,
                        n_div=fasterblock_n_div,
                        mlp_ratio=fasterblock_mlp_ratio,
                    )
                )
            else:
                self.blocks.append(cur_block(**common_kwargs))

            if i == 0:
                ch_in = ch_out * cur_block.expansion

    def forward(self, x):
        out = x
        for block in self.blocks:
            out = block(out)
        return out


@register
class PResNet(nn.Module):
    def __init__(
        self,
        depth,
        variant='d',
        num_stages=4,
        return_idx=[0, 1, 2, 3],
        act='relu',
        freeze_at=-1,
        freeze_norm=True,
        pretrained=False,
        use_fasterblock=False,
        fasterblock_replace_stages=None,
        fasterblock_replace_mode='last',
        fasterblock_n_div=4,
        fasterblock_mlp_ratio=2.0,
    ):
        super().__init__()

        block_nums = ResNet_cfg[depth]
        ch_in = 64
        if variant in ['c', 'd']:
            conv_def = [
                [3, ch_in // 2, 3, 2, "conv1_1"],
                [ch_in // 2, ch_in // 2, 3, 1, "conv1_2"],
                [ch_in // 2, ch_in, 3, 1, "conv1_3"],
            ]
        else:
            conv_def = [[3, ch_in, 7, 2, "conv1_1"]]

        self.conv1 = nn.Sequential(OrderedDict([
            (_name, ConvNormLayer(c_in, c_out, k, s, act=act)) for c_in, c_out, k, s, _name in conv_def
        ]))

        ch_out_list = [64, 128, 256, 512]

        if fasterblock_replace_stages is None:
            fasterblock_replace_stages = []
        fasterblock_replace_stages = set(fasterblock_replace_stages)

        if use_fasterblock and depth >= 50:
            raise ValueError('use_fasterblock=True is currently designed for ResNet18/34 style BasicBlock backbones.')

        block = BottleNeck if depth >= 50 else BasicBlock
        if use_fasterblock:
            print(
                f'Use selective FasterBlock backbone: stages={sorted(fasterblock_replace_stages)}, '
                f'mode={fasterblock_replace_mode}, n_div={fasterblock_n_div}, mlp_ratio={fasterblock_mlp_ratio}'
            )

        _out_channels = [block.expansion * v for v in ch_out_list]
        _out_strides = [4, 8, 16, 32]

        self.res_layers = nn.ModuleList()
        for i in range(num_stages):
            stage_num = i + 2
            self.res_layers.append(
                Blocks(
                    block,
                    ch_in,
                    ch_out_list[i],
                    block_nums[i],
                    stage_num,
                    act=act,
                    variant=variant,
                    use_fasterblock=use_fasterblock and (i in fasterblock_replace_stages),
                    fasterblock_replace_mode=fasterblock_replace_mode,
                    fasterblock_n_div=fasterblock_n_div,
                    fasterblock_mlp_ratio=fasterblock_mlp_ratio,
                )
            )
            ch_in = _out_channels[i]

        self.return_idx = return_idx
        self.out_channels = [_out_channels[_i] for _i in return_idx]
        self.out_strides = [_out_strides[_i] for _i in return_idx]
        self.use_fasterblock = use_fasterblock
        self.fasterblock_replace_stages = fasterblock_replace_stages
        self.fasterblock_replace_mode = fasterblock_replace_mode

        if freeze_at >= 0:
            self._freeze_parameters(self.conv1)
            for i in range(min(freeze_at, num_stages)):
                self._freeze_parameters(self.res_layers[i])

        if freeze_norm:
            self._freeze_norm(self)

        if pretrained:
            local_path = local_pretrained_path.get(depth, None)
            if local_path is None:
                raise FileNotFoundError(
                    f'Local PResNet{depth} pretrained weight path is not configured. '
                    'Please configure local_pretrained_path or set pretrained=False.'
                )
            if not local_path.exists():
                raise FileNotFoundError(
                    f'Local PResNet{depth} pretrained weight not found: {local_path}. '
                    'Please place the pretrained weight at this path or set pretrained=False.'
                )
            print(f'Load PResNet{depth} local pretrained weight: {local_path}')
            state = torch.load(local_path, map_location='cpu')

            # Compatible with common checkpoint wrappers.
            if isinstance(state, dict):
                if 'state_dict' in state:
                    state = state['state_dict']
                elif 'model' in state:
                    state = state['model']

            if self.use_fasterblock:
                missing_keys, unexpected_keys = self.load_state_dict(state, strict=False)
                print(
                    f'Load PResNet{depth} pretrained state_dict with FasterBlock '
                    f'(strict=False): missing={len(missing_keys)}, unexpected={len(unexpected_keys)}'
                )
                if missing_keys:
                    print(f'  missing examples: {missing_keys[:8]}')
                if unexpected_keys:
                    print(f'  unexpected examples: {unexpected_keys[:8]}')
            else:
                self.load_state_dict(state)
                print(f'Load PResNet{depth} state_dict')

    def _freeze_parameters(self, m: nn.Module):
        for p in m.parameters():
            p.requires_grad = False

    def _freeze_norm(self, m: nn.Module):
        if isinstance(m, nn.BatchNorm2d):
            m = FrozenBatchNorm2d(m.num_features)
        else:
            for name, child in m.named_children():
                _child = self._freeze_norm(child)
                if _child is not child:
                    setattr(m, name, _child)
        return m

    def forward(self, x):
        conv1 = self.conv1(x)
        x = F.max_pool2d(conv1, kernel_size=3, stride=2, padding=1)
        outs = []
        for idx, stage in enumerate(self.res_layers):
            x = stage(x)
            if idx in self.return_idx:
                outs.append(x)
        return outs
