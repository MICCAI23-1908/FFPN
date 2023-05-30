import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
import inspect
import re
import sys
from typing import Union, List, Tuple, Any, Dict as TDict, Iterator, Type, Callable

class block(nn.Module):
    """
    Bottomneck:
    定义一个残差块类，方便ResNet调用
        包含：
            conv，
            BatchNorm2d（BN层改变数据分布，加快训练收敛速度），
            relu,
            identity_DownSample,
    """
    def __init__(
            self,in_channels,intermediate_channels,identity_downsample=None,stride=1
    ):
        # 重新继承父类nn.Module的init方法
        super(block,self).__init__()
        self.expansion = 4
        self.conv1 = nn.Conv2d(
            in_channels,
            intermediate_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False
        )
        self.bn1 = nn.BatchNorm2d(intermediate_channels)
        self.conv2 = nn.Conv2d(
            intermediate_channels,
            intermediate_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False
        )
        self.bn2 = nn.BatchNorm2d(intermediate_channels)
        self.conv3 = nn.Conv2d(
            intermediate_channels,
            intermediate_channels * self.expansion,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False
        )
        self.bn3 = nn.BatchNorm2d(intermediate_channels * self.expansion)
        # 定义ReLU激活函数
        self.relu = nn.ReLU(inplace=True)
        # 特征下采样
        self.identity_downsample = identity_downsample
        # 步幅
        self.stride = stride

    def forward(self,x):
        identity = x.clone()

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.conv3(x)
        x = self.bn3(x)


        if self.identity_downsample is not None:
            identity = self.identity_downsample(identity)
        # 残差连接，将没有经过这个残差块之前的特征加到这里，弥补这个过程中卷积导致的梯度消失
        x += identity
        x = self.relu(x)
        return x

class FPN(nn.Module):
    def __init__(self,block, layers, out_channels):
        super(FPN,self).__init__()
        self.in_planes = 64
        # self.in_channels = 64
        self.out_channels = out_channels
        self.conv1 = nn.Conv2d(3,64,kernel_size=7,stride=2,padding=3,bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        # Bottom-up layers
        self.layer1 = self._make_layer(
            block, layers[0], planes=64, stride=1
        )
        self.layer2 = self._make_layer(
            block, layers[1], planes=128, stride=2
        )
        self.layer3 = self._make_layer(
            block, layers[2], planes=256, stride=2
        )
        self.layer4 = self._make_layer(
            block, layers[3], planes=512, stride=2
        )
        # Top layer
        self.toplayer = nn.Sequential(nn.Conv2d(2048,256,kernel_size=1,stride=1,padding=0),nn.BatchNorm2d(256),nn.ReLU(inplace=True))

        # Smooth layers
        self.smooth1 = nn.Sequential(nn.Conv2d(256,256,kernel_size=3,stride=1,padding=1),nn.BatchNorm2d(256),nn.ReLU(inplace=True))
        self.smooth2 = nn.Sequential(nn.Conv2d(256,256,kernel_size=3,stride=1,padding=1),nn.BatchNorm2d(256),nn.ReLU(inplace=True))
        self.smooth3 = nn.Sequential(nn.Conv2d(256,256,kernel_size=3,stride=1,padding=1),nn.BatchNorm2d(256),nn.ReLU(inplace=True))

        # Lateral layers
        self.latlayer1 = nn.Sequential(nn.Conv2d(1024,256,kernel_size=1,stride=1,padding=0),nn.BatchNorm2d(256),nn.ReLU(inplace=True))
        self.latlayer2 = nn.Sequential(nn.Conv2d(512, 256,kernel_size=1,stride=1,padding=0),nn.BatchNorm2d(256),nn.ReLU(inplace=True))
        self.latlayer3 = nn.Sequential(nn.Conv2d(256, 256,kernel_size=1,stride=1,padding=0),nn.BatchNorm2d(256),nn.ReLU(inplace=True))

    def _make_layer(self,block,num_residual_blocks,planes,stride):
        identity_downsample = None
        layers = []

        if stride != 1 or self.in_planes != planes * 4:
            identity_downsample = nn.Sequential(
                nn.Conv2d(
                    self.in_planes,
                    planes * 4,
                    kernel_size=1,
                    stride=stride,
                    bias=False
                ),
                nn.BatchNorm2d(planes * 4),
            )
        layers.append(
            block(self.in_planes,planes,identity_downsample,stride)
        )
        self.in_planes = planes * 4
        for i in range(num_residual_blocks - 1):
            layers.append(block(self.in_planes,planes))

        return nn.Sequential(*layers)
    

    def _upsample_add(self,x,y):
        _,_,H,W = y.size()
        return F.interpolate(x,size=(H,W),mode='nearest') + y
    def forward(self,x):
        # Bottom-up
        c1 = self.relu(self.bn1(self.conv1(x)))
        c1 = self.maxpool(c1)
        c2 = self.layer1(c1)
        c3 = self.layer2(c2)
        c4 = self.layer3(c3)
        c5 = self.layer4(c4)

        # Top-down
        p5 = self.toplayer(c5)
        p4 = self._upsample_add(p5, self.latlayer1(c4))
        p3 = self._upsample_add(p4, self.latlayer2(c3))
        p2 = self._upsample_add(p3, self.latlayer3(c2))

        # Smooth
        p4 = self.smooth1(p4)
        p3 = self.smooth2(p3)
        p2 = self.smooth3(p2)

        return p2, p3, p4, p5

class ResNet(nn.Module):
    """
    残差网络由一个卷积层，一个池化层，以及4个嵌套了block的大层，最后加一个1*1卷积层组成；
    不同的ResNet网络，区别就是深度，改变的就是4个大层中block数量，ResNet-50和ResNet-101区别就在于第四层的block数量
    """
    def __init__(self,block,layers,image_channels,num_classes):
        super(ResNet,self).__init__()
        self.in_channels = 64
        self.conv1 = nn.Conv2d(
            image_channels,64,kernel_size=7,stride=2,padding=3,bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=3,stride=2,padding=1)
        self.layer1 = self._make_layer(
            block,layers[0],intermediate_channels=64,stride=1
        )
        self.layer2 = self._make_layer(
            block, layers[1], intermediate_channels=128, stride=2
        )
        self.layer3 = self._make_layer(
            block, layers[2], intermediate_channels=256, stride=2
        )
        self.layer4 = self._make_layer_dcn(
            block, layers[3], intermediate_channels=512, stride=2
        )
        # 平均池化层
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        # 用全连接层将特征向量连接到分类数
        self.fc = nn.Linear(512*4,num_classes)

    def forward(self,x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        # 调用init中定义好的layer层，即调用_make_layer函数，（1）传入block类，（2）layers列表中第一个元素，即这一层需要放多少个block，（3）通道数，（4）步长
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.reshape(x.shape[0], -1)
        x = self.fc(x)

        return x

    def _make_layer(self,block,num_residual_blocks,intermediate_channels,stride):
        identity_downsample = None
        layers = []
        if stride != 1 or self.in_channels != intermediate_channels * 4:
            identity_downsample = nn.Sequential(
                nn.Conv2d(
                    self.in_channels,
                    intermediate_channels * 4,
                    kernel_size=1,
                    stride=stride,
                    bias=False
                ),
                nn.BatchNorm2d(intermediate_channels * 4),
            )
        layers.append(
            block(self.in_channels,intermediate_channels,identity_downsample,stride)
        )
        self.in_channels = intermediate_channels * 4
        for i in range(num_residual_blocks - 1):
            layers.append(block(self.in_channels,intermediate_channels))

        return nn.Sequential(*layers)


def replace_ndim(s: Union[str, type, Callable], dim: int, allowed_dims=(1, 2, 3)):
    """Replace ndim.

    Replaces dimension statement of ``string``or ``type``.

    Notes:
        - Dimensions are expected to be at the end of the type name.
        - If there is no dimension statement, nothing is changed.

    Examples:
        >>> replace_ndim('BatchNorm2d', 3)
        'BatchNorm3d'
        >>> replace_ndim(nn.BatchNorm2d, 3)
        torch.nn.modules.batchnorm.BatchNorm3d
        >>> replace_ndim(nn.GroupNorm, 3)
        torch.nn.modules.normalization.GroupNorm
        >>> replace_ndim(F.conv2d, 3)
        <function torch._VariableFunctionsClass.conv3d>

    Args:
        s: String or type.
        dim: Desired dimension.
        allowed_dims: Allowed dimensions to look for.

    Returns:
        Input with replaced dimension.
    """
    if isinstance(s, str) and dim in allowed_dims:
        return re.sub(f"[1-3]d$", f'{int(dim)}d', s)
    elif isinstance(s, type) or callable(s):
        return getattr(sys.modules[s.__module__], replace_ndim(s.__name__, dim))
    return s


def lookup_nn(item: str, *a, src=None, call=True, inplace=True, nd=None, **kw):
    """

    Examples:
        >>> lookup_nn('batchnorm2d', 32)
            BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        >>> lookup_nn(torch.nn.BatchNorm2d, 32)
            BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        >>> lookup_nn('batchnorm2d', num_features=32)
            BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        >>> lookup_nn('tanh')
            Tanh()
        >>> lookup_nn('tanh', call=False)
            torch.nn.modules.activation.Tanh
        >>> lookup_nn('relu')
            ReLU(inplace=True)
        >>> lookup_nn('relu', inplace=False)
            ReLU()

    Args:
        item: Lookup item. None is equivalent to `identity`.
        *a: Arguments passed to item if called.
        src: Lookup source.
        call: Whether to call item.
        inplace: Default setting for items that take an `inplace` argument when called.
            As default is True, `lookup_nn('relu')` returns a ReLu instance with `inplace=True`.
        nd: If set, replace dimension statement (e.g. '2d' in nn.Conv2d) with ``nd``.
        **kw: Keyword arguments passed to item when it is called.

    Returns:
        Looked up item.
    """
    src = src or nn
    if isinstance(item, tuple):
        if len(item) == 1:
            item, = item
        elif len(item) == 2:
            item, _kw = item
            kw.update(_kw)
        else:
            raise ValueError('Allowed formats for item: (item,) or (item, kwargs).')
    if item is None:
        v = nn.Identity
    elif isinstance(item, str):
        l_item = item.lower()
        if nd is not None:
            l_item = replace_ndim(l_item, nd)
        v = next((getattr(src, i) for i in dir(src) if i.lower() == l_item))
    elif isinstance(item, nn.Module):
        return item
    elif isinstance(item, type) and nd is not None:
        v = replace_ndim(item, nd)
    else:
        v = item
    if call:
        kwargs = {'inplace': inplace} if 'inplace' in inspect.getfullargspec(v).args else {}
        kwargs.update(kw)
        v = v(*a, **kwargs)
    return v


class ReadOut(nn.Module):
    def __init__(
            self,
            channels_in,
            channels_out,
            kernel_size=3,
            padding=1,
            activation='relu',
            norm='batchnorm2d',
            final_activation=None,
            dropout=0.1,
            channels_mid=None,
            stride=1
    ):
        super().__init__()
        self.channels_out = channels_out
        if channels_mid is None:
            channels_mid = channels_in

        self.block = nn.Sequential(
            nn.Conv2d(channels_in, channels_mid, kernel_size, padding=padding, stride=stride, bias=False),
            lookup_nn(norm, channels_mid),
            lookup_nn(activation),
            nn.Dropout2d(p=dropout) if dropout else nn.Identity(),
            nn.Conv2d(channels_mid, channels_out, 1, bias=False),
        )

        if final_activation is ...:
            self.activation = lookup_nn(activation)
        else:
            self.activation = lookup_nn(final_activation)

    def forward(self, x):
        out = self.block(x)
        return self.activation(out)


class ReadOut_refine(nn.Module):
    def __init__(
            self,
            channels_in,
            channels_out,
            kernel_size=3,
            padding=1,
            activation='relu',
            norm='batchnorm2d',
            final_activation=None,
            dropout=0.1,
            channels_mid=None,
            stride=1
    ):
        super().__init__()
        self.channels_out = channels_out
        if channels_mid is None:
            channels_mid = channels_in

        self.block = nn.Sequential(
            nn.Conv2d(channels_in, channels_mid, 3, padding=1, stride=1, bias=False),
            nn.BatchNorm2d(channels_mid),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels_mid, channels_mid, 3, padding=1, stride=1, bias=False),
            nn.BatchNorm2d(channels_mid),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels_mid, channels_mid, 3, padding=1, stride=1, bias=False),
            nn.BatchNorm2d(channels_mid),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels_mid, channels_mid, kernel_size, padding=padding, stride=stride, bias=False),
            nn.BatchNorm2d(channels_mid),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels_mid, channels_out, 1, bias=False),
        )

        if final_activation is ...:
            self.activation = lookup_nn(activation)
        else:
            self.activation = lookup_nn(final_activation)

    def forward(self, x):
        out = self.block(x)
        return self.activation(out)


def ResNet50(img_channel=3, num_classes=1000):
    return ResNet(block,[3,4,6,3],img_channel,num_classes)

def ResNet_50_FPN():
    return FPN(block,[3,4,6,3],256)

if __name__ == '__main__':
    ResNet_50_FPN()