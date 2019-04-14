#https://github.com/miraclewkf/SENet-PyTorch/blob/master/se_resnet.py
import torch.nn as nn
import math
import torch.utils.model_zoo as model_zoo


__all__ = ['SENet', 'se_resnet_18', 'se_resnet_34', 'se_resnet_50', 'se_resnet_101','se_resnet_152']

def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,padding=1, bias=False)

class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

        if planes == 64:
            self.globalAvgPool = nn.AvgPool2d(56, stride=1)
        elif planes == 128:
            self.globalAvgPool = nn.AvgPool2d(28, stride=1)
        elif planes == 256:
            self.globalAvgPool = nn.AvgPool2d(14, stride=1)
        elif planes == 512:
            self.globalAvgPool = nn.AvgPool2d(7, stride=1)
        self.fc1 = nn.Linear(in_features=planes, out_features=round(planes / 16))
        self.fc2 = nn.Linear(in_features=round(planes / 16), out_features=planes)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        original_out = out
        out = self.globalAvgPool(out)
        out = out.view(out.size(0), -1)
        out = self.fc1(out)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.sigmoid(out)
        out = out.view(out.size(0), out.size(1), 1, 1)
        out = out * original_out

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 1#4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        if planes == 64:
            self.globalAvgPool = nn.AvgPool2d(56, stride=1)
        elif planes == 128:
            self.globalAvgPool = nn.AvgPool2d(28, stride=1)
        elif planes == 256:
            self.globalAvgPool = nn.AvgPool2d(14, stride=1)
        elif planes == 512:
            self.globalAvgPool = nn.AvgPool2d(7, stride=1)
        self.fc1 = nn.Linear(in_features=planes * 4, out_features=round(planes / 4))
        self.fc2 = nn.Linear(in_features=round(planes / 4), out_features=planes * 4)
        self.sigmoid = nn.Sigmoid()
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        original_out = out
        out = self.globalAvgPool(out)
        out = out.view(out.size(0), -1)
        out = self.fc1(out)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.sigmoid(out)
        out = out.view(out.size(0),out.size(1),1,1)
        out = out * original_out

        out += residual
        out = self.relu(out)

        return out


class SENet(nn.Module):

    def __init__(self, block, layers, num_classes=1000):
        self.inplanes = 64
        super(SENet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        self.conv_C1=nn.Conv2d(64,64,kernel_size=1)
        self.conv_C2=nn.Conv2d(128,128,kernel_size=1)
        self.conv_C3=nn.Conv2d(256,256,kernel_size=1)

        ####
        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.fc = nn.Linear(512 * block.expansion, num_classes)
        ###

        self.upsample_C4 = nn.ConvTranspose2d(512, 256, 4, stride=2, bias=False,padding=1)
        self.upsample2 = nn.ConvTranspose2d(256, 128, 4, stride=2, bias=False,padding=1)
        self.upconv2=nn.Conv2d(1,64,kernel_size=7,stride=2,padding=0,bias=False)
        self.upsample8 = nn.ConvTranspose2d(128, 64, 16, stride=8, bias=False,padding=4)


        self.fcn = nn.Conv2d(64,1,3,stride=1,padding=1)
        self.sigmoid=nn.Sigmoid()

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        #print("１　x.size() = ",x.size())#112x112x64
        x = self.bn1(x)
        #print("2   x.size() = ",x.size())#112x112x64
        x = self.relu(x)
        #print("3  x.size() = ",x.size())#112x112x64
        x = self.maxpool(x)
        #print("4  x.size() = ",x.size())#56x56x64
        C1 = x = self.layer1(x)
        #print("5  x.size() = ",x.size())#56x56x64
        C2 = x = self.layer2(x)
        #print("6  x.size() = ",x.size())#28x28x128
        C3 = x = self.layer3(x)
        #print("7  x.size() = ",x.size())#14x14x256
        C4 = x = self.layer4(x)
        #print("8  x.size() = ",x.size())#7x7x512
        x = self.upsample_C4(C4)  #1/16 channel=256
        stage_3=self.relu(self.conv_C3(C3)) + x
        x = self.upsample2(stage_3) #1/8  channel = 128
        stage_2=self.relu(self.conv_C2(C2)) + x
        x = self.upsample8(stage_2) # channel = 64
        #stag3_1 = self.conv_C1(C1)+x
        x = self.fcn(x)
        x = self.sigmoid(x)
        #x = self.avgpool(x)
        #print("9 x.size() = ",x.size())#1x1x512
        #x = x.view(x.size(0), -1)
        #print("10 x.size() = ",x.size())#512x1
        #x = self.fc(x)
        #print("11 x.size() = ",x.size())#1000x1
        return x


def se_resnet_18(pretrained=False, **kwargs):
    model = SENet(BasicBlock, [2, 2, 2, 2], **kwargs)
    return model


def se_resnet_34(pretrained=False, **kwargs):
    model = SENet(BasicBlock, [3, 4, 6, 3], **kwargs)
    return model


def se_resnet_50(pretrained=False, **kwargs):
    model = SENet(Bottleneck, [3, 4, 6, 3], **kwargs)
    return model


def se_resnet_101(pretrained=False, **kwargs):
    model = SENet(Bottleneck, [3, 4, 23, 3], **kwargs)
    return model


def se_resnet_152(pretrained=False, **kwargs):
    model = SENet(Bottleneck, [3, 8, 36, 3], **kwargs)
    return model
