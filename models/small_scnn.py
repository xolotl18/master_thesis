import torch
from torch import nn
from torch.nn import functional as F


class SmallSCNN(nn.Module):
    def __init__(self, in_channels, num_classes):
        super().__init__()

        self.learning_to_down_sample = LearningToDownSample(in_channels)
        self.global_feature_extractor = GlobalFeatureExtractor()
        self.feature_fusion = FeatureFusion(scale_factor=4)
        self.classifier = Classifier(num_classes, scale_factor=8)

    def forward(self, x):
        shared = self.learning_to_down_sample(x)
        x = self.global_feature_extractor(shared)
        x = self.feature_fusion(shared, x)
        x = self.classifier(x)
        return x


class LearningToDownSample(nn.Module):
    def __init__(self, in_channels):
        super().__init__()

        self.conv = ConvBlock(in_channels=in_channels, out_channels=16, stride=2)
        self.dsconv1 = nn.Sequential(
            # depthwise convolution
            nn.Conv2d(16, 16, kernel_size=3, stride=2, padding=1, dilation=1, groups=16, bias=False),
            nn.BatchNorm2d(16),
            # pointwise convolution
            nn.Conv2d(16, 24, kernel_size=1, stride=1, padding=0, dilation=1, groups=1, bias=False),
            nn.BatchNorm2d(24),
            nn.ReLU(inplace=True))
        self.dsconv2 = nn.Sequential(
            nn.Conv2d(24, 24, kernel_size=3, stride=2, padding=1, dilation=1, groups=24, bias=False),
            nn.BatchNorm2d(24),
            nn.Conv2d(24, 32, kernel_size=1, stride=1, padding=0, dilation=1, groups=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True))

    def forward(self, x):
        x = self.conv(x)
        x = self.dsconv1(x)
        x = self.dsconv2(x)
        return x


class GlobalFeatureExtractor(nn.Module):
    def __init__(self):
        super().__init__()

        self.first_block = nn.Sequential(Bottleneck(32, 32, 2, 4),
                                         Bottleneck(32, 32, 1, 4))
        self.second_block = nn.Sequential(Bottleneck(32, 48, 2, 4),
                                          Bottleneck(48, 48, 1, 4))
        self.third_block = nn.Sequential(Bottleneck(48, 64, 1, 4),
                                         Bottleneck(64, 64, 1, 4))
        self.ppm = PPMModule(64, 64)

    def forward(self, x):
        x = self.first_block(x)
        x = self.second_block(x)
        x = self.third_block(x)
        x = self.ppm(x)
        return x


class FeatureFusion(nn.Module):
    def __init__(self, scale_factor):
        super().__init__()

        self.scale_factor = scale_factor
        self.conv_high_res = nn.Conv2d(32, 64, kernel_size=1, stride=1, padding=0, bias=True)

        self.dwconv = ConvBlock(in_channels=64, out_channels=64, stride=1, padding=scale_factor,
                                dilation=scale_factor, groups=64)
        self.conv_low_res = nn.Conv2d(64, 64, kernel_size=1, stride=1, padding=0, bias=True)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, high_res_input, low_res_input):
        low_res_input = F.interpolate(input=low_res_input, scale_factor=self.scale_factor, mode='bilinear',
                                      align_corners=True)
        low_res_input = self.dwconv(low_res_input)
        low_res_input = self.conv_low_res(low_res_input)

        high_res_input = self.conv_high_res(high_res_input)
        x = torch.add(high_res_input, low_res_input)
        return self.relu(x)


class Classifier(nn.Module):
    def __init__(self, num_classes, scale_factor):
        super().__init__()

        self.scale_factor = scale_factor
        self.dsconv1 = nn.Sequential(
            # depthwise convolution
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, dilation=1, groups=64, bias=False),
            nn.BatchNorm2d(64),
            # pointwise convolution
            nn.Conv2d(64, 64, kernel_size=1, stride=1, padding=0, dilation=1, groups=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True))
        self.dsconv2 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, dilation=1, groups=64, bias=False),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 64, kernel_size=1, stride=1, padding=0, dilation=1, groups=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True))
        self.drop_out = nn.Dropout(p=0.1)
        self.conv = nn.Conv2d(64, num_classes, kernel_size=1, stride=1, padding=0, bias=True)

    def forward(self, x):
        x = self.dsconv1(x)
        x = self.dsconv2(x)
        x = self.drop_out(x)
        x = self.conv(x)
        x = F.interpolate(input=x, scale_factor=self.scale_factor, mode='bilinear', align_corners=True)
        return x


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=2, padding=1, dilation=1, groups=1):
        super().__init__()

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding,
                              dilation=dilation, groups=groups, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, input):
        x = self.conv(input)
        return self.relu(self.bn(x))


class Bottleneck(nn.Module):
    def __init__(self, in_channels, out_channels, stride, expand_ratio):
        super().__init__()

        hidden_dim = in_channels * expand_ratio
        self.use_res_connect = stride == 1 and in_channels == out_channels

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim, 1, 1, 0, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
            # depthwise convolution
            nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
            # pw-linear
            nn.Conv2d(hidden_dim, out_channels, 1, 1, 0, bias=False),
            nn.BatchNorm2d(out_channels))

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)

    
class PPMModule(nn.Module):
    #
    # in this simplified version of fast scnn we simplify pyramid pooling using a 
    # single pyramid scale which is computed with a global average pooling operation
    #     
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.in_channels = in_channels   # should be 64
        self.out_channels = out_channels # should be 64
        
        self.pool1 = nn.AvgPool2d(16)    # produces 1x1x64 feature map
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0,
                      dilation=1, bias=False),
            nn.ReLU(inplace=True)
        )
        
        self.conv_last = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0,
                      dilation=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        feat = torch.mean(x, dim=(2, 3), keepdim=True)
        feat = self.conv(feat)
        feat = feat + x
        feat = self.conv_last(feat)
        
        return feat
        
        
        