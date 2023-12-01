import torch
import torch.nn as nn
from .resnet import resnet101
import torch.nn.functional as F
from torchvision.models._utils import IntermediateLayerGetter


class PyramidPoolingModule(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(PyramidPoolingModule, self).__init__()
        
        self.pool_sizes = [1, 2, 3, 6] # subregion size in each level
        self.num_levels = len(self.pool_sizes) # number of pyramid levels
        
        self.conv_layers = nn.ModuleList()
        for i in range(self.num_levels):
            self.conv_layers.append(nn.Sequential(
                nn.AdaptiveAvgPool2d(output_size=self.pool_sizes[i]),
                nn.Conv2d(in_channels, in_channels // self.num_levels, kernel_size=1),
                nn.BatchNorm2d(in_channels // self.num_levels),
                nn.ReLU(inplace=True)
            ))
        self.out_conv = nn.Conv2d(in_channels*2, out_channels, kernel_size=1, stride=1)
    
    def forward(self, x):
        input_size = x.size()[2:] # get input size
        output = [x]
        
        # pyramid pooling
        for i in range(self.num_levels):
            out = self.conv_layers[i](x)
            out = F.interpolate(out, size=input_size, mode='bilinear', align_corners=True)
            output.append(out)
        
        # concatenate features from different levels
        output = torch.cat(output, dim=1)
        output = self.out_conv(output)
        
        return output


class DilatedConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=2, dilation=2):
        super(DilatedConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, 
                              stride=stride, padding=padding, dilation=dilation)

    def forward(self, x):
        out = self.conv(x)
        return out


class MFFSNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=3):
        super(MFFSNet, self).__init__()
        self.out_channels = out_channels
        self.backbone = IntermediateLayerGetter(
            resnet101(in_channels=in_channels),
            return_layers={'layer1':"layer1" ,'layer2': 'layer2'}
        )
        self.dc1 = DilatedConv2d(512, 512)
        self.dc2 = DilatedConv2d(512, 512)

        self.ppm5 = PyramidPoolingModule(512, 512)
        self.ppm4 = PyramidPoolingModule(512, 512)
        self.ppm3 = PyramidPoolingModule(512, 512)

        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.ppm2 = PyramidPoolingModule(256, 512)

        self.out_conv = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, out_channels, kernel_size=1)
        )
        self.apply(self._weights_init)
    
    def _weights_init(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight)
            nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        _, _, h, w = x.size()
        feats = self.backbone(x)
        c2, c3 = feats['layer1'], feats['layer2'] # 256, 512
        c4 = self.dc1(c3) # 512
        c5 = self.dc2(c4) # 512

        f5 = self.ppm5(c5)
        
        p4 = self.ppm4(c4)
        f4 = f5 + p4

        p3 = self.ppm3(c3)
        f3 = f4 + p3

        p2 = self.ppm2(c2)
        f2 = self.up(f3) + p2

        out = self.out_conv(f2)
        out = nn.functional.interpolate(out, size=(h, w), mode='bilinear', align_corners=True)

        return out


if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = MFFSNet(in_channels=4, out_channels=4).to(device)
    a = torch.ones([2, 4, 512, 512]).to(device)
    r = model(a, a)
    print(r.shape)

