import torch
import torch.nn as nn
import torch.nn.functional as F

from torchsummary import summary


class CBR(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, 
                 stride=1, padding=1, dilation=1, bias=True) -> None:
        super(CBR, self).__init__()
        self.module = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, 
                      stride=stride, padding=padding, dilation=dilation, bias=bias),
            nn.BatchNorm2d(out_channels), 
            nn.ReLU()
        )

    def forward(self, x):
        return self.module(x)


class CBRR(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, 
                 stride=1, padding=1, dilation=1, bias=True) -> None:
        super(CBRR, self).__init__()
        self.cbr1 = CBR(in_channels, out_channels, kernel_size=kernel_size, 
                        stride=stride, padding=padding, bias=bias)
        self.cbr2 = CBR(out_channels, out_channels, kernel_size=kernel_size, 
                        stride=stride, padding=padding, dilation=dilation, bias=bias)
        self.cbr3 = CBR(out_channels, out_channels, kernel_size=kernel_size, 
                        stride=stride, padding=padding, dilation=dilation, bias=bias)

    def forward(self, x):
        x = self.cbr1(x)
        residual = x

        x = self.cbr2(x)
        x = self.cbr3(x)
        return residual + x
    

class MFF(nn.Module):
    def __init__(self, out_channels) -> None:
        super(MFF, self).__init__()
        self.cbr1 = CBR(512, out_channels)
        self.up1 = nn.ConvTranspose2d(out_channels, out_channels, kernel_size=8, stride=8, padding=0)

        self.cbr2 = CBR(512, out_channels)
        self.up2 = nn.ConvTranspose2d(out_channels, out_channels, kernel_size=8, stride=8, padding=0)

        self.cbr3 = CBR(512, out_channels)
        self.up3 = nn.ConvTranspose2d(out_channels, out_channels, kernel_size=8, stride=8, padding=0)

        self.cbr4 = CBR(256, out_channels)
        self.up4 = nn.ConvTranspose2d(out_channels, out_channels, kernel_size=4, stride=4, padding=0)

        self.cbr5 = CBR(128, out_channels)
        self.up5 = nn.ConvTranspose2d(out_channels, out_channels, kernel_size=2, stride=2, padding=0)

        self.cbr6 = CBR(64, out_channels)
        self.head = CBR(out_channels*6, out_channels)

    def forward(self, features):
        res8x_3, res8x_2, res8x_1, res4x, res2x, res1x = features

        res8x_3 = self.up1(self.cbr1(res8x_3))
        res8x_2 = self.up2(self.cbr2(res8x_2))
        res8x_1 = self.up3(self.cbr3(res8x_1))
        res4x = self.up4(self.cbr4(res4x))
        res2x = self.up5(self.cbr5(res2x))
        res1x = self.cbr6(res1x)

        ff = torch.cat([res8x_3, res8x_2, res8x_1, res4x, res2x, res1x], dim=1)
        out = self.head(ff)

        return out


class MSCFF(nn.Module):
    def __init__(self, in_channels=4, out_channels=4) -> None:
        super(MSCFF, self).__init__()
        # encoder
        self.cbrr1 = CBRR(in_channels, 64)
        self.pool1 = nn.MaxPool2d(2)

        self.cbrr2 = CBRR(64, 128)
        self.pool2 = nn.MaxPool2d(2)

        self.cbrr3 = CBRR(128, 256)
        self.pool3 = nn.MaxPool2d(2)

        self.cbrr4 = CBRR(256, 512)
        self.cbrr5 = CBRR(512, 512, dilation=2, padding='same')
        self.cbrr6 = CBRR(512, 512, dilation=4, padding='same')

        # decoder
        self.cbrr6_d = CBRR(512, 512, dilation=4, padding='same')
        self.cbrr5_d = CBRR(512, 512, dilation=2, padding='same')
        self.cbrr4_d = CBRR(512, 512)

        self.up1 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2, padding=0)
        self.cbrr3_d = CBRR(256, 256)

        self.up2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2, padding=0)
        self.cbrr2_d = CBRR(128, 128)

        self.up3 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2, padding=0)
        self.cbrr1_d = CBRR(64, 64)

        self.mff = MFF(out_channels)
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
        # encoder1
        res1x = self.cbrr1(x)

        res2x = self.pool1(res1x)
        res2x = self.cbrr2(res2x)

        res4x =self.pool2(res2x)
        res4x = self.cbrr3(res4x)

        res8x_1 = self.pool3(res4x)
        res8x_1 = self.cbrr4(res8x_1)

        res8x_2 = self.cbrr5(res8x_1)
        res8x_3 = self.cbrr6(res8x_2)

        # decoder
        res8x_3 = self.cbrr6_d(res8x_3) + res8x_3
        res8x_2 = self.cbrr5_d(res8x_3) + res8x_2
        res8x_1 = self.cbrr4_d(res8x_2) + res8x_1
        res4x = self.cbrr3_d(self.up1(res8x_1)) + res4x
        res2x = self.cbrr2_d(self.up2(res4x)) + res2x
        res1x = self.cbrr1_d(self.up3(res2x)) + res1x

        out = self.mff([res8x_3, res8x_2, res8x_1, res4x, res2x, res1x])

        return out
