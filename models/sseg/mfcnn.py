import torch
import torch.nn as nn
import torch.nn.functional as F

from torchsummary import summary


class PPM(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(PPM, self).__init__()
        
        self.pool_sizes = [2, 4, 8, 16] # subregion size in each level
        self.num_levels = len(self.pool_sizes) # number of pyramid levels
        
        self.conv_layers = nn.ModuleList()
        for i in range(self.num_levels):
            self.conv_layers.append(nn.Sequential(
                nn.AdaptiveAvgPool2d(output_size=self.pool_sizes[i]),
                nn.Conv2d(in_channels, out_channels, kernel_size=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            ))
    
    def forward(self, x):
        input_size = x.size()[2:] # get input size
        output = []
        
        # pyramid pooling
        for i in range(self.num_levels):
            out = self.conv_layers[i](x)
            out = F.interpolate(out, size=input_size, mode='bilinear', align_corners=True)
            output.append(out)
        
        # concatenate features from different levels
        output = torch.cat(output, dim=1)
        
        return output


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        mid_channels = (in_channels + out_channels) // 2
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class MFCNN(nn.Module):
    def __init__(self, in_channels, out_channels) -> None:
        super().__init__()

        self.input_conv = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, stride=2, padding=1), 
            nn.ReLU(inplace=True)
        )
        self.conv1 = DoubleConv(64, 128)

        self.pool2 = nn.MaxPool2d(2)
        self.conv2 = DoubleConv(128, 256)

        self.pool3 = nn.MaxPool2d(2)
        self.conv3 = DoubleConv(256, 512)

        self.ppm = PPM(512, 256)

        self.d_conv1 = nn.Sequential(
            nn.Conv2d(512+1024, 512, 3, 1, 1), 
            nn.ReLU(inplace=True)
        )
        self.d_conv2 = nn.Sequential(
            nn.Conv2d(256+512, 256, 3, 1, 1), 
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        self.d_conv3 = nn.Sequential(
            nn.Conv2d(128+256, 128, 3, 1, 1), 
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )

        self.dropout = nn.Dropout2d(0.5)
        self.out_conv = nn.Conv2d(128, out_channels, 3, 1, 1)

    def forward(self, x):
        x1 = self.input_conv(x)
        x1 = self.conv1(x1)

        x2 = self.pool2(x1)
        x2 = self.conv2(x2)

        x3 = self.pool3(x2)
        x3 = self.conv3(x3)

        mf = self.ppm(x3)

        x = self.d_conv1(torch.concat([mf, x3], dim=1))
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)

        x = self.d_conv2(torch.concat([x, x2], dim=1))
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)

        x = self.d_conv3(torch.concat([x, x1], dim=1))
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)

        x = self.dropout(x)
        x = self.out_conv(x)

        return x


if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device = 'cpu'
    model = MFCNN(in_channels=3, out_channels=3).to(device)
    a = torch.ones([8, 3, 256, 256]).to(device)
    r = model(a)
    print(r.shape)
    # print(model)
    params_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("%s Params: %.2fM" % ('MFCNN', params_num / 1e6))
    # summary(model, (3, 256, 256))

