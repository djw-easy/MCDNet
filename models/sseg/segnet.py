import torch
import torch.nn as nn
import torch.nn.functional as F


class SegNet(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(SegNet, self).__init__()

        self.pool = nn.MaxPool2d(2, return_indices=True)
        self.unpool = nn.MaxUnpool2d(2)
        
        self.conv1_1 = nn.Conv2d(in_channels, 64, 3, padding=1)
        self.conv1_1_bn = nn.BatchNorm2d(64)
        self.conv1_2 = nn.Conv2d(64, 64, 3, padding=1)
        self.conv1_2_bn = nn.BatchNorm2d(64)
        
        self.conv2_1 = nn.Conv2d(64, 128, 3, padding=1)
        self.conv2_1_bn = nn.BatchNorm2d(128)
        self.conv2_2 = nn.Conv2d(128, 128, 3, padding=1)
        self.conv2_2_bn = nn.BatchNorm2d(128)
        
        self.conv3_1 = nn.Conv2d(128, 256, 3, padding=1)
        self.conv3_1_bn = nn.BatchNorm2d(256)
        self.conv3_2 = nn.Conv2d(256, 256, 3, padding=1)
        self.conv3_2_bn = nn.BatchNorm2d(256)
        self.conv3_3 = nn.Conv2d(256, 256, 3, padding=1)
        self.conv3_3_bn = nn.BatchNorm2d(256)
        
        self.conv4_1 = nn.Conv2d(256, 512, 3, padding=1)
        self.conv4_1_bn = nn.BatchNorm2d(512)
        self.conv4_2 = nn.Conv2d(512, 512, 3, padding=1)
        self.conv4_2_bn = nn.BatchNorm2d(512)
        self.conv4_3 = nn.Conv2d(512, 512, 3, padding=1)
        self.conv4_3_bn = nn.BatchNorm2d(512)
        
        self.conv5_1 = nn.Conv2d(512, 512, 3, padding=1)
        self.conv5_1_bn = nn.BatchNorm2d(512)
        self.conv5_2 = nn.Conv2d(512, 512, 3, padding=1)
        self.conv5_2_bn = nn.BatchNorm2d(512)
        self.conv5_3 = nn.Conv2d(512, 512, 3, padding=1)
        self.conv5_3_bn = nn.BatchNorm2d(512)
        
        self.conv5_3_D = nn.Conv2d(512, 512, 3, padding=1)
        self.conv5_3_D_bn = nn.BatchNorm2d(512)
        self.conv5_2_D = nn.Conv2d(512, 512, 3, padding=1)
        self.conv5_2_D_bn = nn.BatchNorm2d(512)
        self.conv5_1_D = nn.Conv2d(512, 512, 3, padding=1)
        self.conv5_1_D_bn = nn.BatchNorm2d(512)
        
        self.conv4_3_D = nn.Conv2d(512, 512, 3, padding=1)
        self.conv4_3_D_bn = nn.BatchNorm2d(512)
        self.conv4_2_D = nn.Conv2d(512, 512, 3, padding=1)
        self.conv4_2_D_bn = nn.BatchNorm2d(512)
        self.conv4_1_D = nn.Conv2d(512, 256, 3, padding=1)
        self.conv4_1_D_bn = nn.BatchNorm2d(256)
        
        self.conv3_3_D = nn.Conv2d(256, 256, 3, padding=1)
        self.conv3_3_D_bn = nn.BatchNorm2d(256)
        self.conv3_2_D = nn.Conv2d(256, 256, 3, padding=1)
        self.conv3_2_D_bn = nn.BatchNorm2d(256)
        self.conv3_1_D = nn.Conv2d(256, 128, 3, padding=1)
        self.conv3_1_D_bn = nn.BatchNorm2d(128)
        
        self.conv2_2_D = nn.Conv2d(128, 128, 3, padding=1)
        self.conv2_2_D_bn = nn.BatchNorm2d(128)
        self.conv2_1_D = nn.Conv2d(128, 64, 3, padding=1)
        self.conv2_1_D_bn = nn.BatchNorm2d(64)
        
        self.conv1_2_D = nn.Conv2d(64, 64, 3, padding=1)
        self.conv1_2_D_bn = nn.BatchNorm2d(64)
        self.conv1_1_D = nn.Conv2d(64, out_channels, 3, padding=1)
        
        self.apply(self._weights_init)
        
    def _weights_init(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.kaiming_normal(m.weight.data)
        
    def forward(self, x):
        # Encoder block 1
        x = F.relu(self.conv1_1_bn(self.conv1_1(x)))
        x1 = F.relu(self.conv1_2_bn(self.conv1_2(x)))
        size1 = x.size()
        x, mask1 = self.pool(x1)
        
        # Encoder block 2
        x = F.relu(self.conv2_1_bn(self.conv2_1(x)))
        x2 = F.relu(self.conv2_2_bn(self.conv2_2(x)))
        size2 = x.size()
        x, mask2 = self.pool(x2)
        
        # Encoder block 3
        x = F.relu(self.conv3_1_bn(self.conv3_1(x)))
        x = F.relu(self.conv3_2_bn(self.conv3_2(x)))
        x3 = F.relu(self.conv3_3_bn(self.conv3_3(x)))
        size3 = x.size()
        x, mask3 = self.pool(x3)
        
        # Encoder block 4
        x = F.relu(self.conv4_1_bn(self.conv4_1(x)))
        x = F.relu(self.conv4_2_bn(self.conv4_2(x)))
        x4 = F.relu(self.conv4_3_bn(self.conv4_3(x)))
        size4 = x.size()
        x, mask4 = self.pool(x4)
        
        # Encoder block 5
        x = F.relu(self.conv5_1_bn(self.conv5_1(x)))
        x = F.relu(self.conv5_2_bn(self.conv5_2(x)))
        x = F.relu(self.conv5_3_bn(self.conv5_3(x)))
        size5 = x.size()
        x, mask5 = self.pool(x)
        
        # Decoder block 5
        x = self.unpool(x, mask5, output_size = size5)
        x = F.relu(self.conv5_3_D_bn(self.conv5_3_D(x)))
        x = F.relu(self.conv5_2_D_bn(self.conv5_2_D(x)))
        x = F.relu(self.conv5_1_D_bn(self.conv5_1_D(x)))
        
        # Decoder block 4
        x = self.unpool(x, mask4, output_size = size4)
        x = F.relu(self.conv4_3_D_bn(self.conv4_3_D(x)))
        x = F.relu(self.conv4_2_D_bn(self.conv4_2_D(x)))
        x = F.relu(self.conv4_1_D_bn(self.conv4_1_D(x)))
        
        # Decoder block 3
        x = self.unpool(x, mask3, output_size = size3)
        x = F.relu(self.conv3_3_D_bn(self.conv3_3_D(x)))
        x = F.relu(self.conv3_2_D_bn(self.conv3_2_D(x)))
        x = F.relu(self.conv3_1_D_bn(self.conv3_1_D(x)))
        
        # Decoder block 2
        x = self.unpool(x, mask2, output_size = size2)
        x = F.relu(self.conv2_2_D_bn(self.conv2_2_D(x)))
        x = F.relu(self.conv2_1_D_bn(self.conv2_1_D(x)))
        
        # Decoder block 1
        x = self.unpool(x, mask1, output_size = size1)
        x = F.relu(self.conv1_2_D_bn(self.conv1_2_D(x)))
        x = self.conv1_1_D(x)
        return x


