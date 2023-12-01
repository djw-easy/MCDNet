import torch
import torch.nn as nn
import torch.nn.functional as F

from torchsummary import summary


class BnRelu(nn.Module):
    """It adds a Batch_normalization layer before a Relu
    """
    def __init__(self, in_channels) -> None:
        super(BnRelu, self).__init__()
        self.module = nn.Sequential(
            nn.BatchNorm2d(in_channels), 
            nn.ReLU()
        )

    def forward(self, x):
        return self.module(x)

class ContractingArm(nn.Module):
    """It adds a feedforward signal to the output of two following conv layers in contracting path
    """
    def __init__(self, in_channels, out_channels, kernel_size=(3, 3)):
        super(ContractingArm, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding='same')
        self.bn_relu1 = BnRelu(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, padding='same')
        self.bn_relu2 = BnRelu(out_channels)
        self.conv3 = nn.Conv2d(in_channels, out_channels // 2, kernel_size=(kernel_size[0]-2, kernel_size[0]-2),
                               padding='same')
        self.bn_relu3 = BnRelu(out_channels // 2)

    def forward(self, x):
        # x 16
        out = self.conv1(x) # 32
        out = self.bn_relu1(out)
        out = self.conv2(out) # 32
        out = self.bn_relu2(out)

        out_b = self.conv3(x) # 16
        out_b = self.bn_relu3(out_b) # 16

        out_b = torch.cat([x, out_b], dim=1) # 32

        out = out + out_b
        out = nn.ReLU()(out)
        return out

class ImprovingContractingArm(nn.Module):
    """It adds a feedforward signal to the output of two following conv layers in contracting path
    """
    def __init__(self, in_channels, out_channels, kernel_size=(3, 3)):
        super(ImprovingContractingArm, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding='same')
        self.bn_relu1 = BnRelu(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, padding='same')
        self.bn_relu2 = BnRelu(out_channels)
        self.conv3 = nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, padding='same')
        self.bn_relu3 = BnRelu(out_channels)
        self.conv4 = nn.Conv2d(in_channels, out_channels // 2, kernel_size=(kernel_size[0]-2, kernel_size[0]-2),
                               padding='same')
        self.bn_relu4 = BnRelu(out_channels // 2)
        self.conv5 = nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, padding='same')
        self.bn_relu5 = BnRelu(out_channels)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn_relu1(out)
        out0 = self.conv2(out)
        out0 = self.bn_relu2(out0)
        out = self.conv3(out0)
        out = self.bn_relu3(out)

        out_b = self.conv4(x)
        out_b = self.bn_relu4(out_b)

        out_b = torch.cat([x, out_b], dim=1)

        out2 = self.conv5(out0)
        out2 = self.bn_relu5(out2)

        out = out + out_b + out2
        out = nn.ReLU()(out)
        return out

class Bridge(nn.Module):
    """It is exactly like the identity_block plus a dropout layer. This block only uses in the valley of the UNet
    """
    def __init__(self, in_channels, out_channels, kernel_size=(3, 3)):
        super(Bridge, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding='same')
        self.bn_relu1 = BnRelu(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, padding='same')
        self.dropout = nn.Dropout2d(p=0.15)
        self.bn_relu2 = BnRelu(out_channels)
        self.conv3 = nn.Conv2d(in_channels, out_channels // 2,
                               kernel_size=(kernel_size[0]-2, kernel_size[0]-2), padding='same')
        self.bn_relu3 = BnRelu(out_channels // 2)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn_relu1(out)
        out = self.conv2(out)
        out = self.dropout(out)
        out = self.bn_relu2(out)

        out_b = self.conv3(x)
        out_b = self.bn_relu3(out_b)

        out_b = torch.cat([x, out_b], dim=1)

        out = out + out_b
        out = nn.ReLU()(out)
        return out


class DoubleConv(nn.Module):
    """It Is only the convolution part inside each expanding path's block
    """
    def __init__(self, in_channels, out_channels, kernel_size=(3, 3)) -> None:
        super(DoubleConv, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding='same')
        self.bn_relu1 = BnRelu(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, padding='same')
        self.bn_relu2 = BnRelu(out_channels)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn_relu1(x)

        x = self.conv2(x)
        x = self.bn_relu2(x)

        return x
    

class TripleConv(nn.Module):
    """It Is only the convolution part inside each expanding path's block
    """
    def __init__(self, in_channels, out_channels, kernel_size=(3, 3)) -> None:
        super(TripleConv, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding='same')
        self.bn_relu1 = BnRelu(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, padding='same')
        self.bn_relu2 = BnRelu(out_channels)
        self.conv3 = nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, padding='same')
        self.bn_relu3 = BnRelu(out_channels)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn_relu1(x)

        x = self.conv2(x)
        x = self.bn_relu2(x)

        x = self.conv3(x)
        x = self.bn_relu3(x)

        return x


class AddRelu(nn.Module):
    """It is for adding two feed forwards to the output of the two following conv layers in expanding path
    """
    def __init__(self) -> None:
        super(AddRelu, self).__init__()

    def forward(self, input_tensor1, input_tensor2, input_tensor3):
        x = input_tensor1 + input_tensor2 + input_tensor3
        return nn.ReLU()(x)


class ImproveFFBlock4(nn.Module):
    """It improves the skip connection by using previous layers feature maps"""
    def __init__(self):
        super(ImproveFFBlock4, self).__init__()
        self.relu = nn.ReLU(inplace=True)

    def forward(self, input_tensor1, input_tensor2, input_tensor3, input_tensor4, pure_ff):
        x1 = torch.cat([input_tensor1] * 2, dim=1)
        x1 = F.max_pool2d(x1, kernel_size=2)

        x2 = torch.cat([input_tensor2] * 4, dim=1)
        x2 = F.max_pool2d(x2, kernel_size=4)

        x3 = torch.cat([input_tensor3] * 8, dim=1)
        x3 = F.max_pool2d(x3, kernel_size=8)

        x4 = torch.cat([input_tensor4] * 16, dim=1)
        x4 = F.max_pool2d(x4, kernel_size=16)

        x = x1 + x2 + x3 + x4 + pure_ff
        x = self.relu(x)
        return x


class ImproveFFBlock3(nn.Module):
    """It improves the skip connection by using previous layers feature maps"""
    def __init__(self):
        super(ImproveFFBlock3, self).__init__()
        self.relu = nn.ReLU(inplace=True)

    def forward(self, input_tensor1, input_tensor2, input_tensor3, pure_ff):
        x1 = torch.cat([input_tensor1] * 2, dim=1)
        x1 = F.max_pool2d(x1, kernel_size=2)

        x2 = torch.cat([input_tensor2] * 4, dim=1)
        x2 = F.max_pool2d(x2, kernel_size=4)

        x3 = torch.cat([input_tensor3] * 8, dim=1)
        x3 = F.max_pool2d(x3, kernel_size=8)

        x = x1 + x2 + x3 + pure_ff
        x = self.relu(x)
        return x


class ImproveFFBlock2(nn.Module):
    """It improves the skip connection by using previous layers feature maps"""
    def __init__(self):
        super(ImproveFFBlock2, self).__init__()
        self.relu = nn.ReLU(inplace=True)

    def forward(self, input_tensor1, input_tensor2, pure_ff):
        x1 = torch.cat([input_tensor1] * 2, dim=1)
        x1 = F.max_pool2d(x1, kernel_size=2)

        x2 = torch.cat([input_tensor2] * 4, dim=1)
        x2 = F.max_pool2d(x2, kernel_size=4)

        x = x1 + x2 + pure_ff
        x = self.relu(x)
        return x


class ImproveFFBlock1(nn.Module):
    """It improves the skip connection by using previous layers feature maps"""
    def __init__(self):
        super(ImproveFFBlock1, self).__init__()
        self.relu = nn.ReLU(inplace=True)

    def forward(self, input_tensor1, pure_ff):
        x1 = torch.cat([input_tensor1] * 2, dim=1)
        x1 = F.max_pool2d(x1, kernel_size=2)

        x = x1 + pure_ff
        x = self.relu(x)
        return x


class CloudNet(nn.Module):
    def __init__(self, in_channels=4, out_channels=4) -> None:
        super(CloudNet, self).__init__()
        self.prev = nn.Sequential(
            nn.Conv2d(in_channels, 16, (3, 3), padding='same'), 
            nn.ReLU()
        )

        self.contr_arm1 = ContractingArm(16, 32, kernel_size=(3, 3))
        self.pool1 = nn.MaxPool2d(2)

        self.contr_arm2 = ContractingArm(32, 64, kernel_size=(3, 3))
        self.pool2 = nn.MaxPool2d(2)

        self.contr_arm3 = ContractingArm(64, 128, kernel_size=(3, 3))
        self.pool3 = nn.MaxPool2d(2)

        self.contr_arm4 = ContractingArm(128, 256, kernel_size=(3, 3))
        self.pool4 = nn.MaxPool2d(2)

        self.imprv_contr_arm = ImprovingContractingArm(256, 512, kernel_size=(3, 3))
        self.pool5 = nn.MaxPool2d(2)

        self.bridge = Bridge(512, 1024, kernel_size=(3, 3))

        self.convt1 = nn.ConvTranspose2d(1024, 512, kernel_size=(2, 2), stride=(2, 2), padding=0)
        self.improve_ff_block4 = ImproveFFBlock4()
        self.triple_conv = TripleConv(1024, 512, kernel_size=(3, 3))
        self.add_block_exp_path1 = AddRelu()

        self.convt2 = nn.ConvTranspose2d(512, 256, kernel_size=(2, 2), stride=(2, 2), padding=0)
        self.improve_ff_block3 = ImproveFFBlock3()
        self.double_conv1 = DoubleConv(512, 256, kernel_size=(3, 3))
        self.add_block_exp_path2 = AddRelu()

        self.convt3 = nn.ConvTranspose2d(256, 128, kernel_size=(2, 2), stride=(2, 2), padding=0)
        self.improve_ff_block2 = ImproveFFBlock2()
        self.double_conv2 = DoubleConv(256, 128, kernel_size=(3, 3))
        self.add_block_exp_path3 = AddRelu()

        self.convt4 = nn.ConvTranspose2d(128, 64, kernel_size=(2, 2), stride=(2, 2), padding=0)
        self.improve_ff_block1 = ImproveFFBlock1()
        self.double_conv3 = DoubleConv(128, 64, kernel_size=(3, 3))
        self.add_block_exp_path4 = AddRelu()

        self.convt5 = nn.ConvTranspose2d(64, 32, kernel_size=(2, 2), stride=(2, 2), padding=0)
        self.double_conv4 = DoubleConv(64, 32, kernel_size=(3, 3))
        self.add_block_exp_path5 = AddRelu()

        self.final = nn.Sequential(
            nn.Conv2d(32, out_channels, 1), 
            # nn.LogSoftmax(dim=1)
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
        x1 = self.prev(x)
        x1 = self.contr_arm1(x1)
        x1_pool = self.pool1(x1)

        x2 = self.contr_arm2(x1_pool)
        x2_pool = self.pool2(x2)

        x3 = self.contr_arm3(x2_pool)
        x3_pool = self.pool3(x3)

        x4 = self.contr_arm4(x3_pool)
        x4_pool = self.pool4(x4)

        x5 = self.imprv_contr_arm(x4_pool)
        x5_pool = self.pool5(x5)

        x6 = self.bridge(x5_pool)

        x7_1 = self.convt1(x6)
        x7_2 = self.improve_ff_block4(x4, x3, x2, x1, x5)
        x7_2 = torch.cat([x7_1, x7_2], dim=1)
        x7_2 = self.triple_conv(x7_2)
        x7_2 = self.add_block_exp_path1(x7_2, x5, x7_1)

        x8_1 = self.convt2(x7_2)
        x8_2 = self.improve_ff_block3(x3, x2, x1, x4)
        x8_2 = torch.cat([x8_1, x8_2], dim=1)
        x8_2 = self.double_conv1(x8_2)
        x8_2 = self.add_block_exp_path2(x8_2, x4, x8_1)

        x9_1 = self.convt3(x8_2)
        x9_2 = self.improve_ff_block2(x2, x1, x3)
        x9_2 = torch.cat([x9_1, x9_2], dim=1)
        x9_2 = self.double_conv2(x9_2)
        x9_2 = self.add_block_exp_path3(x9_2, x3, x9_1)

        x10_1 = self.convt4(x9_2)
        x10_2 = self.improve_ff_block1(x1, x2)
        x10_2 = torch.cat([x10_1, x10_2], dim=1)
        x10_2 = self.double_conv3(x10_2)
        x10_2 = self.add_block_exp_path4(x10_2, x2, x10_1)

        x11_1 = self.convt5(x10_2)
        x11_2 = torch.cat([x11_1, x1], dim=1)
        x11_2 = self.double_conv4(x11_2)
        x11_2 = self.add_block_exp_path5(x11_2, x1, x11_1)

        return self.final(x11_2)


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = CloudNet(in_channels=4, out_channels=4).to(device)
    x = torch.randn(size=(8, 4, 256, 256)).to(device)
    y = model(x)
    print(y.shape)
    params_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("%s Params: %.2fM" % ('CloudNet', params_num / 1e6))
    # summary(model, [(4, 256, 256), (4, 256, 256)])


