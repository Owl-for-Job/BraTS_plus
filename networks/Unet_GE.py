import torch
import torch.nn as nn

# GE模块的全局增强部分
class GlobalEnhancement(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(GlobalEnhancement, self).__init__()
        self.global_pool = nn.AdaptiveAvgPool3d(1)
        self.conv = nn.Conv3d(in_ch, out_ch, kernel_size=1)

    def forward(self, x):
        x_pool = self.global_pool(x)
        x = self.conv(x_pool)
        return x

# GE模块的局部增强部分
class LocalEnhancement(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(LocalEnhancement, self).__init__()
        self.conv = nn.Conv3d(in_ch, out_ch, kernel_size=3, padding=1)

    def forward(self, x):
        x = self.conv(x)
        return x

# 将GE模块添加到UNet_GE中
class UNet_GE(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(UNet_GE, self).__init__()
        features = [32,64,128,256]
        # 通过连续的 Down 模块提取特征并降低分辨率。
        self.inc = InConv(in_channels, features[0])
        self.ge1 = GlobalEnhancement(features[0], features[0])
        self.down1 = Down(features[0], features[1])
        self.ge2 = GlobalEnhancement(features[1], features[1])
        self.down2 = Down(features[1], features[2])
        self.ge3 = GlobalEnhancement(features[2], features[2])
        self.down3 = Down(features[2], features[3])
        self.ge4 = GlobalEnhancement(features[3], features[3])
        self.down4 = Down(features[3], features[3])
        # 通过连续的 Up 模块恢复分辨率并融合编码器的特征。
        self.up1 = Up(features[3], features[3], features[2])
        self.ge5 = GlobalEnhancement(features[2], features[2])
        self.up2 = Up(features[2], features[2], features[1])
        self.ge6 = GlobalEnhancement(features[1], features[1])
        self.up3 = Up(features[1], features[1], features[0])
        self.ge7 = GlobalEnhancement(features[0], features[0])
        self.up4 = Up(features[0], features[0], features[0])
        self.outc = OutConv(features[0], num_classes)

    def forward(self, x):
        x1 = self.inc(x)
        ge1 = self.ge1(x1)
        x2 = self.down1(x1)
        ge2 = self.ge2(x2)
        x3 = self.down2(x2)
        ge3 = self.ge3(x3)
        x4 = self.down3(x3)
        ge4 = self.ge4(x4)
        x5 = self.down4(x4)

        x = self.up1(x5, x4)
        ge5 = self.ge5(x)
        x = self.up2(x, x3)
        ge6 = self.ge6(x)
        x = self.up3(x, x2)
        ge7 = self.ge7(x)
        x = self.up4(x, x1)
        x = self.outc(x)
        return x

# 其余部分与原来的UNet_GE保持一致

# InConv 模块是 U-Net 的输入卷积模块。它使用 DoubleConv 进行两次卷积操作，以增加特征图的复杂性。
class InConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(InConv, self).__init__()
        self.conv = DoubleConv(in_ch, out_ch)

    def forward(self, x):
        x = self.conv(x)
        return x

# Down 模块通过 3D 最大池化（MaxPool3d）来减少特征图的分辨率，并使用 DoubleConv 提取更高层次的特征。
class Down(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(Down, self).__init__()
        self.mpconv = nn.Sequential(
            nn.MaxPool3d(2, 2),
            DoubleConv(in_ch, out_ch)
        )

    def forward(self, x):
        x = self.mpconv(x)
        return x

# OutConv 模块是 U-Net 的输出卷积模块。它使用一个 1x1 的卷积层将通道数转换为所需的输出类别数。
class OutConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(OutConv, self).__init__()
        self.conv = nn.Conv3d(in_ch, out_ch, 1)
        # self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.conv(x)
        # x = self.sigmoid(x)
        return x

# DoubleConv 模块进行两次连续的 3D 卷积操作，并在每次卷积后使用批归一化和ReLU激活函数。
class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_ch, out_ch, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x

# Up 模块通过 3D 转置卷积（ConvTranspose3d）将特征图上采样，并将上采样的特征图与编码器中的跳跃连接特征图进行拼接，然后通过 DoubleConv 进一步处理。
class Up(nn.Module):
    def __init__(self, in_ch, skip_ch, out_ch):
        super(Up, self).__init__()
        self.up = nn.ConvTranspose3d(in_ch, in_ch, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_ch + skip_ch, out_ch)

        def forward(self, x1, x2):
            x1 = self.up(x1)
            x = torch.cat([x2, x1], dim=1)
            x = self.conv(x)
            return x

    if __name__ == '__main__':
        x = torch.randn(1, 4, 160, 160, 128)
        net = UNet_GE(in_channels=4, num_classes=4)
        y = net(x)
        print("params: ", sum(p.numel() for p in net.parameters()))
        print(y.shape)

