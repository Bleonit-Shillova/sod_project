import torch
import torch.nn as nn


class UNetBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),

            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class SimpleUNet(nn.Module):
    """
    UNet-like CNN from scratch:
    - Encoder: 4 Conv blocks with MaxPool
    - Decoder: 4 ConvTranspose2d upsampling + Conv blocks
    - Output: 1-channel sigmoid saliency mask
    """
    def __init__(self):
        super().__init__()

        # Encoder
        self.enc1 = UNetBlock(3,   32)
        self.enc2 = UNetBlock(32,  64)
        self.enc3 = UNetBlock(64,  128)
        self.enc4 = UNetBlock(128, 256)

        self.pool = nn.MaxPool2d(2)

        # Bottleneck
        self.bottleneck = UNetBlock(256, 512)

        # Decoder
        self.up4 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.dec4 = UNetBlock(512, 256)

        self.up3 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec3 = UNetBlock(256, 128)

        self.up2 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec2 = UNetBlock(128, 64)

        self.up1 = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.dec1 = UNetBlock(64, 32)

        # Output: 1-channel
        self.out_conv = nn.Conv2d(32, 1, kernel_size=1)

    def forward(self, x):
        # Encoder
        x1 = self.enc1(x)
        x2 = self.enc2(self.pool(x1))
        x3 = self.enc3(self.pool(x2))
        x4 = self.enc4(self.pool(x3))

        b = self.bottleneck(self.pool(x4))

        # Decoder with skip connections
        d4 = self.up4(b)
        d4 = self.dec4(torch.cat([d4, x4], dim=1))

        d3 = self.up3(d4)
        d3 = self.dec3(torch.cat([d3, x3], dim=1))

        d2 = self.up2(d3)
        d2 = self.dec2(torch.cat([d2, x2], dim=1))

        d1 = self.up1(d2)
        d1 = self.dec1(torch.cat([d1, x1], dim=1))

        return torch.sigmoid(self.out_conv(d1))
