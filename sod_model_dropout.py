import torch
import torch.nn as nn

class UNetBlockDropout(nn.Module):
    def __init__(self, in_ch, out_ch, dropout=0.3):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),

            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class UNetDropout(nn.Module):
    def __init__(self):
        super().__init__()

        # Encoder
        self.enc1 = UNetBlockDropout(3, 32)
        self.enc2 = UNetBlockDropout(32, 64)
        self.enc3 = UNetBlockDropout(64, 128)
        self.enc4 = UNetBlockDropout(128, 256)

        self.pool = nn.MaxPool2d(2)
        self.bottleneck = UNetBlockDropout(256, 512, dropout=0.5)

        # Decoder
        self.up4 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.dec4 = UNetBlockDropout(512, 256)

        self.up3 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec3 = UNetBlockDropout(256, 128)

        self.up2 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec2 = UNetBlockDropout(128, 64)

        self.up1 = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.dec1 = UNetBlockDropout(64, 32)

        self.out = nn.Conv2d(32, 1, 1)

    def forward(self, x):
        x1 = self.enc1(x)
        x2 = self.enc2(self.pool(x1))
        x3 = self.enc3(self.pool(x2))
        x4 = self.enc4(self.pool(x3))

        b = self.bottleneck(self.pool(x4))

        d4 = self.dec4(torch.cat([self.up4(b), x4], dim=1))
        d3 = self.dec3(torch.cat([self.up3(d4), x3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), x2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), x1], dim=1))

        return torch.sigmoid(self.out(d1))
