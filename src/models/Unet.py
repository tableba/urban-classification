import torch
import torch.nn as nn


class UNet(nn.Module):
    def __init__(self, in_channels=9, num_classes=9):
        super().__init__()

        # Encoder
        self.enc1 = self.conv_block(in_channels, 32)
        self.enc2 = self.conv_block(32, 64)
        self.enc3 = self.conv_block(64, 128)
        self.enc4 = self.conv_block(128, 256)

        self.pool = nn.MaxPool2d(2)

        # Bottleneck
        self.bottleneck = self.conv_block(256, 512)

        # Decoder
        self.up1 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dec1 = self.conv_block(512, 256)  # 512 = 256 (up1) + 256 (enc4 skip)

        self.up2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec2 = self.conv_block(256, 128)  # 256 = 128 (up2) + 128 (enc3 skip)

        self.up3 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec3 = self.conv_block(128, 64)   # 128 = 64  (up3) + 64  (enc2 skip)

        self.up4 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.dec4 = self.conv_block(64, 32)    # 64  = 32  (up4) + 32  (enc1 skip)

        # Classifier
        self.classifier = nn.Conv2d(32, num_classes, kernel_size=1)

    def conv_block(self, in_ch, out_ch):
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)                # [B,  32, H,    W]
        e2 = self.enc2(self.pool(e1))    # [B,  64, H/2,  W/2]
        e3 = self.enc3(self.pool(e2))    # [B, 128, H/4,  W/4]
        e4 = self.enc4(self.pool(e3))    # [B, 256, H/8,  W/8]

        # Bottleneck
        b = self.bottleneck(self.pool(e4))  # [B, 512, H/16, W/16]

        # Decoder with skip connections
        d1 = self.dec1(torch.cat([self.up1(b),  e4], dim=1))  # [B, 256, H/8,  W/8]
        d2 = self.dec2(torch.cat([self.up2(d1), e3], dim=1))  # [B, 128, H/4,  W/4]
        d3 = self.dec3(torch.cat([self.up3(d2), e2], dim=1))  # [B,  64, H/2,  W/2]
        d4 = self.dec4(torch.cat([self.up4(d3), e1], dim=1))  # [B,  32, H,    W]

        return self.classifier(d4)  # [B, num_classes, H, W]