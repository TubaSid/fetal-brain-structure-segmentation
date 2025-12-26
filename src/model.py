"""U-Net with Attention Gates for medical image segmentation."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class AttentionBlock(nn.Module):
    """Attention Gate for U-Net skip connections."""
    
    def __init__(self, F_g: int, F_l: int, F_int: int):
        """
        Args:
            F_g: Gates channels (from decoder)
            F_l: Skip connection channels (from encoder)
            F_int: Intermediate channels
        """
        super().__init__()
        self.conv_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, padding=0),
            nn.BatchNorm2d(F_int)
        )
        self.conv_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, padding=0),
            nn.BatchNorm2d(F_int)
        )
        self.conv_psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, padding=0),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, g, x):
        """
        Args:
            g: Gates (decoder feature)
            x: Skip connection (encoder feature)
            
        Returns:
            Attention-weighted skip connection
        """
        g1 = self.conv_g(g)
        x1 = self.conv_x(x)
        psi = self.relu(g1 + x1)
        psi = self.conv_psi(psi)
        return x * psi


class DoubleConv(nn.Module):
    """Double convolution block."""
    
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
    
    def forward(self, x):
        return self.conv(x)


class UNet(nn.Module):
    """U-Net with Attention Gates."""
    
    def __init__(self, in_channels: int = 1, n_classes: int = 4, base_filters: int = 64):
        """
        Args:
            in_channels: Number of input channels
            n_classes: Number of output classes
            base_filters: Base number of filters
        """
        super().__init__()
        
        self.in_channels = in_channels
        self.n_classes = n_classes
        
        # Encoder
        self.enc1 = DoubleConv(in_channels, base_filters)
        self.pool1 = nn.MaxPool2d(2, 2)
        
        self.enc2 = DoubleConv(base_filters, base_filters * 2)
        self.pool2 = nn.MaxPool2d(2, 2)
        
        self.enc3 = DoubleConv(base_filters * 2, base_filters * 4)
        self.pool3 = nn.MaxPool2d(2, 2)
        
        self.enc4 = DoubleConv(base_filters * 4, base_filters * 8)
        self.pool4 = nn.MaxPool2d(2, 2)
        
        # Bottleneck
        self.bottleneck = DoubleConv(base_filters * 8, base_filters * 16)
        
        # Decoder
        self.upconv4 = nn.ConvTranspose2d(base_filters * 16, base_filters * 8, kernel_size=2, stride=2)
        self.att4 = AttentionBlock(base_filters * 8, base_filters * 8, base_filters * 4)
        self.dec4 = DoubleConv(base_filters * 16, base_filters * 8)
        
        self.upconv3 = nn.ConvTranspose2d(base_filters * 8, base_filters * 4, kernel_size=2, stride=2)
        self.att3 = AttentionBlock(base_filters * 4, base_filters * 4, base_filters * 2)
        self.dec3 = DoubleConv(base_filters * 8, base_filters * 4)
        
        self.upconv2 = nn.ConvTranspose2d(base_filters * 4, base_filters * 2, kernel_size=2, stride=2)
        self.att2 = AttentionBlock(base_filters * 2, base_filters * 2, base_filters)
        self.dec2 = DoubleConv(base_filters * 4, base_filters * 2)
        
        self.upconv1 = nn.ConvTranspose2d(base_filters * 2, base_filters, kernel_size=2, stride=2)
        self.att1 = AttentionBlock(base_filters, base_filters, base_filters // 2)
        self.dec1 = DoubleConv(base_filters * 2, base_filters)
        
        # Output
        self.final = nn.Conv2d(base_filters, n_classes, kernel_size=1)
    
    def forward(self, x):
        """Forward pass."""
        # Encoder
        e1 = self.enc1(x)
        p1 = self.pool1(e1)
        
        e2 = self.enc2(p1)
        p2 = self.pool2(e2)
        
        e3 = self.enc3(p2)
        p3 = self.pool3(e3)
        
        e4 = self.enc4(p3)
        p4 = self.pool4(e4)
        
        # Bottleneck
        b = self.bottleneck(p4)
        
        # Decoder with attention
        d4 = self.upconv4(b)
        e4_att = self.att4(d4, e4)
        d4 = self.dec4(torch.cat([d4, e4_att], dim=1))
        
        d3 = self.upconv3(d4)
        e3_att = self.att3(d3, e3)
        d3 = self.dec3(torch.cat([d3, e3_att], dim=1))
        
        d2 = self.upconv2(d3)
        e2_att = self.att2(d2, e2)
        d2 = self.dec2(torch.cat([d2, e2_att], dim=1))
        
        d1 = self.upconv1(d2)
        e1_att = self.att1(d1, e1)
        d1 = self.dec1(torch.cat([d1, e1_att], dim=1))
        
        # Output
        logits = self.final(d1)
        return logits
