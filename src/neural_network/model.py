import torch
import torch.nn as nn

# ============================================================================
# UNET ARCHITECTURE - EXACT COPY FROM train_final_refined.py
# ============================================================================

class UNet(nn.Module):
    """
    UNet pentru Change Detection
    
    Input:  6 channels (3 before + 3 after RGB)
    Output: 1 channel (binary mask)
    
    Architecture synchronized with train_final_refined.py
    """
    
    def __init__(self, in_channels=6, out_channels=1):
        super().__init__()
        
        # ====================================================================
        # ENCODER - Downsampling path
        # ====================================================================
        self.enc1 = self.conv_block(in_channels, 64)
        self.pool1 = nn.MaxPool2d(2)
        
        self.enc2 = self.conv_block(64, 128)
        self.pool2 = nn.MaxPool2d(2)
        
        self.enc3 = self.conv_block(128, 256)
        self.pool3 = nn.MaxPool2d(2)
        
        # ====================================================================
        # BOTTLENECK - Feature extraction
        # ====================================================================
        self.bottleneck = self.conv_block(256, 512)
        
        # ====================================================================
        # DECODER - Upsampling path with skip connections
        # ====================================================================
        self.upconv3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dec3 = self.conv_block(512, 256)  # 256+256=512 from skip connection
        
        self.upconv2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec2 = self.conv_block(256, 128)  # 128+128=256 from skip connection
        
        self.upconv1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec1 = self.conv_block(128, 64)   # 64+64=128 from skip connection
        
        # ====================================================================
        # OUTPUT LAYER
        # ====================================================================
        self.out = nn.Conv2d(64, out_channels, kernel_size=1)
    
    def conv_block(self, in_ch, out_ch):
        """
        Double Convolution Block
        Conv2d -> BatchNorm -> ReLU -> Conv2d -> BatchNorm -> ReLU
        """
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        """
        Forward pass with skip connections
        
        Args:
            x: Input tensor (B, 6, H, W)
        
        Returns:
            out: Output tensor (B, 1, H, W) - logits for sigmoid
        """
        # ====================================================================
        # ENCODER - Extract features at multiple scales
        # ====================================================================
        enc1_out = self.enc1(x)           # (B, 64, H, W)
        enc1_pool = self.pool1(enc1_out)  # (B, 64, H/2, W/2)
        
        enc2_out = self.enc2(enc1_pool)   # (B, 128, H/2, W/2)
        enc2_pool = self.pool2(enc2_out)  # (B, 128, H/4, W/4)
        
        enc3_out = self.enc3(enc2_pool)   # (B, 256, H/4, W/4)
        enc3_pool = self.pool3(enc3_out)  # (B, 256, H/8, W/8)
        
        # ====================================================================
        # BOTTLENECK - Central feature extraction
        # ====================================================================
        bottleneck_out = self.bottleneck(enc3_pool)  # (B, 512, H/8, W/8)
        
        # ====================================================================
        # DECODER - Reconstruct spatial information with skip connections
        # ====================================================================
        
        # Level 3 (Decoder)
        dec3_up = self.upconv3(bottleneck_out)  # (B, 256, H/4, W/4)
        # Concatenate with skip connection from encoder
        dec3_concat = torch.cat([dec3_up, enc3_out], dim=1)  # (B, 512, H/4, W/4)
        dec3_out = self.dec3(dec3_concat)  # (B, 256, H/4, W/4)
        
        # Level 2 (Decoder)
        dec2_up = self.upconv2(dec3_out)  # (B, 128, H/2, W/2)
        # Concatenate with skip connection from encoder
        dec2_concat = torch.cat([dec2_up, enc2_out], dim=1)  # (B, 256, H/2, W/2)
        dec2_out = self.dec2(dec2_concat)  # (B, 128, H/2, W/2)
        
        # Level 1 (Decoder)
        dec1_up = self.upconv1(dec2_out)  # (B, 64, H, W)
        # Concatenate with skip connection from encoder
        dec1_concat = torch.cat([dec1_up, enc1_out], dim=1)  # (B, 128, H, W)
        dec1_out = self.dec1(dec1_concat)  # (B, 64, H, W)
        
        # ====================================================================
        # OUTPUT - Final segmentation mask (logits)
        # ====================================================================
        out = self.out(dec1_out)  # (B, 1, H, W)
        
        return out


# ============================================================================
# LEGACY COMPATIBILITY - DoubleConv class (if needed by old code)
# ============================================================================

class DoubleConv(nn.Module):
    """
    Legacy double convolution block
    Kept for backward compatibility with older scripts
    """
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)


# ============================================================================
# MODEL UTILITIES
# ============================================================================

def print_model_info(model):
    """Print model architecture and parameter count"""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print("\n" + "=" * 80)
    print("🏗️  MODEL ARCHITECTURE INFO")
    print("=" * 80)
    print(f"Model: UNet")
    print(f"Total Parameters: {total_params:,}")
    print(f"Trainable Parameters: {trainable_params:,}")
    print(f"Model Size (float32): {total_params * 4 / 1024 / 1024:.1f} MB")
    print("=" * 80 + "\n")


# ============================================================================
# TEST & VERIFICATION
# ============================================================================

if __name__ == "__main__":
    print("=" * 80)
    print("🧪 MODEL VERIFICATION - src/neural_network/model.py")
    print("=" * 80)
    
    # Test 1: Model instantiation
    print("\n[1] Instantiating UNet model...")
    model = UNet(in_channels=6, out_channels=1)
    print("    ✅ Model created successfully")
    
    # Test 2: Print info
    print_model_info(model)
    
    # Test 3: Forward pass
    print("[2] Testing forward pass...")
    x = torch.randn(1, 6, 256, 256)
    print(f"    Input shape: {x.shape}")
    
    with torch.no_grad():
        output = model(x)
    
    print(f"    Output shape: {output.shape}")
    expected_shape = (1, 1, 256, 256)
    assert output.shape == expected_shape, f"Expected {expected_shape}, got {output.shape}"
    print("    ✅ Forward pass successful")
    
    # Test 4: State dict keys (for debugging)
    print("[3] State dict keys (first 10):")
    for i, key in enumerate(list(model.state_dict().keys())[:10]):
        print(f"    {i+1}. {key}")
    
    print("\n✅ ALL TESTS PASSED - Model is ready for training/inference")
    print("=" * 80)