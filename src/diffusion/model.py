"""
model.py
"""
import math
import minari
import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.quantization


class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
    
    def forward(self, t):
        # t: (batch_size, 1)
        device = t.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = t[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, time_emb_dim, dropout_rate=0.1, num_groups=32):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.norm1 = nn.GroupNorm(num_groups, in_channels)
        self.act1 = nn.SiLU()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)

        self.time_mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, out_channels)
        )

        self.norm2 = nn.GroupNorm(num_groups, out_channels)
        self.act2 = nn.SiLU()
        self.dropout = nn.Dropout(dropout_rate)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)

        if in_channels != out_channels:
            self.skip_connection = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        else:
            self.skip_connection = nn.Identity()
    
    def forward(self, x, t_emb):
        h = self.act1(self.norm1(x))
        h = self.conv1(h)

        # Add time embedding
        time_cond = self.time_mlp(t_emb)
        h = h + time_cond[:, :, None, None] # Add to channels, broadcast over H, W

        h = self.act2(self.norm2(h))
        h = self.dropout(h)
        h = self.conv2(h)

        return h + self.skip_connection(x)


class AttentionBlock(nn.Module):
    def __init__(self, channels, num_heads=8, num_groups=32):
        super().__init__()
        self.channels = channels
        self.num_heads = num_heads

        self.norm = nn.GroupNorm(num_groups, channels)
        self.qkv_proj = nn.Conv2d(channels, channels * 3, kernel_size=1) # Project to Q, K, V
        self.attention = nn.MultiheadAttention(embed_dim=channels, num_heads=num_heads, batch_first=True)
        self.out_proj = nn.Conv2d(channels, channels, kernel_size=1)

    def forward(self, x):
        B, C, H, W = x.shape
        h = self.norm(x)
        qkv = self.qkv_proj(h) # B, 3*C, H, W

        q, k, v = qkv.chunk(3, dim=1) # B, C, H, W each

        # Reshape for MultiheadAttention
        q = q.view(B, C, H * W).permute(0, 2, 1) # B, H*W, C
        k = k.view(B, C, H * W).permute(0, 2, 1) # B, H*W, C
        v = v.view(B, C, H * W).permute(0, 2, 1) # B, H*W, C

        attn_output, _ = self.attention(q, k, v) # B, H*W, C

        # Reshape back
        attn_output = attn_output.permute(0, 2, 1).view(B, C, H, W) # B, C, H, W
        attn_output = self.out_proj(attn_output)

        return x + attn_output # Residual connection


class Downsample(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        return self.conv(x)


class Upsample(nn.Module):
    def __init__(self, channels):
        super().__init__()
        # Upsample followed by conv to maintain/adjust features
        self.upsample = nn.Upsample(scale_factor=2, mode="nearest")
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, padding=1)

    def forward(self, x):
        x = self.upsample(x)
        return self.conv(x)


class ConditionalUNet(nn.Module):
    def __init__(
        self,
        in_img_channels=6,  # 3 for noisy_next_img, 3 for current_img
        out_img_channels=3, # For predicted noise
        base_channels=64,
        channel_mults=(1, 2, 3, 4),
        num_res_blocks_per_level=2,
        attention_resolutions=(120, 60), # Image resolutions where attention is applied
        dropout_rate=0.1,
        time_emb_dim=256, # Dimension of sinusoidal embedding
        time_emb_mlp_dim=1024, # Dimension of MLP for time embedding
        num_heads_attn=8,
        num_groups_norm=32,
        initial_img_resolution=120,
    ):
        super().__init__()

        self.quant = torch.quantization.QuantStub()
        self.dequant = torch.quantization.DeQuantStub()

        self.initial_img_resolution = initial_img_resolution
        self.time_emb_dim = time_emb_dim
        self.channel_mults = channel_mults

        # Time embedding
        self.time_positional_encoding = SinusoidalPositionalEncoding(time_emb_dim)
        self.time_mlp = nn.Sequential(
            nn.Linear(time_emb_dim, time_emb_mlp_dim),
            nn.SiLU(),
            nn.Linear(time_emb_mlp_dim, time_emb_mlp_dim)
        )

        # Initial convolution
        self.conv_in = nn.Conv2d(in_img_channels, base_channels, kernel_size=3, padding=1)

        # Encoder (Downsampling path)
        self.encoder_blocks = nn.ModuleList()
        self.encoder_downsamples = nn.ModuleList()
        current_channels = base_channels
        current_resolution = initial_img_resolution

        for i, mult in enumerate(channel_mults):
            out_channels_level = base_channels * mult
            level_blocks = nn.ModuleList()
            
            for _ in range(num_res_blocks_per_level):
                level_blocks.append(ResidualBlock(current_channels, out_channels_level, time_emb_mlp_dim, dropout_rate, num_groups_norm))
                current_channels = out_channels_level
            
            if current_resolution in attention_resolutions:
                level_blocks.append(AttentionBlock(current_channels, num_heads_attn, num_groups_norm))
            
            self.encoder_blocks.append(level_blocks)
            
            if i != len(channel_mults) - 1:  # Don't add downsample after the last encoder level
                self.encoder_downsamples.append(Downsample(current_channels))
                current_resolution //= 2
        
        # Bottleneck
        self.bottleneck_res1 = ResidualBlock(current_channels, current_channels, time_emb_mlp_dim, dropout_rate, num_groups_norm)
        self.bottleneck_attn = AttentionBlock(current_channels, num_heads_attn, num_groups_norm)
        self.bottleneck_res2 = ResidualBlock(current_channels, current_channels, time_emb_mlp_dim, dropout_rate, num_groups_norm)

        # Decoder (Upsampling path)
        self.decoder_blocks = nn.ModuleList()
        self.decoder_upsamples = nn.ModuleList()
        
        # Start from bottleneck output channels
        dec_current_channels = current_channels
        dec_current_resolution = initial_img_resolution // (2**(len(channel_mults)-1))

        for i, mult in reversed(list(enumerate(channel_mults))):
            target_level_channels = base_channels * mult
            skip_channels = base_channels * mult # from encoder at this resolution
            
            level_blocks = nn.ModuleList()
            # First ResidualBlock takes concatenated channels
            res_block_in_channels = dec_current_channels + skip_channels
            level_blocks.append(ResidualBlock(res_block_in_channels, target_level_channels, time_emb_mlp_dim, dropout_rate, num_groups_norm))
            
            # Subsequent ResidualBlocks
            for _ in range(num_res_blocks_per_level - 1):
                 level_blocks.append(ResidualBlock(target_level_channels, target_level_channels, time_emb_mlp_dim, dropout_rate, num_groups_norm))
            
            if dec_current_resolution in attention_resolutions:
                 level_blocks.append(AttentionBlock(target_level_channels, num_heads_attn, num_groups_norm))

            self.decoder_blocks.append(level_blocks)
            
            if i != 0:  # If not the last decoder stage (which outputs at full res)
                self.decoder_upsamples.append(Upsample(target_level_channels))
            
            dec_current_channels = target_level_channels
            dec_current_resolution *= 2

        # Final convolution
        self.conv_out_norm = nn.GroupNorm(num_groups_norm, base_channels)
        self.conv_out_act = nn.SiLU()
        self.conv_out = nn.Conv2d(base_channels, out_img_channels, kernel_size=3, padding=1)

    def forward(self, x_concat, time):
        # x_concat: [B, 6, H, W]
        # time: [B]
        
        x_q = self.quant(x_concat)

        # 1. Time embedding
        t_emb = self.time_positional_encoding(time)
        t_emb = self.time_mlp(t_emb)  # [B, time_emb_mlp_dim]

        # 2. Initial convolution
        h = self.conv_in(x_q)  # Use quantized input
        
        skip_connections = []

        # 3. Encoder
        downsample_idx = 0
        for i, mult in enumerate(self.channel_mults):
            level_blocks = self.encoder_blocks[i]
            
            for block in level_blocks:
                if isinstance(block, ResidualBlock):
                    h = block(h, t_emb)
                else:  # AttentionBlock
                    h = block(h)
            
            skip_connections.append(h)
            
            if i != len(self.channel_mults) - 1:
                h = self.encoder_downsamples[downsample_idx](h)
                downsample_idx += 1
        
        # 4. Bottleneck
        h = self.bottleneck_res1(h, t_emb)
        h = self.bottleneck_attn(h)
        h = self.bottleneck_res2(h, t_emb)

        # 5. Decoder
        upsample_idx = 0
        for i in range(len(self.channel_mults)):
            level_blocks = self.decoder_blocks[i]
            
            skip_h = skip_connections.pop()  # Pop from the end (deepest encoder skip first)
            h = torch.cat([h, skip_h], dim=1)
            
            for block in level_blocks:
                if isinstance(block, ResidualBlock):
                    h = block(h, t_emb)
                else:  # AttentionBlock
                    h = block(h)

            if i != len(self.channel_mults) - 1:  # If not the last decoder stage
                h = self.decoder_upsamples[upsample_idx](h)
                upsample_idx += 1
        
        # 6. Final output
        h = self.conv_out_norm(h)
        h = self.conv_out_act(h)
        out_noise = self.conv_out(h)
        
        out_noise = self.dequant(out_noise)
        return out_noise
