# -*- coding: utf-8 -*-
# @Time    : 2023/4/1 11:46
# @Author  : Lin Junhong
# @FileName: LDRM.py
# @Software: PyCharm
# @E_mails ：SPJLinn@163.com

import torch
from torch import nn
from torch.nn import functional as F
from timm.models.layers import DropPath
from ptflops import get_model_complexity_info


# ======================================================================================================================
class units:
    class ChannelAttention(nn.Module):
        def __init__(self, k_size=7):
            super().__init__()
            self.avgpool = nn.AdaptiveAvgPool2d(1)
            self.maxpool = nn.AdaptiveAvgPool2d(1)
            self.conv1d = nn.Conv1d(2, 1, kernel_size=k_size, padding=(k_size // 2), bias=False)
            self.sigmoid = nn.Sigmoid()

        def forward(self, x):
            y_avg = self.avgpool(x)
            y_max = self.avgpool(x)
            y = torch.cat([y_avg, y_max], dim=2)
            y = self.conv1d(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
            y = self.sigmoid(y.float())
            return x * y.expand_as(x)

    class ResBlock(nn.Module):
        def __init__(self, channels):
            super().__init__()
            self.res = nn.Sequential(
                nn.Conv2d(channels, channels // 2, kernel_size=3, padding=1, bias=False),
                nn.LeakyReLU(inplace=True),
                nn.Conv2d(channels // 2, channels, kernel_size=3, padding=1, bias=False),
            )

        def forward(self, x):
            return x + self.res(x)

    @staticmethod
    def Squeeze(channels):
        return nn.Conv2d(channels * 2, channels, 1, bias=False)

    @staticmethod
    def UpDownSample(ch_in, ch_out, scale):
        return nn.Sequential(nn.Upsample(scale_factor=scale, mode='bicubic', align_corners=False),
                             nn.Conv2d(ch_in, ch_out, kernel_size=1, bias=False))


# ----------------------------------------------------------------------------------------------------------------------
class LatentCodeExtractModule(nn.Module):
    def __init__(self, ch_z, space, mode):
        super().__init__()
        self.space = space
        self.mode = mode
        channels = ch_z // 2
        assert self.mode == 'Input' or 'GT'

        if space == 'Y':
            ch_in = 1
            self.afm = self.AlignModule(channels=ch_in, ch_hidden=16, k_size=7)
        elif space == 'UV':
            ch_in = 2
        if mode == 'GT':
            ch_res = 2
        elif mode == 'Input':
            ch_res = 1

        self.feat = nn.Conv2d(ch_in, channels, kernel_size=3, padding=1, bias=False)

        if (self.mode == 'Input' and self.space == 'UV') is not True:
            self.res_feat = nn.Conv2d(ch_res, channels, kernel_size=3, padding=1, bias=False)

        self.res = nn.Sequential(
            units.ResBlock(channels),
            nn.Upsample(scale_factor=0.5, mode='bicubic', align_corners=False),
            units.ResBlock(channels),
            nn.Upsample(scale_factor=0.5, mode='bicubic', align_corners=False),
            units.ResBlock(channels),
            nn.Conv2d(channels, ch_z, kernel_size=1, padding=0, bias=False)
        )
        self.avg = nn.AdaptiveAvgPool2d((1, 1))
        self.max = nn.AdaptiveMaxPool2d((1, 1))
        self.conv1d = nn.Conv1d(ch_z, ch_z, kernel_size=2, padding=0, bias=False)

    class AlignModule(nn.Module):
        def __init__(self, channels, ch_hidden=32, k_size=3):
            super().__init__()
            self.grid = nn.Sequential(
                nn.Conv2d(channels, ch_hidden, kernel_size=k_size, padding=k_size // 2, bias=False),
                nn.ReLU(inplace=True),
                nn.Conv2d(ch_hidden, 2, kernel_size=k_size, padding=k_size // 2, bias=False),
            )

        def forward(self, Yc, Ym):
            grid = self.grid(Yc).permute(0, 2, 3, 1).contiguous()
            align_Ym = F.grid_sample(Ym, grid, mode='nearest', align_corners=True)
            return align_Ym

    def forward_GT(self, x, x_GT):
        if self.space == 'Y':
            x1, Ym = x[0], x[1]
            xs = x1
            res_GT = x_GT - xs
            if Ym is not None:
                Y_align = self.afm(xs, Ym)
                res_Ym = Y_align - xs
                res = torch.cat([res_GT, res_Ym], dim=1)
                x_feat = self.feat(x1)
                res_feat = self.res_feat(res)
                z_feat = x_feat + res_feat
            else:
                z_feat = self.feat(x1)

        elif self.space == 'UV':
            x1 = x
            xs = x1
            res = x_GT - xs

            x_feat = self.feat(x1)
            res_feat = self.res_feat(res)
            z_feat = x_feat + res_feat

        return z_feat

    def reverse(self, x):
        if self.space == 'Y':
            Yc, Ym = x[0], x[1]
            xs = Yc
            if Ym is not None:
                Y_align = self.afm(xs, Ym)
                res = Y_align - xs

                x_feat = self.feat(xs)
                res_feat = self.res_feat(res)
                z_feat = x_feat + res_feat
            else:
                z_feat = self.feat(xs)

        elif self.space == 'UV':
            UVc = x
            xs = UVc

            z_feat = self.feat(xs)

        return z_feat

    def forward(self, x, x_GT=None):
        if self.mode == 'GT':
            z_feat = self.forward_GT(x, x_GT)
        elif self.mode == 'Input':
            z_feat = self.reverse(x)
        z_feat = self.res(z_feat)
        z_avg = self.avg(z_feat)
        z_max = self.max(z_feat)
        z = torch.cat((z_avg, z_max), dim=2).squeeze(-1)
        z = self.conv1d(z).unsqueeze(-1)
        # z = self.out(torch.cat((z_avg, z_max), dim=1))
        return z


class CodeEstimateNet(nn.Module):
    def __init__(self, ch_inout=64, ch_hidden=256):
        super().__init__()

        self.linears = nn.Sequential(
            nn.Linear(ch_inout, ch_hidden),
            nn.LeakyReLU(True),
            nn.Linear(ch_hidden, ch_hidden * 2),
            nn.Linear(ch_hidden * 2, ch_hidden * 2),
            nn.Linear(ch_hidden * 2, ch_hidden),
            nn.LeakyReLU(True),
            nn.Linear(ch_hidden, ch_inout),
            nn.LeakyReLU(True),
        )

    def forward(self, z):
        z = z.view(z.shape[0], -1)
        x = self.linears(z)
        return x


# ----------------------------------------------------------------------------------------------------------------------
class AdditivityRectifyNet(nn.Module):
    def __init__(self, ch_img=3, channels=32, groups=1):
        super().__init__()
        self.channels = channels
        self.img_in = nn.Conv2d(ch_img, channels, kernel_size=1, bias=False)
        self.img_out = nn.Conv2d(channels, ch_img, kernel_size=1, groups=1, bias=False)

        self.enc = nn.ModuleList([
            self.Block(channels, k_size=3, groups=groups),
            self.Block(channels, k_size=3, groups=groups),
            self.Block(channels, k_size=3, groups=groups)
        ])

        self.dec = nn.ModuleList([
            self.Block(channels, k_size=3, groups=groups),
            self.Block(channels, k_size=3, groups=groups),
            self.Block(channels, k_size=3, groups=groups)
        ])

        self.squ = nn.ModuleList([
            units.Squeeze(channels),
            units.Squeeze(channels)
        ])

        self.down = nn.ModuleList([
            nn.Upsample(scale_factor=0.5, mode='bicubic', align_corners=False),
            nn.Upsample(scale_factor=0.5, mode='bicubic', align_corners=False),
        ])

        self.up = nn.ModuleList([
            nn.Upsample(scale_factor=2, mode='bicubic', align_corners=False),
            nn.Upsample(scale_factor=2, mode='bicubic', align_corners=False),
        ])

    class Block(nn.Module):
        def __init__(self, channels, k_size, groups):
            super().__init__()
            self.res = nn.Sequential(
                nn.Conv2d(channels, channels, kernel_size=k_size, padding=(k_size // 2), bias=False, groups=groups),
                nn.Conv2d(channels, channels, kernel_size=1, bias=False),
                units.ChannelAttention(),
                nn.ReLU(True)
            )

        def forward(self, x):
            return x + self.res(x)

    def forward(self, noise_img):
        feats = self.img_in(noise_img)

        enc1 = self.enc[0](feats)
        enc2 = self.enc[1](self.down[0](enc1))
        enc3 = self.enc[2](self.down[1](enc2))
        dec3 = self.dec[2](enc3)
        dec2 = self.squ[1](torch.cat((self.up[1](dec3), enc2), dim=1))
        dec2 = self.dec[1](dec2)
        dec1 = self.squ[0](torch.cat((self.up[0](dec2), enc1), dim=1))
        dec = self.dec[0](dec1)

        noise = self.img_out(dec)
        # noise, ld1 = torch.chunk(out_feats, 2, dim=1)
        # ld1 = torch.exp(-nn.ReLU()(ld1))
        return noise
        # return noise, ld1


class CodeEmbedConvModule(nn.Module):
    def __init__(self, channels, ch_z, k_size=3, mlp_ratio=4, groups=1, REF=True, CODE=True):
        super().__init__()
        if CODE:
            self.z1 = nn.Conv2d(ch_z, channels, kernel_size=1, bias=False)
            self.z2 = nn.Conv2d(ch_z, channels, kernel_size=1, bias=False)

        self.res = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=k_size, padding=(k_size // 2), bias=False, groups=groups),
            nn.Conv2d(channels, channels, kernel_size=1, bias=False),
            units.ChannelAttention(),
            nn.LeakyReLU(inplace=True),
        )

        self.norm1 = nn.BatchNorm2d(channels)
        self.norm2 = nn.BatchNorm2d(channels)
        self.mlp = self.MLP(channels=channels, mlp_ratio=mlp_ratio)

        if REF:
            self.afm = self.AlignFusionModule(channels=channels)

    class AlignFusionModule(nn.Module):
        def __init__(self, channels, ch_hidden=32, k_size=3):
            super().__init__()
            self.grid = nn.Sequential(
                # units.ResBlock(channels),
                nn.Conv2d(channels, ch_hidden, kernel_size=k_size, padding=k_size // 2, bias=False),
                nn.ReLU(inplace=True),
                nn.Conv2d(ch_hidden, 2, kernel_size=k_size, padding=k_size // 2, bias=False),
                # nn.ReLU(inplace=True),
            )

            self.sigmoid = nn.Sigmoid()
            self.conv = nn.Conv2d(channels * 2, channels, kernel_size=1, padding=0, bias=False)

        def forward(self, Yc, Ym):
            grid = self.grid(Yc).permute(0, 2, 3, 1).contiguous()
            align_Ym = F.grid_sample(Ym, grid, mode='nearest', align_corners=True)
            z = self.sigmoid(self.conv(torch.cat([Yc, align_Ym], dim=1)))
            Y = Yc * z + align_Ym * (1 - z)
            return Y

    class MLP(nn.Module):
        def __init__(self, channels, mlp_ratio=1):
            super().__init__()
            self.fc1 = nn.Conv2d(channels, int(channels * mlp_ratio), kernel_size=1, bias=False)
            self.act = nn.LeakyReLU(inplace=True)
            self.fc2 = nn.Conv2d(int(channels * mlp_ratio), channels, kernel_size=1, bias=False)

        def forward(self, x):
            return self.fc2(self.act(self.fc1(x)))

    def forward(self, x, z=None, Y=None):
        if Y is not None:
            Y_align = self.afm(x, Y)
            x = x + Y_align
        if z is None:
            x = x + self.res(self.norm1(x))
            x = x + self.mlp(self.norm2(x))
        else:
            z = z.unsqueeze(-1).unsqueeze(-1)
            z1 = self.z1(z)
            z2 = self.z2(z)
            x = x + self.res(self.norm1(x) + z1)
            x = x + self.mlp(self.norm2(x) + z2)

        if Y is not None:
            return x, Y_align
        else:
            return x


class CodeEmbedTransModule(nn.Module):
    def __init__(self, dim, ch_z, num_heads=8, split_size=8, mlp_ratio=4,
                 drop=0., attn_drop=0., drop_path=0., mode='Y', REF=True, CODE=True):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.split_size = split_size
        if mode == 'Y' and REF:
            self.kv = nn.Linear(dim, dim * 2, bias=False)
            self.q = nn.Linear(dim, dim * 1, bias=False)
        else:
            self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(drop)

        if CODE:
            self.z1 = nn.Linear(ch_z, dim, bias=False)
            self.z2 = nn.Linear(ch_z, dim, bias=False)

        self.sigmoid = nn.Sigmoid()

        self.attns = nn.ModuleList([
            self.CSwinAttention(dim // 2, idx=i, split_size=split_size,
                                num_heads=num_heads // 2, dim_out=dim // 2, attn_drop=attn_drop)
            for i in range(2)])

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.mlp = self.MLP(in_features=dim, hidden_features=int(dim * mlp_ratio), out_features=dim, drop=drop)

        self.afm = self.AlignFusionModule(channels=dim)

    class AlignFusionModule(nn.Module):
        def __init__(self, channels, ch_hidden=32, k_size=3):
            super().__init__()
            self.grid = nn.Sequential(
                # units.ResBlock(channels),
                nn.Conv2d(channels, ch_hidden, kernel_size=k_size, padding=k_size // 2, bias=False),
                nn.ReLU(inplace=True),
                nn.Conv2d(ch_hidden, 2, kernel_size=k_size, padding=k_size // 2, bias=False),
                # nn.ReLU(inplace=True),
            )

            self.sigmoid = nn.Sigmoid()
            self.conv = nn.Conv2d(channels * 2, channels, kernel_size=1, padding=0, bias=False)

        def forward(self, grid, Yc, Ym):
            grid = self.grid(grid).permute(0, 2, 3, 1).contiguous()
            align_Ym = F.grid_sample(Ym, grid, mode='nearest', align_corners=True)
            z = self.sigmoid(self.conv(torch.cat([Yc, align_Ym], dim=1)))
            Y = Yc * z + align_Ym * (1 - z)
            return Y

    class MLP(nn.Module):
        def __init__(self, in_features, hidden_features=None, out_features=None, drop=0.):
            super().__init__()
            out_features = out_features or in_features
            hidden_features = hidden_features or in_features
            self.fc1 = nn.Linear(in_features, hidden_features)
            self.act = nn.GELU()
            self.fc2 = nn.Linear(hidden_features, out_features)
            self.drop = nn.Dropout(drop)

        def forward(self, x):
            x = self.fc1(x)
            x = self.act(x)
            x = self.drop(x)
            x = self.fc2(x)
            x = self.drop(x)
            return x

    class CSwinAttention(nn.Module):
        def __init__(self, dim, idx, split_size=7, dim_out=None, num_heads=8, attn_drop=0.):
            super().__init__()
            self.dim = dim
            self.dim_out = dim_out or dim
            self.split_size = split_size
            self.num_heads = num_heads
            head_dim = dim // num_heads
            self.scale = head_dim ** -0.5
            self.idx = idx
            self.get_v = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim)

            self.attn_drop = nn.Dropout(attn_drop)

        def img2cswin(self, x, H, W, H_sp, W_sp):
            B, N, C = x.shape
            x = x.transpose(-2, -1).contiguous().view(B, C, H, W)
            img_reshape = x.view(B, C, H // H_sp, H_sp, W // W_sp, W_sp)
            x = img_reshape.permute(0, 2, 4, 3, 5, 1).contiguous().reshape(-1, H_sp * W_sp, C)

            x = x.reshape(-1, H_sp * W_sp, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3).contiguous()
            return x

        def cswin2img(self, img_splits_hw, H_sp, W_sp, H, W):
            """
            img_splits_hw: B' H W C
            """
            B = int(img_splits_hw.shape[0] / (H * W / H_sp / W_sp))

            img = img_splits_hw.view(B, H // H_sp, W // W_sp, H_sp, W_sp, -1)
            img = img.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
            return img

        def get_lepe(self, x, func, H, W, H_sp, W_sp):
            B, N, C = x.shape
            x = x.transpose(-2, -1).contiguous().view(B, C, H, W)
            x = x.view(B, C, H // H_sp, H_sp, W // W_sp, W_sp)
            x = x.permute(0, 2, 4, 1, 3, 5).contiguous().reshape(-1, C, H_sp, W_sp)  ### B', C, H', W'

            lepe = func(x)  ### B', C, H', W'
            lepe = lepe.reshape(-1, self.num_heads, C // self.num_heads, H_sp * W_sp).permute(0, 1, 3, 2).contiguous()

            x = x.reshape(-1, self.num_heads, C // self.num_heads, H_sp * W_sp).permute(0, 1, 3, 2).contiguous()
            return x, lepe

        def forward(self, qkv, H, W):
            """
            x: B L C
            """
            if self.idx == 0:
                H_sp, W_sp = H, self.split_size
            elif self.idx == 1:
                W_sp, H_sp = W, self.split_size
            else:
                print("ERROR MODE", self.idx)
                exit(0)

            q, k, v = qkv[0], qkv[1], qkv[2]
            B, L, C = q.shape
            k = self.img2cswin(k, H, W, H_sp, W_sp)
            v = self.img2cswin(v, H, W, H_sp, W_sp)
            q, lepe = self.get_lepe(q, self.get_v, H, W, H_sp, W_sp)

            q = q * self.scale
            attn = (q @ k.transpose(-2, -1))  # B head N C @ B head C N --> B head N N
            attn = nn.functional.softmax(attn, dim=-1, dtype=attn.dtype)
            attn = self.attn_drop(attn)

            x = (attn @ v) + lepe
            x = x.transpose(1, 2).reshape(-1, H_sp * W_sp, C)  # B head N N @ B head N C

            # Window2Img
            x = self.cswin2img(x, H_sp, W_sp, H, W).view(B, -1, C)  # B H' W' C

            return x

    def forward(self, x, z=None, Y=None):
        B, C, H, W = x.shape
        x = x.reshape(B, C, -1).permute(0, 2, 1).contiguous()

        if z is None:
            x = self.norm1(x)
        else:
            z = z.unsqueeze(-2)
            z1 = self.z1(z)
            x = self.norm1(x) + z1

        if Y is not None:
            Ym = Y.reshape(B, C, -1).permute(0, 2, 1).contiguous()
            kv = self.kv(Ym).reshape(B, -1, 2, C).permute(2, 0, 1, 3)
            q = self.q(x).reshape(B, -1, 1, C).permute(2, 0, 1, 3)
            qkv = torch.cat([q, kv], dim=0)
        else:
            qkv = self.qkv(x).reshape(B, -1, 3, C).permute(2, 0, 1, 3)

        x1 = self.attns[0](qkv[:, :, :, :C // 2], H, W)
        x2 = self.attns[1](qkv[:, :, :, C // 2:], H, W)
        attened_x = torch.cat([x1, x2], dim=2)
        attened_x = self.proj(attened_x)
        x = x + self.drop_path(attened_x)

        if Y is not None:
            attened_grid = attened_x.permute(0, 2, 1).contiguous().view(B, C, H, W)
            Yc = x.permute(0, 2, 1).contiguous().view(B, C, H, W)
            Y_align = self.afm(attened_grid, Yc, Y)
            x = (Y_align + Yc).reshape(B, C, -1).permute(0, 2, 1).contiguous()

        if z is None:
            x = x + self.drop_path(self.mlp(self.norm2(x)))
        else:
            z2 = self.z2(z)
            x = x + self.drop_path(self.mlp(self.norm2(x) + z2))

        x = x.permute(0, 2, 1).contiguous().view(B, C, H, W)

        if Y is not None:
            return x, Y_align
        else:
            return x


class DegradationRectifyNet(nn.Module):
    def __init__(self, ch_img=3, ch_z=64, ch_level=32, ch_add=16, k_size=3,
                 mlp_ratio=4, groups=1, REF=True, CODE=True):
        super().__init__()
        # Configs
        self.num_head = 8
        self.split_size = 4
        self.ch_level = ch_level
        self.ch_level1 = ch_level + ch_add
        self.ch_level2 = ch_level + ch_add * 2
        self.ch_level3 = self.ch_level2 * 2
        self.REF = REF
        self.CODE = CODE

        # In_feats
        self.feat = nn.ModuleList([
            nn.ModuleList([
                nn.Conv2d(2, ch_level, kernel_size=k_size, padding=(k_size // 2), bias=False, groups=groups),
            ]),
            nn.ModuleList([
                nn.Conv2d(1, ch_level, kernel_size=k_size, padding=(k_size // 2), bias=False, groups=groups),
            ])
        ])
        if self.REF:
            self.feat.append(
                nn.Conv2d(1, ch_level, kernel_size=k_size, padding=(k_size // 2), bias=False, groups=groups)
            )

        # UpDownSample & Squeeze
        self.UVdown = self.UpDown_List(mode='Down')
        self.Ydown = self.UpDown_List(mode='Down')
        self.enc_UVdown = self.UpDown_List(mode='Down')
        self.enc_Ydown = self.UpDown_List(mode='Down')
        self.dec_Yup = self.UpDown_List(mode='Up')
        self.dec_UVup = self.UpDown_List(mode='Up')
        self.squ = nn.Conv2d(ch_level * 2, ch_level, kernel_size=1, bias=False)

        # Encoder Y/UV & Decoder
        self.enc_UV = self.Module_List(ch_z, mlp_ratio, groups, mode='UV')
        self.enc_Y = self.Module_List(ch_z, mlp_ratio, groups, mode='Y')
        self.dec_UV = self.Module_List(ch_z, mlp_ratio, groups, mode='UV')
        self.dec_Y = self.Module_List(ch_z, mlp_ratio, groups, mode='Y')

        # Pixel Detail Enhance & Out
        self.drm = self.DetailRefineModule(ch_level, 7, ch_level)
        self.out = nn.Conv2d(ch_level, ch_img, 1, groups=1, bias=False)

    def Module_List(self, ch_z, mlp_ratio, groups, mode):
        if mode == 'UV':
            return nn.ModuleList([
                nn.ModuleList([
                    CodeEmbedConvModule(self.ch_level, ch_z, 7, mlp_ratio, groups, False, self.CODE),
                    CodeEmbedConvModule(self.ch_level, ch_z, 7, mlp_ratio, groups, False, self.CODE)
                ]),
                nn.ModuleList([
                    CodeEmbedConvModule(self.ch_level1, ch_z, 5, mlp_ratio, groups, False, self.CODE),
                    CodeEmbedConvModule(self.ch_level1, ch_z, 5, mlp_ratio, groups, False, self.CODE)
                ]),
                nn.ModuleList([
                    CodeEmbedConvModule(self.ch_level2, ch_z, 3, mlp_ratio, groups, False, self.CODE),
                    CodeEmbedConvModule(self.ch_level2, ch_z, 3, mlp_ratio, groups, False, self.CODE)
                ]),
                nn.ModuleList([
                    CodeEmbedTransModule(self.ch_level3, ch_z, self.num_head * 4, self.split_size, mlp_ratio,
                                         mode=mode, REF=False, CODE=self.CODE),
                    CodeEmbedTransModule(self.ch_level3, ch_z, self.num_head * 4, self.split_size, mlp_ratio,
                                         mode=mode, REF=False, CODE=self.CODE)
                ])
            ])
        elif mode == 'Y':
            return nn.ModuleList([
                nn.ModuleList([
                    CodeEmbedConvModule(self.ch_level, ch_z, 7, mlp_ratio, groups, self.REF, self.CODE),
                    CodeEmbedConvModule(self.ch_level, ch_z, 7, mlp_ratio, groups, self.REF, self.CODE)
                ]),
                nn.ModuleList([
                    CodeEmbedConvModule(self.ch_level1, ch_z, 5, mlp_ratio, groups, self.REF, self.CODE),
                    CodeEmbedConvModule(self.ch_level1, ch_z, 5, mlp_ratio, groups, self.REF, self.CODE)
                ]),
                nn.ModuleList([

                    CodeEmbedConvModule(self.ch_level2, ch_z, 3, mlp_ratio, groups, self.REF, self.CODE),
                    CodeEmbedConvModule(self.ch_level2, ch_z, 3, mlp_ratio, groups, self.REF, self.CODE)
                ]),
                nn.ModuleList([
                    CodeEmbedTransModule(self.ch_level3, ch_z, self.num_head * 4, self.split_size, mlp_ratio,
                                         mode=mode, REF=self.REF, CODE=self.CODE),
                    CodeEmbedTransModule(self.ch_level3, ch_z, self.num_head * 4, self.split_size, mlp_ratio,
                                         mode=mode, REF=self.REF, CODE=self.CODE)
                ])
            ])

    def UpDown_List(self, mode='Down'):
        if mode == 'Down':
            return nn.ModuleList([
                units.UpDownSample(self.ch_level, self.ch_level1, 0.5),
                units.UpDownSample(self.ch_level1, self.ch_level2, 0.5),
                units.UpDownSample(self.ch_level2, self.ch_level3, 0.5)
            ])
        elif mode == 'Up':
            return nn.ModuleList([
                units.UpDownSample(self.ch_level1, self.ch_level, 2),
                units.UpDownSample(self.ch_level2, self.ch_level1, 2),
                units.UpDownSample(self.ch_level3, self.ch_level2, 2)
            ])

    class DetailRefineModule(nn.Module):
        def __init__(self, channels, k_size, groups):
            super().__init__()
            self.res = nn.Sequential(
                self.DetailBlock(channels, k_size, groups),
                self.DetailBlock(channels, k_size, groups),
                self.DetailBlock(channels, k_size, groups),
            )

        class DetailBlock(nn.Module):
            def __init__(self, channels, k_size, groups):
                super().__init__()
                self.pdwconv = nn.Sequential(
                    nn.Conv2d(channels, channels, kernel_size=k_size, padding=(k_size // 2), bias=False, groups=groups),
                    nn.Conv2d(channels, channels, kernel_size=1, bias=False)
                )
                self.ca = units.ChannelAttention()
                self.act = nn.LeakyReLU(inplace=True)

            def forward(self, x):
                x = self.pdwconv(x)
                x = x + self.ca(x)
                x = self.act(x)
                return x

        def forward(self, x):
            return self.res(x)

    def UV_coding(self, UV, z_UV=None):
        UV1 = self.UVdown[0](UV)
        UV2 = self.UVdown[1](UV1)
        UV3 = self.UVdown[2](UV2)

        enc_UV0 = UV
        for blk in self.enc_UV[0]:
            enc_UV0 = blk(enc_UV0, z_UV)

        enc_UV1 = self.enc_UVdown[0](enc_UV0)
        enc_UV1 = enc_UV1 + UV1
        for blk in self.enc_UV[1]:
            enc_UV1 = blk(enc_UV1, z_UV)

        enc_UV2 = self.enc_UVdown[1](enc_UV1)
        enc_UV2 = enc_UV2 + UV2
        for blk in self.enc_UV[2]:
            enc_UV2 = blk(enc_UV2, z_UV)

        enc_UV3 = self.enc_UVdown[2](enc_UV2)
        enc_UV3 = enc_UV3 + UV3
        for blk in self.enc_UV[3]:
            enc_UV3 = blk(enc_UV3, z_UV)

        for blk in self.dec_UV[3]:
            dec3 = blk(enc_UV3, z_UV)

        dec2 = enc_UV2 + self.dec_UVup[2](dec3)
        for blk in self.dec_UV[2]:
            dec2 = blk(dec2, z_UV)

        dec1 = enc_UV1 + self.dec_UVup[1](dec2)
        for blk in self.dec_UV[1]:
            dec1 = blk(dec1, z_UV)

        dec0 = enc_UV0 + self.dec_UVup[0](dec1)
        for blk in self.dec_UV[0]:
            dec0 = blk(dec0, z_UV)

        return dec0

    def Y_coding(self, Yc, Ym, z_Y=None):

        if Ym is not None:
            Ym1 = self.Ydown[0](Ym)
            Ym2 = self.Ydown[1](Ym1)
            Ym3 = self.Ydown[2](Ym2)

            Ye, Ye1, Ye2, Ye3 = Ym, Ym1, Ym2, Ym3
            Yd, Yd1, Yd2, Yd3 = Ye, Ye1, Ye2, Ye3

        # Encoding
        enc_Y0 = Yc
        for blk in self.enc_Y[0]:
            if self.REF:
                enc_Y0, Ye = blk(enc_Y0, z_Y, Ye)
            else:
                enc_Y0 = blk(enc_Y0, z_Y)

        enc_Y1 = self.enc_Ydown[0](enc_Y0)
        for blk in self.enc_Y[1]:
            if self.REF:
                enc_Y1, Ye1 = blk(enc_Y1, z_Y, Ye1)
            else:
                enc_Y1 = blk(enc_Y1, z_Y)

        enc_Y2 = self.enc_Ydown[1](enc_Y1)
        for blk in self.enc_Y[2]:
            if self.REF:
                enc_Y2, Ye2 = blk(enc_Y2, z_Y, Ye2)
            else:
                enc_Y2 = blk(enc_Y2, z_Y)

        enc_Y3 = self.enc_Ydown[2](enc_Y2)
        for blk in self.enc_Y[3]:
            if self.REF:
                enc_Y3, Ye3 = blk(enc_Y3, z_Y, Ye3)
            else:
                enc_Y3 = blk(enc_Y3, z_Y)

        # Decoding
        for blk in self.dec_Y[3]:
            if self.REF:
                dec3, Yd3 = blk(enc_Y3, z_Y, Yd3)
            else:
                dec3 = blk(enc_Y3, z_Y)

        dec2 = enc_Y2 + self.dec_Yup[2](dec3)
        for blk in self.dec_Y[2]:
            if self.REF:
                dec2, Yd2 = blk(dec2, z_Y, Yd2)
            else:
                dec2 = blk(dec2, z_Y)

        dec1 = enc_Y1 + self.dec_Yup[1](dec2)
        for blk in self.dec_Y[1]:
            if self.REF:
                dec1, Yd1 = blk(dec1, z_Y, Yd1)
            else:
                dec1 = blk(dec1, z_Y)

        dec0 = enc_Y0 + self.dec_Yup[0](dec1)
        for blk in self.dec_Y[0]:
            if self.REF:
                dec0, Yd = blk(dec0, z_Y, Yd)
            else:
                dec0 = blk(dec0, z_Y)

        return dec0

    def forward(self, UVc, Yc, Ym, z_UV, z_Y):
        for blk in self.feat[0]:
            UVc = blk(UVc)
        for blk in self.feat[1]:
            Yc = blk(Yc)
        if Ym is not None:
            Ym = self.feat[2](Ym)

        dec_UV = self.UV_coding(UVc, z_UV)
        dec_Y = self.Y_coding(Yc, Ym, z_Y)
        dec = self.squ(torch.cat([dec_Y, dec_UV], dim=1))
        h_feat = self.drm(dec)
        h_map = self.out(h_feat)

        Y, U, V = torch.split(h_map, 1, dim=1)
        R = Y + 1.14 * V
        G = Y - 0.39 * U - 0.58 * V
        B = Y + 2.03 * U
        h_map = torch.cat([R, G, B], dim=1)

        return h_map


# ----------------------------------------------------------------------------------------------------------------------
class LDRM(nn.Module):
    def __init__(self, Ch_img=3, Ch_z=64, Channels=32, state='Test', REF=True, CODE=True, tests=False):
        super().__init__()
        self.state = state
        self.tests = tests
        self.REF = REF
        self.CODE = CODE

        self.ARN = AdditivityRectifyNet(ch_img=Ch_img, channels=Channels)

        if self.CODE:
            self.LCEM_UV = LatentCodeExtractModule(ch_z=Ch_z, space='UV', mode='Input')
            self.LCEM_Y = LatentCodeExtractModule(ch_z=Ch_z, space='Y', mode='Input')

            self.LCEM_UV_GT = LatentCodeExtractModule(ch_z=Ch_z, space='UV', mode='GT')
            self.LCEM_Y_GT = LatentCodeExtractModule(ch_z=Ch_z, space='Y', mode='GT')

            self.CEN = CodeEstimateNet()

        self.DRN = DegradationRectifyNet(ch_img=Ch_img, ch_z=Ch_z, ch_level=Channels, REF=REF, CODE=CODE)

    def state_train(self, color_img, mono_img=None, gt_img=None, mode='Pretrain'):
        # AdditivityRectifyNet
        if mode == 'Codetrain' and self.CODE:
            color_noise = self.ARN(color_img).detach()
        else:
            color_noise = self.ARN(color_img)
        color = color_img - color_noise

        # RGB => YUV
        R, G, B = torch.split(color, 1, dim=1)
        Uc = -0.147 * R - 0.289 * G + 0.436 * B
        Vc = 0.615 * R - 0.515 * G - 0.100 * B
        UVc = torch.cat([Uc, Vc], dim=1)
        Yc = 0.299 * R + 0.587 * G + 0.114 * B
        if mono_img is not None:
            Ym = torch.unsqueeze(mono_img[:, 0, :, :], dim=1)
        else:
            Ym = mono_img

        R, G, B = torch.split(gt_img, 1, dim=1)
        Uo = -0.147 * R - 0.289 * G + 0.436 * B
        Vo = 0.615 * R - 0.515 * G - 0.100 * B
        UVo = torch.cat([Uo, Vo], dim=1)
        Yo = 0.299 * R + 0.587 * G + 0.114 * B

        # Degradation Rectify
        if mode == 'Pretrain':
            if self.CODE:
                z_UV = self.LCEM_UV_GT(UVc, UVo)
                z_Y = self.LCEM_Y_GT((Yc, Ym), Yo)
                Batch = z_UV.shape[0]
                z_UV = z_UV.view(Batch, -1)
                z_Y = z_Y.view(Batch, -1)
            else:
                z_UV, z_Y = None, None
            h_map = self.DRN(UVc, Yc, Ym, z_UV, z_Y)
            restored_img = color * h_map
            return restored_img, h_map, color_noise, color

        elif mode == 'Codetrain' and self.CODE:
            z_UVf = self.LCEM_UV_GT(UVc, UVo).detach()
            z_Yf = self.LCEM_Y_GT((Yc, Ym), Yo).detach()

            Batch = z_Yf.shape[0]

            z_UVi = self.LCEM_UV(UVc)
            z_Yi = self.LCEM_Y((Yc, Ym))
            z_UV = self.CEN(z_UVi)
            z_Y = self.CEN(z_Yi)

            loss_z = nn.MSELoss()(z_UVf.view(Batch, -1), z_UV.view(Batch, -1)) \
                     + nn.MSELoss()(z_Yf.view(Batch, -1), z_Y.view(Batch, -1))

            return loss_z

        elif mode == 'Finetune' and self.CODE:
            # out:[B, C, 1, 1]
            z_UVf = self.LCEM_UV_GT(UVc, UVo).detach()
            z_Yf = self.LCEM_Y_GT((Yc, Ym), Yo).detach()

            Batch = z_Yf.shape[0]

            z_UVi = self.LCEM_UV(UVc)
            z_Yi = self.LCEM_Y((Yc, Ym))
            z_UV = self.CEN(z_UVi)
            z_Y = self.CEN(z_Yi)

            loss_z = nn.MSELoss()(z_UVf.view(Batch, -1), z_UV.view(Batch, -1)) \
                     + nn.MSELoss()(z_Yf.view(Batch, -1), z_Y.view(Batch, -1))

            h_map = self.DRN(UVc, Yc, Ym, z_UV, z_Y)
            restored_img = color * h_map

            return restored_img, h_map, color_noise, color, loss_z

    def state_test(self, color_img, mono_img=None):
        # Denoising
        color_noise = self.ARN(color_img)
        color = color_img - color_noise

        # RGB => YUV
        R, G, B = torch.split(color, 1, dim=1)
        Uc = -0.147 * R - 0.289 * G + 0.436 * B
        Vc = 0.615 * R - 0.515 * G - 0.100 * B
        UVc = torch.cat([Uc, Vc], dim=1)
        Yc = 0.299 * R + 0.587 * G + 0.114 * B
        Ym = torch.unsqueeze(mono_img[:, 0, :, :], dim=1)

        z_UVi = self.LCEM_UV(UVc)
        z_Yi = self.LCEM_Y((Yc, Ym))
        z_UV = self.CEN(z_UVi)
        z_Y = self.CEN(z_Yi)

        h_map = self.DRN(UVc, Yc, Ym, z_UV, z_Y)
        restored_img = color * h_map
        return restored_img, h_map, color

    def forward(self, color_img, mono_img=None, gt_img=None, mode='Pretrain'):
        if self.tests:
            if self.REF:
                mono_img = color_img
            gt_img = color_img
        if self.REF is not True:
            mono_img = None
        if self.state == 'Train':
            return self.state_train(color_img=color_img, mono_img=mono_img, gt_img=gt_img, mode=mode)
        else:
            return self.state_test(color_img=color_img, mono_img=mono_img)


# ======================================================================================================================
if __name__ == '__main__':
    def model_complex(model, input_shape):
        macs, params = get_model_complexity_info(model, input_shape, as_strings=True,
                                                 print_per_layer_stat=False, verbose=True)
        print(f'====> Number of Model Params: {params}')
        print(f'====> Computational complexity: {macs}')


    model = LDRM(3, state='Train', REF=True, CODE=True, tests=True)
    model_complex(model, (3, 160, 480))
